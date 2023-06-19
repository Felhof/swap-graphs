import einops
import numpy as np
import tqdm
import gc
from swap_graphs.core import CompMetric, ModelComponent, WildPosition, component_patching_hook
from swap_graphs.utils import get_components_at_position

import torch
from attrs import define, field
from jaxtyping import Float, Int
from transformer_lens import ActivationCache, HookedTransformer


from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, List, Optional


@define
class BaseActivationStore(ABC):
    model: HookedTransformer = field(kw_only=True)
    dataset: Float[torch.Tensor, "batch pos"] = field(kw_only=True)
    dataset_logits: Float[torch.Tensor, "batch pos vocab"] = field(init=False)
    listOfComponents: Optional[List[ModelComponent]] = field(kw_only=True, default=None)

    @abstractmethod
    def compute_cache(self):
        pass

    def __attrs_post_init__(self):
        self.compute_cache()


@define
class ActivationStore(BaseActivationStore):
    """Stores the activations of a model for a given dataset (the patched dataset), and create hooks to patch the activations of a given component (head, layer, etc)."""

    force_cache_all: bool = field(kw_only=True, default=False)
    transformerLensCache: Dict[str, torch.Tensor] | ActivationCache = field(init=False)

    def compute_cache(self):
        if self.listOfComponents is None or self.force_cache_all:
            dataset_logits, cache = self.model.run_with_cache(
                self.dataset
            )  # default, but memory inneficient
        else:
            cache = {}

            def save_hook(tensor, hook):
                cache[hook.name] = tensor.detach().to("cuda")

            dataset_logits = (
                self.model.run_with_hooks(  # only cache the components we need
                    self.dataset,
                    fwd_hooks=[(c.hook_name, save_hook) for c in self.listOfComponents],
                )
            )
        self.transformerLensCache = cache
        self.dataset_logits = dataset_logits  # type: ignore

    def getPatchingHooksByIdx(
        self,
        source_idx: List[int],
        target_idx: List[int],
        verbose: bool = False,
        list_of_components: Optional[List[ModelComponent]] = None,
    ):
        """Create a list of hook function where the cache is computed from the stored dataset cache on the indices idx."""
        assert source_idx is not None
        assert max(source_idx) < self.dataset.shape[0]
        patchingHooks = []

        if (
            list_of_components is None
        ):  # TODO : quite dirty, remove the listOfComponents attribute
            list_of_components = self.listOfComponents

        assert list_of_components is not None

        for component in list_of_components:
            patchingHooks.append(
                (
                    component.hook_name,
                    partial(
                        component_patching_hook,
                        component=component,
                        cache=self.transformerLensCache[component.hook_name][
                            source_idx
                        ],
                        source_idx=source_idx,
                        target_idx=target_idx,
                        verbose=verbose,
                    ),
                )
            )

        return patchingHooks

    def change_component_list(self, new_list):
        """Change the list of components to patch. Update the cache accordingly (only when needed)."""
        if self.listOfComponents is not None and not self.force_cache_all:
            if [c.hook_name for c in new_list] != [
                c.hook_name for c in self.listOfComponents
            ]:
                self.listOfComponents = new_list
                self.compute_cache()  # only recompute when the list changed and when the cache is partial
        self.listOfComponents = new_list


@define
class FastActivationStore(BaseActivationStore):
    """Stores the activations of a model for a given dataset (the patched dataset), and create hooks to patch the activations of a given component (head, layer, etc)."""

    comp_metric: CompMetric = field(kw_only=True)
    source_ids: List[Int] = field(kw_only=True)
    target_ids: List[Int] = field(kw_only=True)
    position_to_patch: Optional[WildPosition] = field(kw_only=True, default=None)
    include_mlp: bool = field(kw_only=True, default=True)
    head_subpart: str = field(kw_only=True, default="z")
    activation_cache: Dict[str, torch.Tensor] | ActivationCache = field(init=False)
    grad_cache: Dict[str, torch.Tensor] = field(init=False)
    comparison_results: torch.Tensor = field(init=False)
    position_to_idx: np.ndarray = field(init=False)
    
    
    def compute_cache(self):
        if self.listOfComponents is None:
            assert self.position_to_patch is not None, "Must either provide list of components or positions to patch to FastActivationStore"
            self.listOfComponents = get_components_at_position(
                position=self.position_to_patch,
                nb_layers=self.model.cfg.n_layers,
                nb_heads=self.model.cfg.n_heads,
                include_mlp=self.include_mlp,
                head_subpart=self.head_subpart,
            ) 

        self.model.reset_hooks()
        
        forward_cache = {}
        def forward_cache_hook(act, hook):
            forward_cache[hook.name] = act.detach().to("cuda")

        self.grad_cache = {}
        def backward_cache_hook(act, hook):
            self.grad_cache[hook.name] = act.detach().to("cuda")

        for c in self.listOfComponents:
            self.model.add_hook(
                c.hook_name, forward_cache_hook, "fwd"
            )
            self.model.add_hook(
                c.hook_name, backward_cache_hook, "bwd"
            )

        positions = self.listOfComponents[0].position.position
        position_set = set(positions)
        unique_positions = list(position_set)
        self.position_to_idx = np.zeros(max(position_set) + 1)
        for idx, pos in enumerate(unique_positions):
            self.position_to_idx[pos] = idx

        dataset_logits = self.model(self.dataset)[:,unique_positions,:]

        self.activation_cache = ActivationCache(forward_cache, self.model)
        self.dataset_logits = dataset_logits.clone()  # type: ignore
        del dataset_logits


    # TODO: Make this work for multiple components
    def get_weights(self: "FastActivationStore"):
        assert self.listOfComponents is not None
        weights = {
            c.hook_name : [] for c in self.listOfComponents
        }

        n_inputs = self.dataset.shape[0]
        batch_size = n_inputs - 1 # batch size must be a multiple of n_inputs - 1

        for i in tqdm.tqdm(range(0, len(self.target_ids), batch_size)):
            source_idx = self.source_ids[
                i : min(i + batch_size, len(self.source_ids))
            ]  # the index that will send the cache, the once by which we patch
            target_idx = self.target_ids[
                i : min(i + batch_size, len(self.target_ids))
            ]  # The index of the datapoints that the majority of the model will run on

            gc.collect()
            torch.cuda.empty_cache()

            position = self.listOfComponents[0].position

            logits_target=self.dataset_logits[target_idx, self.position_to_idx[position.positions_from_idx(target_idx)]]
            logits_source=self.dataset_logits[source_idx, self.position_to_idx[position.positions_from_idx(target_idx)]]

            comparison_result = self.comp_metric(
                    logits_target=logits_target,
                    logits_source=logits_source.detach(),
            )

            comparison_result.mean().backward(retain_graph=True)
            for c in self.listOfComponents:
                source_position = c.position.positions_from_idx(source_idx)
                target_position = c.position.positions_from_idx(target_idx)

                a_source = self.activation_cache[c.hook_name][source_idx,source_position,c.head]
                a_target = self.activation_cache[c.hook_name][target_idx,target_position,c.head]

                grad_target = self.grad_cache[c.hook_name][target_idx,target_position,c.head]

                target_weights = grad_target * (a_source - a_target)
                target_weights = torch.where(target_weights >= 0., target_weights, torch.zeros_like(target_weights))
                target_weights = einops.reduce(target_weights, "... d_head -> ...", "sum")

                weights[c.hook_name].extend(target_weights.tolist())


            self.grad_cache = {}
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.zero_()

        return weights