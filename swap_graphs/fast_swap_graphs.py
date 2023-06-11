import gc
from attrs import define, field
import einops
from functools import partial
from jaxtyping import Float, Int
import networkx as nx
import torch
import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from typing import Callable, Dict, List, Optional 

from swap_graphs.core import ActivationStore, CompMetric, ModelComponent, SwapGraph, print_gpu_mem
from swap_graphs.utils import KL_div_sim

@define
class FastActivationStore(ActivationStore):
    """Stores the activations of a model for a given dataset (the patched dataset), and create hooks to patch the activations of a given component (head, layer, etc)."""

    model: HookedTransformer = field(kw_only=True)
    dataset: Float[torch.Tensor, "batch pos"] = field(kw_only=True)
    listOfComponents: Optional[List[ModelComponent]] = field(kw_only=True, default=None)
    force_cache_all: bool = field(kw_only=True, default=False)
    comp_metric: CompMetric = field(kw_only=True)
    source_ids: List[Int] = field(kw_only=True)
    target_ids: List[Int] = field(kw_only=True)
    dataset_logits: Float[torch.Tensor, "batch pos vocab"] = field(init=False)
    activation_cache: Dict[str, torch.Tensor] | ActivationCache = field(init=False)
    grad_caches: Dict[str, torch.Tensor] = field(init=False)
    comparison_results: torch.Tensor = field(init=False)

    def compute_cache(self):
        self.model.reset_hooks()
        forward_cache = {}
        def forward_cache_hook(act, hook):
            forward_cache[hook.name] = act.detach().to("cuda")

        grad_cache = {}
        def backward_cache_hook(act, hook):
            grad_cache[hook.name] = act.detach().to("cuda")


        if self.listOfComponents is None or self.force_cache_all:
            dataset_logits, cache = self.model.run_with_cache(
                self.dataset
            )  # default, but memory inneficient
        else:
            for c in self.listOfComponents:
                self.model.add_hook(
                    c.hook_name, forward_cache_hook, "fwd"
                )
                self.model.add_hook(
                    c.hook_name, backward_cache_hook, "bwd"
                )

            dataset_logits = self.model(self.dataset)

        n_inputs = self.dataset.shape[0]
        batch_size = 294 # batch size must be a multiple of n_inputs - 1

        # TODO: make this work for dimensions of generic components
        self.grad_caches = {
            c.hook_name: torch.zeros((50, 21, 64), device="cuda")
            for c in self.listOfComponents
        }
        input_idx = 0

        for i in tqdm.tqdm(range(0, len(self.target_ids), batch_size)):
            source_idx = self.source_ids[
                i : min(i + batch_size, len(self.source_ids))
            ]  # the index that will send the cache, the once by which we patch
            target_idx = self.target_ids[
                i : min(i + batch_size, len(self.target_ids))
            ]  # The index of the datapoints that the majority of the model will run on

            comparison_result = self.comp_metric(
                    logits_target=dataset_logits[target_idx],
                    logits_source=dataset_logits[source_idx].clone().detach(),
                    target_seqs=self.dataset[target_idx],
                    target_idx=target_idx
            )

            for idx in range(0, comparison_result.shape[0], n_inputs-1):
                results_for_token = comparison_result[idx: idx+n_inputs-1]
                results_for_token.mean().backward(retain_graph= True)
                for c in self.listOfComponents:
                    self.grad_caches[c.hook_name][input_idx] = grad_cache[c.hook_name][input_idx,:,c.head,:]
                grad_cache = {}
                input_idx += 1
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.zero_()

        self.activation_cache = ActivationCache(forward_cache, self.model)
        self.dataset_logits = dataset_logits  # type: ignore

    def getPatchingHooksByIdx(
        self,
        source_idx: List[int],
        target_idx: List[int],
        verbose: bool = False,
        list_of_components: Optional[List[ModelComponent]] = None,
    ):
        pass

    # TODO: Make this work for multiple components
    def get_activations_and_gradients(
            self: "FastActivationStore",
            source_idx: int | List[int],
            target_idx: int | List[int],
    ):
        assert self.listOfComponents is not None

        component = self.listOfComponents[0]
        source_position = component.position.positions_from_idx(source_idx)
        target_position = component.position.positions_from_idx(target_idx)
        a_source = self.activation_cache[component.hook_name][source_idx,source_position,component.head]
        a_target = self.activation_cache[component.hook_name][target_idx,target_position,component.head]
        grad_target = self.grad_caches[component.hook_name][target_idx,target_position]
        
        return a_source, a_target, grad_target

        


class FastSwapGraph(SwapGraph):
    def build(
        self,
        additional_info_gathering: Optional[Callable] = None,
        verbose: bool = False,
        progress_bar: bool = True,
    ):
        torch.set_grad_enabled(True)
        edges = []
        source_IDs = []
        target_IDs = []
        for target_id in range(len(self.tok_dataset)):
            for source_id, y in enumerate(self.tok_dataset):
                if torch.rand(1) > self.proba_edge:
                    continue
                if source_id == target_id:
                    continue
                source_IDs.append(source_id)
                target_IDs.append(target_id)
        if verbose:
            print(f"Number of edges: {len(edges)}")

        weights = compute_batched_weights_fast(
            self.model,
            self.tok_dataset,
            source_IDs,
            target_IDs,
            self.batch_size,
            self.patchedComponents,
            self.comp_metric,
            additional_info_gathering,
            verbose,
            progress_bar=progress_bar,
        ).tolist()

        self.raw_edges = list(
            zip(source_IDs, target_IDs, weights)
        )  # the raw edges, the ones with the output from the comparison metric. Before plotting the edges need to go through a post-processing step to get the weight of the graph.
        self.all_comp_metrics = [x[2] for x in self.raw_edges]

        self.G = nx.DiGraph()
        for i in range(len(self.tok_dataset)):
            self.G.add_node(i, label=str(i))

        for u, v, w in self.raw_edges:
            self.G.add_edge(
                u,
                v,
                weight=w,
            )
        torch.set_grad_enabled(False)


def compute_batched_weights_fast(
    model: HookedTransformer,
    dataset: Float[torch.Tensor, "batch pos"],
    source_IDs: List[int],
    target_IDs: List[int],
    batch_size: int,
    components_to_patch: List[ModelComponent],
    comp_metric: CompMetric,
    additional_info_gathering: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    verbose: bool = False,
    activation_store: Optional[FastActivationStore] = None,
    progress_bar: bool = True,
):
    if activation_store is None:
        activation_store = FastActivationStore(
            model=model, 
            dataset=dataset, 
            listOfComponents=components_to_patch,
            comp_metric=comp_metric, # type: ignore
            source_ids=source_IDs,
            target_ids=target_IDs,
        )
    
    all_weights = []

    # get A(source_idx) and A(target_idx) from activation store
    # get grad for target index from activation store
    a_source, a_target, grad_target = activation_store.get_activations_and_gradients(source_IDs, target_IDs)

    # componentwise multiplication: target_grad * (A(source_idx) - A(target_idx))
    # sum over pos and d_head dimension
    weights = grad_target * (a_source - a_target)
    weights = torch.where(weights > 0.0, weights, torch.zeros_like(weights))
    weigths = einops.reduce(weights, "... d_head -> ...", "sum")

    return weigths

