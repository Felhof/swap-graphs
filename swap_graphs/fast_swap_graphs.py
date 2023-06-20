import gc
from attrs import define, field
import einops
from functools import partial
from jaxtyping import Float, Int
import networkx as nx
import numpy as np
import torch
import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from typing import Callable, Dict, List, Optional 

from swap_graphs.core import FastActivationStore, CompMetric, ModelComponent, SwapGraph, print_gpu_mem
from swap_graphs.utils import KL_div_sim

# TODO: Currently this assumes that we are patching all components at the same positions
# @define
# class FastActivationStore(ActivationStore):
#     """Stores the activations of a model for a given dataset (the patched dataset), and create hooks to patch the activations of a given component (head, layer, etc)."""

#     model: HookedTransformer = field(kw_only=True)
#     dataset: Float[torch.Tensor, "batch pos"] = field(kw_only=True)
#     listOfComponents: Optional[List[ModelComponent]] = field(kw_only=True, default=None)
#     force_cache_all: bool = field(kw_only=True, default=False)
#     comp_metric: CompMetric = field(kw_only=True)
#     source_ids: List[Int] = field(kw_only=True)
#     target_ids: List[Int] = field(kw_only=True)
#     dataset_logits: Float[torch.Tensor, "batch pos vocab"] = field(init=False)
#     activation_cache: Dict[str, torch.Tensor] | ActivationCache = field(init=False)
#     grad_cache: Dict[str, torch.Tensor] = field(init=False)
#     comparison_results: torch.Tensor = field(init=False)
#     position_to_idx: np.ndarray = field(init=False)

#     def compute_cache(self):
#         self.model.reset_hooks()
#         forward_cache = {}
#         def forward_cache_hook(act, hook):
#             forward_cache[hook.name] = act.detach().to("cuda")

#         self.grad_cache = {}
#         def backward_cache_hook(act, hook):
#             self.grad_cache[hook.name] = act.detach().to("cuda")


#         if self.listOfComponents is None or self.force_cache_all:
#             dataset_logits, cache = self.model.run_with_cache(
#                 self.dataset
#             )  # default, but memory inneficient
#         else:
#             for c in self.listOfComponents:
#                 self.model.add_hook(
#                     c.hook_name, forward_cache_hook, "fwd"
#                 )
#                 self.model.add_hook(
#                     c.hook_name, backward_cache_hook, "bwd"
#                 )

#             positions = self.listOfComponents[0].position.position
#             position_set = set(positions)
#             unique_positions = list(position_set)
#             self.position_to_idx = np.zeros(max(position_set) + 1)
#             for idx, pos in enumerate(unique_positions):
#                 self.position_to_idx[pos] = idx

#             dataset_logits = self.model(self.dataset)[:,unique_positions,:]

#         self.activation_cache = ActivationCache(forward_cache, self.model)
#         self.dataset_logits = dataset_logits.clone()  # type: ignore
#         del dataset_logits    


#     def getPatchingHooksByIdx(
#         self,
#         source_idx: List[int],
#         target_idx: List[int],
#         verbose: bool = False,
#         list_of_components: Optional[List[ModelComponent]] = None,
#     ):
#         pass

#     # TODO: Make this work for multiple components
#     def get_weights(self: "FastActivationStore"):
#         assert self.listOfComponents is not None
#         weights = {
#             c.hook_name : [] for c in self.listOfComponents
#         }

#         n_inputs = self.dataset.shape[0]
#         batch_size = n_inputs - 1 # batch size must be a multiple of n_inputs - 1
        
#         for i in tqdm.tqdm(range(0, len(self.target_ids), batch_size)):
#             source_idx = self.source_ids[
#                 i : min(i + batch_size, len(self.source_ids))
#             ]  # the index that will send the cache, the once by which we patch
#             target_idx = self.target_ids[
#                 i : min(i + batch_size, len(self.target_ids))
#             ]  # The index of the datapoints that the majority of the model will run on

#             gc.collect()
#             torch.cuda.empty_cache()

#             position = self.listOfComponents[0].position

#             logits_target=self.dataset_logits[target_idx, self.position_to_idx[position.positions_from_idx(target_idx)]]
#             logits_source=self.dataset_logits[source_idx, self.position_to_idx[position.positions_from_idx(target_idx)]]
            
#             comparison_result = self.comp_metric(
#                     logits_target=logits_target,
#                     logits_source=logits_source.detach(),
#             )

#             comparison_result.mean().backward(retain_graph=True)
#             for c in self.listOfComponents:
#                 source_position = c.position.positions_from_idx(source_idx)
#                 target_position = c.position.positions_from_idx(target_idx)
                
#                 a_source = self.activation_cache[c.hook_name][source_idx,source_position,c.head]
#                 a_target = self.activation_cache[c.hook_name][target_idx,target_position,c.head]

#                 grad_target = self.grad_cache[c.hook_name][target_idx,target_position,c.head]

#                 target_weights = grad_target * (a_source - a_target)
#                 target_weights = torch.where(target_weights >= 0., target_weights, torch.zeros_like(target_weights))
#                 target_weights = einops.reduce(target_weights, "... d_head -> ...", "sum")

#                 weights[c.hook_name].extend(target_weights.tolist())


#             self.grad_cache = {}
#             for param in self.model.parameters():
#                 if param.grad is not None:
#                     param.grad.zero_()
        
#         return weights

        
class FastSwapGraph(SwapGraph):
    def build(
        self,
        activation_store: FastActivationStore,
    ):
        weights = activation_store.get_weights(str(self.patchedComponents[0]))

        self.raw_edges = list(
            zip(activation_store.source_ids, activation_store.target_ids, weights)
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
