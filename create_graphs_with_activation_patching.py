# %%
import inspect, os, sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# %%
from functools import partial

# from sympy import expand, symbols
import plotly.express as px
import torch
import tqdm.auto as tqdm
from swap_graphs.core import (
    CompMetric,
    ModelComponent,
    SgraphDataset,
    SwapGraph,
    WildPosition,
)
from swap_graphs.fast_swap_graphs import FastSwapGraph
from swap_graphs.datasets.ioi.ioi_dataset import NAMES_GENDER, IOIDataset
from swap_graphs.datasets.ioi.ioi_utils import get_ioi_features_dict
from swap_graphs.utils import KL_div_sim, plotHistLogLog
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
import numpy as np

torch.set_grad_enabled(False)
# %%
### Install the model
model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

# %%
### Create an IOI dataset with 50 sequences and 5 possible names

ioi_dataset = IOIDataset(N=100, seed=42, nb_names=5)
# %%
for s in ioi_dataset.prompts_text[:10]:
  print(s)
# %%

### Parse IOI dataset to extract input features

def get_ioi_features_dict_toy(ioi_dataset: IOIDataset) -> dict[str, list[str]]:
  f_dict = {}
  f_dict["IO token"] = [metadata["IO"] for metadata in ioi_dataset.prompts_metadata]
  f_dict["object"] = [metadata["[OBJECT]"] for metadata in ioi_dataset.prompts_metadata]
  return f_dict
# %%
feature_dict = get_ioi_features_dict(ioi_dataset)
# %%
for k in feature_dict:
  print(f"Feature: '{k}' -- 10 first values: {feature_dict[k][:10]}")
# %%
sgraph_dataset = SgraphDataset(
    tok_dataset=ioi_dataset.prompts_tok,
    str_dataset=ioi_dataset.prompts_text,
    feature_dict=feature_dict,
)
# %%
### Comparison Metric: KL Divergence

PATCHED_POSITION = "END"
position = WildPosition(ioi_dataset.word_idx[PATCHED_POSITION], label=PATCHED_POSITION)

comp_metric: CompMetric = partial(
    KL_div_sim,
    position_to_evaluate=position,  
)

# %%
def create_swap_graph_with_activation_patching(
        layer,
        head,
        name,
        plot_histograms=True, 
        compute_communities=True,
        PATCHED_POSITION = "END",
        abstract_variable = "IO token"
    ):

    position = WildPosition(ioi_dataset.word_idx[PATCHED_POSITION], label=PATCHED_POSITION)

    comp_metric: CompMetric = partial(
        KL_div_sim,
        position_to_evaluate=position,  
    )

    sgraph = SwapGraph(
        model=model,
        tok_dataset=ioi_dataset.prompts_tok,
        comp_metric=comp_metric,
        batch_size=300,
        proba_edge=1.0, # proportion of pairs of inputs to run swaps on.
        patchedComponents=[
            ModelComponent(
                position=position,
                layer=layer,
                head=head,
                name=name,
            )
        ],
    )

    sgraph.build(verbose=False)
    sgraph.compute_weights()

    if plot_histograms:
        plotHistLogLog(
            sgraph.all_comp_metrics, only_y_log=False, metric_name="KL divergence histogram"
        )
        plotHistLogLog(
            sgraph.all_weights,
            only_y_log=False,
            metric_name="KL divergence-based graph weights",
        )

    if compute_communities:
        com_cmap = sgraph.compute_communities()

        # We use the Louvain communities to compute the adjusted rand index with the features from sgraph_dataset
        metrics = sgraph_dataset.compute_feature_rand(sgraph)

        for f in metrics["rand"]:
            print(f"Feature: {f} - Adjusted Rand index: {metrics['rand'][f]:2f}")
        print()

        title = f"{sgraph.patchedComponents[0]} swap graph. gpt2-small. <br>Adjused Rand index with '{abstract_variable}': {metrics['rand'][abstract_variable]:.2f} "
    else:
        title = f"{sgraph.patchedComponents[0]} swap graph. gpt2-small."
        sgraph.commu_labels = {}
        for i in range(2450):
            sgraph.commu_labels[i] = 0 


    fig = sgraph.show_html(
        title=title,
        sgraph_dataset=sgraph_dataset,
        feature_to_show="all",
        display=False,
        recompute_positions=True,
        iterations=1000,
    )

    fig.update_layout(height=700, width=700)

    fig.show()
# %%
create_swap_graph_with_activation_patching(
   layer=9,
   head=9,
   name="z"
)
# %%
create_swap_graph_with_activation_patching(
    layer=9,
    head=9,
    name="q",
    compute_communities=True,
    abstract_variable="Order of first names"
)
# %%
create_swap_graph_with_activation_patching(
    layer=8,
    head=6,
    name="z",
)
# %%
create_swap_graph_with_activation_patching(
    layer=5,
    head=5,
    name="z",
    PATCHED_POSITION="S2"
)

# %%
