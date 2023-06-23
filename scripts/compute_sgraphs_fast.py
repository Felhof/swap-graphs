import inspect, os, sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from copy import deepcopy
from functools import partial
import os
import torch
from typing import Literal
import time

import fire
from names_generator import generate_name
import numpy as np
from transformer_lens import HookedTransformer
from tqdm import tqdm

from swap_graphs.fast_swap_graphs import FastSwapGraph
from swap_graphs.datasets.ioi.ioi_dataset import (
    NAMES_GENDER,
    IOIDataset,
    check_tokenizer,
)
from swap_graphs.datasets.ioi.ioi_utils import (
    get_ioi_features_dict,
    logit_diff,
    logit_diff_comp,
    probs,
    assert_model_perf_ioi,
)
from swap_graphs.core import (
    FastActivationStore,
    CompMetric,
    ModelComponent,
    SwapGraph,
    WildPosition,
    find_important_components,
    SgraphDataset,
    compute_clustering_metrics,
    get_components_at_position,
)
from swap_graphs.utils import (
    KL_div_pos,
    KL_div_sim,
    L2_dist,
    L2_dist_in_context,
    imshow,
    line,
    plotHistLogLog,
    print_gpu_mem,
    save_object,
    scatter,
    show_attn,
    load_object,
    wrap_str,
    show_mtx,
)


def auto_sgraph(
    model_name: str,
    head_subpart: str = "z",
    include_mlp: bool = True,
    proportion_to_sgraph: float = 1.0,
    batch_size: int = 200,
    batch_size_sgraph: int = 200,
    nb_sample_eval: int = 200,
    nb_datapoints_sgraph: int = 100,
    xp_path: str = "../xp",
    dataset_name: Literal["IOI"] = "IOI",
):
    assert dataset_name in [
        "IOI",
    ], "dataset_name must be IOI"

    COMP_METRIC = "KL"
    PATCHED_POSITION = "END"

    torch.set_grad_enabled(True)
    model = HookedTransformer.from_pretrained(model_name, device="cuda")

    if dataset_name == "IOI":
        assert check_tokenizer(
            model.tokenizer
        ), "The tokenizer is tokenizing some word into two tokens."
        dataset = IOIDataset(
            N=nb_datapoints_sgraph,
            seed=42,
            wild_template=False,
            nb_names=5,
            tokenizer=model.tokenizer,
        )
        assert_model_perf_ioi(model, dataset)

        feature_dict = get_ioi_features_dict(dataset)
        sgraph_dataset = SgraphDataset(
            tok_dataset=dataset.prompts_tok,
            str_dataset=dataset.prompts_text,
            feature_dict=feature_dict,
        )

    else:
        raise ValueError("Unknown dataset_name")

    position = WildPosition(
            dataset.word_idx[PATCHED_POSITION], label=PATCHED_POSITION
    )

    if COMP_METRIC == "KL":
        comp_metric: CompMetric = KL_div_pos # type: ignore
    elif COMP_METRIC == "LDiff":
        comp_metric: CompMetric = partial(logit_diff_comp, ioi_dataset=dataset, keep_sign=True)  # type: ignore
    else:
        raise ValueError("Unknown comp_metric")

    components = get_components_at_position(
        position=position,
        nb_layers=model.cfg.n_layers,
        nb_heads=model.cfg.n_heads,
        include_mlp=include_mlp,
        head_subpart=head_subpart,
    )

    if not os.path.exists(xp_path):
        os.mkdir(xp_path)

    xp_name = (
        model_name.replace("/", "-")
        + "-"
        + head_subpart
        + "-"
        + dataset_name
        + "-"
        + generate_name(seed=int(time.clock_gettime(0)))
    )
    xp_path = os.path.join(xp_path, xp_name)
    os.mkdir(xp_path)

    fig_path = os.path.join(xp_path, "figs")
    os.mkdir(fig_path)

    print(f"Experiment name: {xp_name} -- Experiment path: {xp_path}")

    date = time.strftime("%Hh%Mm%Ss %d-%m-%Y")  # add time stamp to the experiments
    open(os.path.join(xp_path, date), "a").close()

    config = {}
    config["model_name"] = model_name
    config["head_subpart"] = head_subpart
    config["include_mlp"] = include_mlp
    config["proportion_to_sgraph"] = proportion_to_sgraph
    config["batch_size"] = batch_size
    config["batch_size_sgraph"] = batch_size_sgraph
    config["nb_sample_eval"] = nb_sample_eval
    config["nb_datapoints_sgraph"] = nb_datapoints_sgraph
    config["xp_path"] = xp_path
    config["xp_name"] = xp_name
    config["dataset_name"] = dataset_name
    config["COMP_METRIC"] = COMP_METRIC
    config["PATCHED_POSITION"] = PATCHED_POSITION
    config["date"] = date
    save_object(config, xp_path, "config.pkl")

    save_object(sgraph_dataset, xp_path, "sgraph_dataset.pkl")
    save_object(dataset, xp_path, "dataset.pkl")

    activation_store = FastActivationStore(
        model=model,
        dataset=dataset.prompts_tok,
        comp_metric=comp_metric,
        position_to_patch=position,
    )

    all_data = {}
    results = []
    for i in tqdm(range(len(components))):
        c = components[i]
        results.append(torch.tensor(activation_store.get_weights(str(c))))
        sgraph = FastSwapGraph(
            model=model,
            tok_dataset=dataset.prompts_tok,
            comp_metric=comp_metric,
            batch_size=batch_size_sgraph,
            proba_edge=1.0,
            patchedComponents=[c],
        )
        sgraph.build(activation_store=activation_store)
        sgraph.compute_weights()
        sgraph.compute_communities()

        component_data = {}
        component_data["clustering_metrics"] = compute_clustering_metrics(sgraph)
        component_data["feature_metrics"] = sgraph_dataset.compute_feature_rand(sgraph)
        component_data["sgraph_edges"] = sgraph.raw_edges
        component_data["commu"] = sgraph.commu_labels

        # deepcopy the component data
        all_data[str(c)] = deepcopy(component_data)

        # create html plot for the graph
        largest_rand_feature, max_rand_idx = max(
            component_data["feature_metrics"]["rand"].items(), key=lambda x: x[1]
        )
        title = wrap_str(
            f"<b>{sgraph.patchedComponents[0]}</b> Average CompMetric: {np.mean(sgraph.all_comp_metrics):.2f} (#{components.index(c)}), Rand idx commu-{largest_rand_feature}: {max_rand_idx:.2f}, modularity: {component_data['clustering_metrics']['modularity']:.2f}",
            max_line_len=70,
        )

        sgraph.show_html(
            sgraph_dataset,
            feature_to_show="all",
            title=title,
            display=False,
            save_path=fig_path,
            color_discrete=True,
        )
        if i % 10 == 0:  # save every 10 iterations
            save_object(all_data, xp_path, "all_data.pkl")
    save_object(all_data, xp_path, "all_data.pkl")

    if include_mlp:
        sec_dim = model.cfg.n_heads + 1
    else:
        sec_dim = model.cfg.n_heads

    save_object(
        torch.cat(results).reshape(model.cfg.n_layers, sec_dim, (nb_datapoints_sgraph - 1) * nb_datapoints_sgraph),
        xp_path,
        "comp_metric.pkl",
    )

if __name__ == "__main__":
    fire.Fire(auto_sgraph)
