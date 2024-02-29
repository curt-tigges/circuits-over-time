#%%
from functools import partial
import argparse
import yaml
import json
import os

from transformer_lens import HookedTransformer
import torch
from torch.utils.data import Dataset, DataLoader

from edge_attribution_patching.graph import  Graph
from edge_attribution_patching.attribute_mem import attribute 
from edge_attribution_patching.evaluate_graph import evaluate_graph, evaluate_baseline
from edge_attribution_patching.utils import kl_div

from utils.model_utils import load_model
from utils.data_utils import UniversalPatchingDataset
from utils.metrics import (
    CircuitMetric,
    compute_logit_diff,
    compute_probability_diff,
)
#%%

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download & assess model checkpoints")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to config file",
    )
    parser.add_argument(
        "-t",
        "--task",
        default="ioi",
        help="Name of task dataset for which to find the circuit",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="pythia-160m",
        help="Name of model to load",
    )
    parser.add_argument(
        "-e",
        "--eval_metric",
        default="logit_diff",
        help="Name of metric to use for EAP evaluation",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "-l",
        "--large_model",
        default=False,
        help="Whether to load a large model",
    )
    parser.add_argument(
        "-cs",
        "--ckpt_schedule",
        default="other",
        help="Checkpoint schedule over which to iterate",
    )
    parser.add_argument(    
        "-cd",
        "--cache_dir",
        default="model_cache",
        help="Directory for cache",
    )
    parser.add_argument(
        "-tn",
        "--top_n",
        default=400,
        help="Number of edges to keep in the graph",
    )
    parser.add_argument(
        "-v",
        "--verify",
        default=False,
        help="Whether to get the faithfulness curve for the graph",
    )   
    return parser.parse_args()


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def process_args():
    # Returns a namespace of arguments either from a config file or from the command line
    args = get_args()
    if args.config is not None:
        config = read_config(args.config)
        for key, value in config.items():
            setattr(args, key, value)
    # Placeholder to revisit when we want to add different model seed variants
    setattr(args, "canonical_model", True)
    return args


def get_ckpts(schedule):
    if schedule == "linear":
        ckpts = [i * 1000 for i in range(1, 144)]
    elif schedule == "exponential":
        ckpts = [
            round((2**i) / 1000) * 1000 if 2**i > 1000 else 2**i
            for i in range(18)
        ]
    elif schedule == "exp_plus_detail":
        ckpts = (
            [2**i for i in range(10)]
            + [i * 1000 for i in range(1, 16)]
            + [i * 5000 for i in range(3, 14)]
            + [i * 10000 for i in range(7, 15)]
        )
    else:
        ckpts = [1, 143000]

    return ckpts


def collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.stack([item[key] for item in batch])
    return batch_dict


def metric_mapper(metric_name):
    if metric_name == "logit_diff":
        return compute_logit_diff
    elif metric_name == "prob_diff":
        return compute_probability_diff
    #elif metric_name == "kl_div":
    #    return compute_kl_divergence
    #elif metric_name == "js_div":
    #    return compute_js_divergence
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")
                         

def get_data_and_metrics(
        model: HookedTransformer,
        task_name: str,
        eap: bool=True,
    ):
    assert task_name in ["ioi", "greater_than", "sentiment_cont", "sentiment_class", "mood_sentiment"]

    if task_name == "ioi":
        ds = UniversalPatchingDataset.from_ioi(model, 70)
        logit_diff_metric = partial(compute_logit_diff,mode='simple')
        metric = CircuitMetric("logit_diff", logit_diff_metric, eap = eap)

    elif task_name == "greater_than":
        # Get data
        ds = UniversalPatchingDataset.from_greater_than(model, 200)
        prob_diff_metric = partial(
            compute_probability_diff, 
            mode="group_sum"
        )
        metric = CircuitMetric("prob_diff", prob_diff_metric, eap = eap)

    elif task_name == "sentiment_cont":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "cont")
        logit_diff_metric = partial(compute_logit_diff, mode="pairs")
        metric = CircuitMetric("logit_diff", logit_diff_metric, eap = eap)

    elif task_name == "sentiment_class":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "class")
        logit_diff_metric = partial(compute_logit_diff,  mode="pairs")
        metric = CircuitMetric("logit_diff", logit_diff_metric, eap = eap)

    return ds, metric


def get_faithfulness_metrics(
        graph: Graph,
        model: HookedTransformer, 
        dataloader: DataLoader, 
        metric: CircuitMetric,
        baseline: float,
        start: int = 100,
        end: int = 1000,
        step: int = 100,
    ):

    faithfulness = dict()

    for size in range(start, end, step):
        graph.apply_greedy(size, absolute=True)
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
        faithfulness[size] = (evaluate_graph(model, graph, dataloader, metric).mean() / baseline).item()

    return faithfulness

#%%
def main(args):
    print(f"Arguments: {args}")
    schedule = args.ckpt_schedule
    ckpts = get_ckpts(schedule)

    for ckpt in ckpts:

        print(f"Loading model for step {ckpt}...")
        if args.large_model or args.canonical_model:
            model = HookedTransformer.from_pretrained(
                args.model, 
                checkpoint_value=int(ckpt),
                center_unembed=False,
                center_writing_weights=False,
                fold_ln=False,
                dtype=torch.bfloat16,
                **{"cache_dir": args.cache_dir},
            )
        else:
            ckpt_key = f"step{ckpt}"
            # TODO: Add support for different model seeds
            model = load_model(args.model, args.model, ckpt_key, args.cache_dir)
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        
        # Set up for task 
        task = args.task
        ds, metric = get_data_and_metrics(model, task, eap=True)
        graph = Graph.from_model(model)
        dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)
        
        # Evaluate baseline and graph
        baseline = evaluate_baseline(model, dataloader, metric).mean()
        print(f"Baseline metric value for {args.task}: {baseline}")
        attribute(model, graph, dataloader, partial(metric, loss=True), integrated_gradients=30)
        graph.apply_greedy(args.top_n)
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
        results = evaluate_graph(model, graph, dataloader, metric).mean()
        faithfulness[args.top_n] = (results / baseline).item()
        print(results)


        faithfulness = None

        if args.verify:
            faithfulness = get_faithfulness_metrics(graph, model, dataloader, metric, baseline, start=25, end=1600, step=25)
            print(faithfulness)

        # Save graph and results
        os.makedirs(f"results/graphs/{args.model}/{task}", exist_ok=True)
        os.makedirs(f"results/images/{args.model}/{task}", exist_ok=True)
        os.makedirs(f"results/faithfulness/{args.model}/{task}", exist_ok=True)
        graph.to_json(f'results/graphs/{args.model}/{task}/{ckpt}.json')
        gz = graph.to_graphviz()
        gz.draw(f'results/images/{args.model}/{task}/{ckpt}.png', prog='dot')

        if args.verify:
            with open(f"results/faithfulness/{args.model}/{task}/{args.ckpt}.json", "w") as f:
                json.dump(faithfulness, f)

if __name__ == "__main__":
    args = process_args()
    main(args)

# %%
# from transformer_lens import HookedTransformer
# # Set up for task 
# task = "ioi"

# model = HookedTransformer.from_pretrained(
#             'pythia-160m', 
#             #checkpoint_value=143000,
#             center_unembed=False,
#             center_writing_weights=False,
#             fold_ln=False,
#             dtype=torch.bfloat16
#         )
# ds, metric = get_data_and_metrics(model, task, eap=True)
# graph = Graph.from_model(model)
# dataloader = DataLoader(ds, batch_size=8, collate_fn=collate_fn)
# baseline = evaluate_baseline(model, dataloader, metric).mean()
# print(baseline)
# %%