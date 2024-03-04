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
        "-alt",
        "--alt_model",
        default=None,
        help="Name of alternate model to load, with architecture the same as the main model",
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
        "-cust",
        "--custom_schedule",
        default=[],
        help="Custom schedule for checkpoints",
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
    parser.add_argument(
        "-strt",
        "--start",
        default=25,
        help="Start point for faithfulness curve",
    )
    parser.add_argument(
        "-end",
        "--end",
        default=1600,
        help="End point for faithfulness curve",
    )
    parser.add_argument(
        "-stp",
        "--step",
        default=25,
        help="Step for faithfulness curve",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        help="Whether to overwrite existing results",
    )
    parser.add_argument(
        "-st",
        "--search_type",
        default="linear",
        help="Search type for faithfulness curve; can be linear or binary",
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
    if not args.alt_model:
        setattr(args, "canonical_model", True)
    else:
        setattr(args, "canonical_model", False)
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
    elif schedule == "late_start_exp_plus_detail":
        ckpts = (
            [i * 4000 for i in range(1, 16)]
            + [i * 5000 for i in range(3, 14)]
            + [i * 10000 for i in range(7, 15)]
        )
    elif schedule == "late_start_all":
        ckpts = (
            [i * 1000 for i in range(4, 143)]
        )
    elif schedule == "custom":
        ckpts = []
    else:
        ckpts = [10000, 143000]

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
        target_minimum: float = 0.8,
        start: int = 100,
        end: int = 1000,
        step: int = 100,
    ):

    faithfulness = dict()

    for size in range(start, end, step):

        graph.apply_greedy(size, absolute=True)
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
        faithfulness[size] = (evaluate_graph(model, graph, dataloader, metric).mean() / baseline).item()

    # Define the graph with this threshold
    for size in range(start, end, step):
        print(f"Size: {size}, Faithfulness: {faithfulness[size]}")
        exceeds_threshold = False
        if faithfulness[size] > target_minimum:
            exceeds_threshold = True
            min_size = int(size)
            print(f"Exceeds threshold: {min_size}")

            break

    if not exceeds_threshold:
        min_size = end

    return faithfulness, min_size


def get_faithfulness_metrics_adaptive(
    graph: Graph,
    model: HookedTransformer, 
    dataloader: DataLoader, 
    metric: CircuitMetric,
    baseline: float,
    threshold: float = 0.8,
    start: int = 25,
    end: int = 1000,
    initial_step: int = 25,
    step_reduction_factor: float = 0.5,  # Factor by which to reduce the step size
    min_step: int = 1,  # Minimum step size
    local_search_radius: int = 5,  # Radius for local linear search around the middle element
):
    faithfulness = dict()
    step = initial_step
    size = start
    exceeds_threshold = False
    min_size = None

    while size < end:
        graph.apply_greedy(size, absolute=True)
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
        score = (evaluate_graph(model, graph, dataloader, metric).mean() / baseline).item()
        faithfulness[size] = score
        
        if score > threshold and not exceeds_threshold:
            exceeds_threshold = True
            # Perform a binary search with local linear search to handle noise
            binary_faithfulness, min_size = get_faithfulness_metrics_binary_search_with_local_search(
                graph, 
                model, 
                dataloader, 
                metric, 
                baseline, 
                threshold, 
                start=max(size - step, 1), 
                end=size,
                local_search_radius=local_search_radius
            )
            step = initial_step * 2

            # Add faithfulness metrics from binary search
            for k, v in binary_faithfulness.items():
                faithfulness[k] = v

        print(f"Size: {size}, Faithfulness: {score}, Exceeds threshold: {exceeds_threshold}")

        if step < initial_step and not exceeds_threshold and (score < threshold * 0.75 or score < faithfulness[size - step]):
            step = max(initial_step, int(step / step_reduction_factor))
            print(f"Resetting step size at size: {size} to {step}")

        # Adapt the step size
        if not exceeds_threshold and score > threshold * 0.75:
            step = max(min_step, int(step * step_reduction_factor))
            print(f"Reducing step size at size: {size} to {step}")

        size += step

    if min_size is None:
        min_size = end

    print(f"Optimal size is {min_size} with faithfulness {faithfulness[min_size]}")
    return faithfulness, min_size


def get_faithfulness_metrics_binary_search_with_local_search(
    graph: Graph,
    model: HookedTransformer, 
    dataloader: DataLoader, 
    metric: CircuitMetric,
    baseline: float,
    threshold: float = 0.8,
    start: int = 100,
    end: int = 1000,
    local_search_radius: int = 5,
):
    def evaluate_size(size: int) -> float:
        if size not in faithfulness:
            graph.apply_greedy(size, absolute=True)
            graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
            faithfulness[size] = (evaluate_graph(model, graph, dataloader, metric).mean() / baseline).item()
            print(f"Size: {size}, Faithfulness: {faithfulness[size]}, Exceeds threshold: {faithfulness[size] >= threshold}")
        return faithfulness[size]

    print("Entering binary search")
    faithfulness = dict()
    low = start
    high = end
    min_size = None
    while low <= high:
        mid = (low + high) // 2
        print(f"Low: {low}, High: {high}, Mid: {mid}")
        mid_score = evaluate_size(mid)
        if mid_score >= threshold:
            min_size = mid
            high = mid - 1
        else:
            low = mid + 1

        # Perform local linear search around the middle element if the score is close to the threshold
        if abs(mid_score - threshold) <= threshold * 0.1:  # Adjust the tolerance as needed
            for offset in range(-local_search_radius, local_search_radius + 1):
                if offset == 0:
                    continue
                local_size = mid + offset
                if start <= local_size <= end:
                    local_score = evaluate_size(local_size)
                    if local_score >= threshold and (min_size is None or local_size < min_size):
                        min_size = local_size
                        high = local_size - 1
                        break

    if min_size is None:
        return faithfulness, end  # No size found that meets the threshold

    # Return the faithfulness metrics for sizes up to the minimal size found
    return faithfulness, min_size


#%%
def main(args):
    print(f"Arguments: {args}")
    schedule = args.ckpt_schedule
    task = args.task
    ckpts = get_ckpts(schedule)
    alt = args.alt_model
    model_folder = f"{alt[11:]}" if alt is not None else f"{args.model}"
    if args.custom_schedule:
        ckpts = args.custom_schedule

    for ckpt in ckpts:
        # first check if graph json already exists
        if os.path.exists(f"results/graphs/{model_folder}/{args.task}/{ckpt}.json"):
            if not args.overwrite:
                continue

        os.makedirs(f"results/graphs/{model_folder}/{task}", exist_ok=True)
        os.makedirs(f"results/images/{model_folder}/{task}", exist_ok=True)
        os.makedirs(f"results/faithfulness/{model_folder}/{task}", exist_ok=True)
        os.makedirs(f"results/baselines/{model_folder}", exist_ok=True)

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
            model = load_model(args.model, args.alt_model, ckpt_key, args.cache_dir)
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        
        # Set up for task 
        ds, metric = get_data_and_metrics(model, task, eap=True)
        graph = Graph.from_model(model)
        dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)
        
        # load the baseline dict
        if os.path.exists(f"results/graphs/{model_folder}/{task}.json"):
            baseline_dict = json.load(open(f"results/baselines/{model_folder}/{task}.json"))
        else:
            baseline_dict = dict()

        # Evaluate baseline and graph
        baseline = evaluate_baseline(model, dataloader, metric).mean()
        baseline_dict[ckpt] = baseline.item()
        
        # save the baseline dict
        with open(f"results/baselines/{model_folder}/{task}.json", "w") as f:
            json.dump(baseline_dict, f)

        print(f"Baseline metric value for {args.task}: {baseline}")
        attribute(model, graph, dataloader, partial(metric, loss=True), integrated_gradients=30)

        
        faithfulness = dict()

        if args.verify:
            # if args.search_type == "linear":
            #     search_fn = get_faithfulness_metrics
            # elif args.search_type == "binary":
            #     search_fn = get_faithfulness_metrics_binary_search
            # elif args.search_type == "adaptive":
            #     search_fn = get_faithfulness_metrics_adaptive
            
            faithfulness, args.top_n = get_faithfulness_metrics_adaptive(graph, model, dataloader, metric, baseline, start=args.start, end=args.end, initial_step=args.step)
            
            

        graph.apply_greedy(args.top_n, absolute=True)
        graph.prune_dead_nodes(prune_childless=True, prune_parentless=True)
        results = evaluate_graph(model, graph, dataloader, metric).mean()
        faithfulness[args.top_n] = (results / baseline).item()
        print(results)

        # Save graph and results
        
        graph.to_json(f'results/graphs/{model_folder}/{task}/{ckpt}.json')
        gz = graph.to_graphviz()
        gz.draw(f'results/images/{model_folder}/{task}/{ckpt}.png', prog='dot')

        if args.verify:
        # Save faithfulness to JSON
            print(f"Faithfulness: {faithfulness}")
            print(f"Optimal size: {args.top_n}")
            with open(f"results/faithfulness/{model_folder}/{task}/{ckpt}.json", "w") as f:
                print(f"Saving faithfulness to JSON for {model_folder} and {task} to {ckpt}.json...")
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