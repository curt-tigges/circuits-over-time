import os
from collections import namedtuple
from pathlib import Path
from functools import partial

import torch
import pickle
import argparse
import yaml

from torchtyping import TensorType as TT
from transformer_lens import HookedTransformer

from utils.model_utils import load_model, clear_gpu_memory
from utils.data_utils import UniversalPatchingDataset
import utils.circuit_utils as cu

from utils.metrics import (
    CircuitMetric,
    compute_logit_diff,
    compute_probability_diff,
    compute_probability_mass,
    compute_rank_0_rate,
    compute_accuracy
)

# Settings
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_args():
    parser = argparse.ArgumentParser(description="Download & assess model checkpoints")
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/defaults.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "-t",
        "--task",
        default="ioi",
        help="Name of task on which to evaluate model",
    )
    return parser.parse_args()


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def get_ckpts(config):
    if config["checkpoint_schedule"] == "linear":
        ckpts = [i * 1000 for i in range(1, 144)]
    elif config["checkpoint_schedule"] == "exponential":
        ckpts = [
            round((2**i) / 1000) * 1000 if 2**i > 1000 else 2**i
            for i in range(18)
        ]
    elif config["checkpoint_schedule"] == "exp_plus_detail":
        ckpts = (
            [2**i for i in range(10)]
            + [i * 1000 for i in range(1, 16)]
            + [i * 5000 for i in range(3, 14)]
            + [i * 10000 for i in range(7, 15)]
        )
    else:
        ckpts = [1, 2]

    return ckpts


def get_data_and_metrics(
        model: HookedTransformer,
        task_name: str,
    ):
    assert task_name in ["ioi", "greater_than", "sentiment_cont", "sentiment_class", "mood_sentiment"]

    if task_name == "ioi":
        ds = UniversalPatchingDataset.from_ioi(model, 70)
        logit_diff_metric = partial(compute_logit_diff, answer_token_indices=ds.answer_toks, positions=ds.positions)
        logit_diff = CircuitMetric("logit_diff", logit_diff_metric)
        accuracy_metric = partial(compute_accuracy, answer_token_indices=ds.answer_toks, positions=ds.positions)
        accuracy = CircuitMetric("accuracy", accuracy_metric)
        rank_0_metric = partial(compute_rank_0_rate, answer_token_indices=ds.answer_toks, positions=ds.positions)
        rank_0 = CircuitMetric("rank_0", rank_0_metric)
        probability_diff_metric = partial(compute_probability_diff, answer_token_indices=ds.answer_toks, positions=ds.positions)
        probability_diff = CircuitMetric("probability_diff", probability_diff_metric)
        probability_mass_metric = partial(compute_probability_mass, answer_token_indices=ds.answer_toks, positions=ds.positions)
        probability_mass = CircuitMetric("probability_mass", probability_mass_metric)
        metrics = [logit_diff, accuracy, rank_0, probability_diff, probability_mass]

    elif task_name == "greater_than":
        # Get data
        ds = UniversalPatchingDataset.from_greater_than(model, 1000)
        logit_diff_metric = partial(
            compute_logit_diff, 
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags, 
            mode="groups"
        )
        logit_diff = CircuitMetric("logit_diff", logit_diff_metric)
        prob_diff_metric = partial(
            compute_probability_diff, 
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags,
            mode="group_sum"
        )
        probability_diff = CircuitMetric("prob_diff", prob_diff_metric)
        probability_mass_metric = partial(
            compute_probability_mass,
            answer_token_indices=ds.answer_toks,
            flags_tensor=ds.group_flags,
            mode="group_sum"
        )
        metrics = [logit_diff, probability_diff, probability_mass]

    elif task_name == "sentiment_cont":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "cont")
        logit_diff_metric = partial(compute_logit_diff, answer_token_indices=ds.answer_toks, mode="pairs")
        metric = CircuitMetric("logit_diff", logit_diff_metric)
        metrics = [metric]

    elif task_name == "sentiment_class":
        # Get data
        ds = UniversalPatchingDataset.from_sentiment(model, "class")
        logit_diff_metric = partial(compute_logit_diff, answer_token_indices=ds.answer_toks, mode="pairs")
        metric = CircuitMetric("logit_diff", logit_diff_metric)
        metrics = [metric]

    elif task_name == "mood_sentiment":
        raise ValueError("Not yet implemented")
    
    return ds, metrics

def main(args):

    torch.set_grad_enabled(False)

    config = read_config(args.config)
    task = args.task

    print(config)

    model_name = config["model_name"]
    model_tl_name = config["model_tl_name"]

    model_full_name = f"EleutherAI/{model_name}"
    model_tl_full_name = f"EleutherAI/{model_tl_name}"

    cache_dir = config["cache_dir"]
    batch_size = config["batch_size"]

    # load model
    model = load_model(
        model_full_name, model_tl_full_name, "step143000", cache_dir=cache_dir
    )
    
    # set up data
    ds, metrics = get_data_and_metrics(model, task)

    # get baselines
    clean_logits = cu.run_with_batches(model, ds.toks, batch_size=20, max_seq_len=ds.max_seq_len)
    flipped_logits = cu.run_with_batches(model, ds.flipped_toks, batch_size=20, max_seq_len=ds.max_seq_len)

    clean_primary_metric = metrics[0](clean_logits)
    print(f"Clean {metrics[0].name}: {clean_primary_metric:.4f}")

    flipped_primary_metric = metrics[0](flipped_logits)
    print(f"Flipped {metrics[0].name}: {flipped_primary_metric:.4f}")

    clear_gpu_memory(model)

    # specify checkpoint schedule
    ckpts = get_ckpts(config)

    # get values over time
    results_dict = cu.get_chronological_circuit_performance_flexible(
        model_full_name,
        model_tl_full_name,
        cache_dir,
        ckpts,
        dataset=ds,
        metrics=metrics,
        batch_size=batch_size,
    )

    # save results
    os.makedirs(f"results/{model_name}-no-dropout/{task}", exist_ok=True)
    
    for metric in results_dict.keys():
        torch.save(
            results_dict[metric], f"results/{model_name}-no-dropout/{task}/{metric}.pt"
        )


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
