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

    if "large_model" in config.keys():
        large_model = config["large_model"]
    else:
        large_model = False

    # specify checkpoint schedule
    ckpts = get_ckpts(config)

    # get values over time
    results_dict = cu.get_chronological_circuit_performance_flexible(
        model_full_name,
        model_tl_full_name,
        config,
        cache_dir,
        ckpts,
        task=task,
        batch_size=batch_size,
        large_model=large_model
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
