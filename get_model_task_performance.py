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

import utils.circuit_utils as cu
from utils.data_processing import get_ckpts

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


def main(args):

    torch.set_grad_enabled(False)

    args = process_args()

    print(args)

    if "large_model" in args:
        large_model = args.large_model
    else:
        large_model = False

    # specify checkpoint schedule
    ckpts = get_ckpts(args.checkpoint_schedule)
    print(f"Checkpoints: {ckpts}")
    # get values over time
    results_dict = cu.get_chronological_multi_task_performance(
        args.model,
        args.alt_model,
        args,
        args.cache_dir,
        ckpts,
        batch_size=args.batch_size,
        large_model=large_model
    )

    # save results


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
