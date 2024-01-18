import os
from collections import namedtuple
from pathlib import Path
from functools import partial

import torch
import pickle
import argparse
import yaml

from torchtyping import TensorType as TT

from utils.model_utils import load_model, clear_gpu_memory
from utils.data_utils import generate_data_and_caches
from utils.metrics import CircuitMetric, get_prob_diff
from data.greater_than_dataset import YearDataset, get_valid_years
import utils.circuit_utils as cu

from data.sentiment_datasets import get_dataset
from utils.circuit_analysis import get_logit_diff

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


def read_data(file_path):
    with open(file_path, "r") as f:
        content = f.read()

    prompts_str, answers_str = content.split("\n\n")
    prompts = prompts_str.split("\n")  # Remove the last empty item
    answers = [
        tuple(answer.split(",")) for answer in answers_str.split(";")[:-1]
    ]  # Remove the last empty item

    return prompts, answers


def main(args):

    torch.set_grad_enabled(False)

    config = read_config(args.config)

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
    ds = get_dataset(model, device)
    
    logit_diff_metric = CircuitMetric("logit_diff_multi", partial(get_logit_diff, answer_tokens=ds.answer_tokens))

    metrics = [logit_diff_metric]
    
    # get baselines
    clean_logits = cu.run_with_batches(model, ds.clean_tokens.to(device), batch_size=20, max_seq_len=12)
    corrupted_logits = cu.run_with_batches(model, ds.corrupted_tokens.to(device), batch_size=20, max_seq_len=12)

    clean_prob_diff = logit_diff_metric(clean_logits)
    print(f"Clean logit diff: {clean_prob_diff:.4f}")

    corrupted_prob_diff = logit_diff_metric(corrupted_logits)
    print(f"Corrupted logit diff: {corrupted_prob_diff:.4f}")

    clear_gpu_memory(model)

    # specify checkpoint schedule
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

    # get values over time
    results_dict = cu.get_chronological_circuit_performance_flexible(
        model_full_name,
        model_tl_full_name,
        cache_dir,
        ckpts,
        clean_tokens=ds.clean_tokens.to(device),
        corrupted_tokens=ds.corrupted_tokens.to(device),
        metrics=metrics,
        max_seq_len=12,
        batch_size=batch_size,
    )

    # save results
    os.makedirs(f"results/{model_name}-no-dropout", exist_ok=True)
    
    for metric in results_dict.keys():
        torch.save(
            results_dict[metric], f"results/{model_name}-no-dropout/{metric}.pt"
        )


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
