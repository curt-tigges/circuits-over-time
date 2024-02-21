import os
from collections import namedtuple

import torch
import pickle
import argparse
import yaml

from torchtyping import TensorType as TT

from utils.model_utils import load_model, clear_gpu_memory
from utils.data_utils import generate_data_and_caches
from utils.metrics import _logits_to_mean_logit_diff, _logits_to_mean_accuracy, _logits_to_rank_0_rate
from EAP-positional.graph import get_acdcpp_results

import utils.circuit_utils as cu

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
    N = 70
    ioi_dataset, abc_dataset, _, _, _ = generate_data_and_caches(model, N, verbose=True)


    # get baselines
    clean_logits = cu.run_with_batches(model, ioi_dataset.toks, batch_size, 21)
    corrupted_logits = cu.run_with_batches(model, abc_dataset.toks, batch_size, 21)

    clean_logit_diff = _logits_to_mean_logit_diff(clean_logits, ioi_dataset)
    print(f"Clean logit diff: {clean_logit_diff:.4f}")

    corrupted_logit_diff = _logits_to_mean_logit_diff(corrupted_logits, ioi_dataset)
    print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

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
    get_acdcpp_results()

    results_dict = cu.get_chronological_circuit_performance(
        model_full_name,
        model_tl_full_name,
        cache_dir,
        ckpts,
        clean_tokens=ioi_dataset.toks,
        corrupted_tokens=abc_dataset.toks,
        dataset=ioi_dataset,
        max_seq_len=21,
        batch_size=batch_size,
    )

    # save results
    os.makedirs(f"results/{model_name}-no-dropout", exist_ok=True)
    torch.save(
        results_dict["logit_diffs"], f"results/{model_name}-no-dropout/logit_diffs.pt"
    )
    torch.save(
        results_dict["ld_clean_baselines"],
        f"results/{model_name}-no-dropout/ld_clean_baselines.pt",
    )
    torch.save(
        results_dict["ld_corrupted_baselines"],
        f"results/{model_name}-no-dropout/ld_corrupted_baselines.pt",
    )
    torch.save(
        results_dict["accuracy_vals"],
        f"results/{model_name}-no-dropout/accuracy_vals.pt",
    )
    torch.save(
        results_dict["accuracy_clean_baselines"],
        f"results/{model_name}-no-dropout/acc_clean_baselines.pt",
    )
    torch.save(
        results_dict["accuracy_corrupted_baselines"],
        f"results/{model_name}-no-dropout/acc_corrupted_baselines.pt",
    )
    torch.save(
        results_dict["rank_0_rate_vals"],
        f"results/{model_name}-no-dropout/rank_0_rate_vals.pt",
    )
    torch.save(
        results_dict["rank_0_rate_clean_baselines"],
        f"results/{model_name}-no-dropout/rank_clean_baselines.pt",
    )
    torch.save(
        results_dict["rank_0_rate_corrupted_baselines"],
        f"results/{model_name}-no-dropout/rank_corrupted_baselines.pt",
    )


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
