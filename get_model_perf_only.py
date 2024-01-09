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

import circuit_utils as cu

# Settings
if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"

torch.set_grad_enabled(False)
DO_SLOW_RUNS = True


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
    config = read_config(args.config)

    print(config)

    model_name = config["model_name"]
    model_tl_name = config["model_tl_name"]

    model_full_name = f"EleutherAI/{model_name}"
    model_tl_full_name = f"EleutherAI/{model_tl_name}"

    cache_dir = config["cache_dir"]

    # load model
    model = load_model(
        model_full_name, model_tl_full_name, "step143000", cache_dir=cache_dir
    )

    # set up data
    N = 70
    ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric_noising = generate_data_and_caches(model, N, verbose=True)


    # get baselines
    clean_logits, clean_cache = model.run_with_cache(ioi_dataset.toks)
    corrupted_logits, corrupted_cache = model.run_with_cache(abc_dataset.toks)

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
    results_dict = cu.get_chronological_circuit_performance(
        model_full_name,
        model_tl_full_name,
        cache_dir,
        ckpts,
        clean_tokens=ioi_dataset.toks,
        corrupted_tokens=abc_dataset.toks,
        dataset=ioi_dataset
    )

    # save results
    os.makedirs(f"results/{model_name}-no-dropout", exist_ok=True)
    torch.save(
        results_dict["logit_diffs"], f"results/{model_name}-no-dropout/overall_perf.pt"
    )
    torch.save(
        results_dict["clean_baselines"],
        f"results/{model_name}-no-dropout/clean_baselines.pt",
    )
    torch.save(
        results_dict["corrupted_baselines"],
        f"results/{model_name}-no-dropout/corrupted_baselines.pt",
    )


if __name__ == "__main__":
    args = get_args()
    print(args)
    main(args)
