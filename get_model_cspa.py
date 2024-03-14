import os
from collections import namedtuple

import torch
import pickle
import argparse
import yaml

from torchtyping import TensorType as TT

import torch
from utils.cspa_main import (
    get_cspa_per_checkpoint
)

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def get_args():
    parser = argparse.ArgumentParser(description="Get CPSA per checkpoint and attention head")
    parser.add_argument(
        "-c",
        "--config",
        default="./configs/cspa/160m-canonical.yml",
        help="Path to config file",
    )
    return parser.parse_args()


def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main(args):

    torch.set_grad_enabled(False)

    config = read_config(args.config)

    print(config)


    get_cspa_per_checkpoint(
        config['base_model'], 
        config['variant'], 
        config['cache'], 
        device, 
        config["checkpoint_schedule"], 
        start_layer=config["start_layer"], 
        overwrite=config["overwrite"], 
        display_all=False
    )


if __name__ == "__main__":
    args = get_args()
    main(args)