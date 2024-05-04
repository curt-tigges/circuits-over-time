from typing import Tuple, List
from functools import partial

import os
import argparse
import yaml
import numpy as np 
import pandas as pd
import torch

from utils.data_processing import (
    load_edge_scores_into_dictionary,
    get_ckpts,
    get_ckpts
)
from utils.backup_analysis import load_model
from utils.data_utils import generate_data_and_caches

from utils.component_evaluation import (
    evaluate_direct_effect_heads,
    filter_name_movers,
    evaluate_s2i_candidates,
    evaluate_induction_scores
)

def print_gpu_memory_usage(label="", device="cuda:0"):
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)  # Convert bytes to GB
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
    print(f"{label} - Memory Allocated: {allocated:.2f} GB, Memory Reserved: {reserved:.2f} GB")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download & assess model checkpoints")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
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
    if not args.variant:
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
            [i * 1000 for i in range(4, 144)]
        )
    elif schedule == "sparse":
        ckpts = (
            [2**i for i in range(8, 10)]
            + [i * 1000 for i in range(1, 10)]
            + [i * 5000 for i in range(2, 10)]
            + [i * 10000 for i in range(5, 10)]
            + [i * 20000 for i in range(5, 8)]
            + [143000]
        )
    elif schedule == "custom":
        ckpts = []
    else:
        ckpts = [10000, 143000]

    return ckpts


def main(args):

    torch.set_grad_enabled(False)

    config = read_config(args.config)
    print(config)

    TASK = config['task']
    BASE_MODEL = config['base_model']
    VARIANT = config['variant']
    MODEL_SHORTNAME =BASE_MODEL if not VARIANT else VARIANT[11:]
    CACHE = config['cache']
    DATASET_SIZE = config['dataset_size']
    BATCH_SIZE = config['batch_size']
    CHECKPOINT_SCHEDULE = get_ckpts(config['checkpoint_schedule'])
    if 'device' in config:
        DEVICE = config['device']
    else:
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    overwrite = config['overwrite']
    
    # load circuit data
    folder_path = f'results/graphs/{MODEL_SHORTNAME}/{TASK}'
    df = load_edge_scores_into_dictionary(folder_path)

    # filter everything before 1000 steps
    df = df[df['checkpoint'] >= 1000]

    df[['source', 'target']] = df['edge'].str.split('->', expand=True)

    # load model and dataset
    model = load_model(BASE_MODEL, VARIANT, 143000, CACHE, DEVICE, large_model=True)
    model.tokenizer.add_bos_token = False
    print_gpu_memory_usage("After loading model")
    ioi_dataset, abc_dataset = generate_data_and_caches(model, DATASET_SIZE, verbose=True, prepend_bos=True)
    print_gpu_memory_usage("After generating data")

    for checkpoint in CHECKPOINT_SCHEDULE:
        # check if file exists; if not, create
        if not os.path.exists(f'results/components/{MODEL_SHORTNAME}/components_over_time.pt'):
            os.makedirs(f'results/components/{MODEL_SHORTNAME}', exist_ok=True)
            components_over_time = dict()
            heads_over_time = dict()
        else:
            components_over_time = torch.load(f'results/components/{MODEL_SHORTNAME}/components_over_time.pt')
            heads_over_time = torch.load(f'results/components/{MODEL_SHORTNAME}/heads_over_time.pt')

        if checkpoint in components_over_time and not overwrite:
            continue

        print(f"Processing checkpoint {checkpoint}")
        model = load_model(BASE_MODEL, VARIANT, checkpoint, CACHE, DEVICE, large_model=True)
        print_gpu_memory_usage("After loading first checkpoint model")
        checkpoint_df = df[df['checkpoint'] == checkpoint].copy()
        component_scores = dict()
        model_heads = dict()

        component_scores['direct_effect_scores'] = evaluate_direct_effect_heads(model, checkpoint_df, ioi_dataset, verbose=False, cuda_device=int(DEVICE[-1]), batch_size=BATCH_SIZE)
        
        if component_scores['direct_effect_scores'] is not None:
            nmh_list = filter_name_movers(component_scores['direct_effect_scores'], copy_score_threshold=10)
        else:
            nmh_list = []
        
        model_heads['nmh'] = nmh_list
        print(f"Found {len(nmh_list)} NMHs")
        print(nmh_list)
        
        if len(nmh_list) > 0:
            component_scores['s2i_scores'], s2i_list = evaluate_s2i_candidates(model, checkpoint_df, ioi_dataset, nmh_list, batch_size=BATCH_SIZE, verbose=False)
            print(f"Found {len(s2i_list)} S2I heads")
            print(s2i_list)
        else:
            component_scores['s2i_scores'] = None
            s2i_list = []

        model_heads['s2i'] = s2i_list

        component_scores['tertiary_head_scores'] = evaluate_induction_scores(model, checkpoint_df)

        components_over_time[checkpoint] = component_scores
        heads_over_time[checkpoint] = model_heads

        torch.save(components_over_time, f'results/components/{MODEL_SHORTNAME}/components_over_time.pt')
        torch.save(heads_over_time, f'results/components/{MODEL_SHORTNAME}/heads_over_time.pt')

    return components_over_time
   
if __name__ == "__main__":
    args = process_args()
    main(args)