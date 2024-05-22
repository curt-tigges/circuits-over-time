#%%
import argparse
from pathlib import Path
from collections import defaultdict 

import yaml
from tqdm import tqdm
import pandas as pd 
import torch
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

from data.greater_than_dataset import get_year_indices
from utils.data_utils import UniversalPatchingDataset
from utils.model_utils import load_model
from edge_attribution_patching.graph import Graph, MLPNode, AttentionNode, Node

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download & assess model checkpoints")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to config file",
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
        "-cd",
        "--cache_dir",
        default="/mnt/hdd-0/circuits-over-time/model_cache",
        help="Directory for cache",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        help="Whether to overwrite existing results",
    )
    parser.add_argument(
        "-rd",
        "--results_dir",
        default="results/circuit_verification/",
        help="dir into which to write graphs",
    )
    parser.add_argument(
        "-grd",
        "--graph_results_dir",
        default="/mnt/hdd-0/circuits-over-time/",
        help="dir into which graphs were written",
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

def collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = torch.stack([item[key] for item in batch])
    return batch_dict


args = process_args()

checkpoint_path = Path(args.graph_results_dir) / f'results/graphs/{args.model}/greater_than/raw/'
checkpoints = [ckpt for ckpt in checkpoint_path.iterdir() if ckpt.suffix == '.json']
checkpoints.sort(key=lambda ckpt: int(ckpt.stem))

results_path = Path(args.results_dir) / args.model 
results_path.mkdir(exist_ok=True, parents=True)
results_file = results_path / 'greater_than.pt'
results = {}
if not args.overwrite and results_file.exists():
    results = torch.load(results_file)

for ckpt_file in checkpoints:
    ckpt = ckpt_file.stem
    if ckpt in results:
        continue 
    g = Graph.from_json(str(ckpt_file))
    logits = g.nodes['logits']
    mlps = {edge.parent for edge in logits.parent_edges if edge.in_graph and isinstance(edge.parent, MLPNode)}
    attns = {edge.parent for mlp in mlps for edge in mlp.parent_edges if edge.in_graph and isinstance(edge.parent, AttentionNode)}

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
    
    ds = UniversalPatchingDataset.from_greater_than(model)
    dataloader = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate_fn)
    year_indices = get_year_indices(model.tokenizer)

    
    d = defaultdict(list)
    def make_acts_hook(node: Node, positions: torch.Tensor):
        def acts_hook(x, hook):
            x = x[node.index].cpu()
            x = x[torch.arange(positions.size(0)), positions]
            d[node.name].append(x)
        return node.out_hook, acts_hook

    def make_attn_hook(node: Node, positions: torch.Tensor):
        def attn_hook(x, hook):
            x = x[:,node.head].cpu()
            x = x[torch.arange(positions.size(0)), positions]
            d[node.name + '_pattern'].append(x)
        return f'blocks.{node.layer}.attn.hook_pattern', attn_hook

    all_years = []
    all_flags = []
    for batch in dataloader:
        clean = batch['toks']
        flags = batch['flags_tensor']
        positions = batch['positions']
        years = (flags == -1).float().sum(-1) - 1
        all_flags.append(flags)
        all_years.append(years)
        hooks = [make_acts_hook(node, positions) for node in mlps | attns] + [make_attn_hook(attn, positions) for attn in attns]
        with torch.inference_mode():
            with model.hooks(hooks):
                model(clean)

    d = {k: torch.cat(v, dim=0) for k,v in d.items()}
    all_years = torch.cat(all_years, dim=0)
    all_flags = torch.cat(all_flags, dim=0)
    
    W_U = model.unembed.W_U[:, year_indices]
    def logit_lens(activations):
        return torch.einsum('dv,bd->bv', W_U, activations)

    with torch.inference_mode():
        ld = {k: logit_lens(v.to('cuda')).cpu() for k, v in d.items() if 'pattern' not in k}
    
    good_flag = (all_flags != -1).float()
    bad_flag = (all_flags == -1).float()
    patterns_same = {node.name: (d[node.name + '_pattern'][:, -7] == d[node.name + '_pattern'].max(-1).values).float().mean() for node in attns}
    logits_correct = {node.name: ((ld[node.name].argmax(-1) == all_years) | (ld[node.name].argmax(-1) == all_years + 1)).float().mean() for node in attns}
    mlps_correct = {node.name: (((ld[node.name] * good_flag).sum(-1) / good_flag.sum(-1)) - ((ld[node.name] * bad_flag).sum(-1) / bad_flag.sum(-1))).mean() for node in mlps}
    same_algorithm = {'attn_patterns':patterns_same, 'attn_logits':logits_correct, 'mlp_logits':mlps_correct}

    results[ckpt] = same_algorithm

torch.save(results, results_file)