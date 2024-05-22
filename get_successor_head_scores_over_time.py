#%%
import os
import argparse
import string
import yaml

import torch
from num2words import num2words
from einops import einsum
from transformer_lens import HookedTransformer
from huggingface_hub import HfApi

from utils.model_utils import load_model
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
        "-l",
        "--large_model",
        default=False,
        help="Whether to load a large model",
    )
    parser.add_argument(    
        "-cd",
        "--cache_dir",
        default="model_cache",
        help="Directory for cache",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        default=False,
        help="Whether to overwrite existing results",
    )
    return parser.parse_args()

def read_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

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

alt = args.alt_model
model_folder = f"{alt[11:]}" if alt is not None else f"{args.model}"
os.makedirs(f"results/components/{model_folder}/", exist_ok=True)

accuracies = {}
if os.path.exists(f"results/components/{model_folder}/successor_heads_over_time.pt") and not args.overwrite:
    accuracies = torch.load(f"results/components/{model_folder}/successor_heads_over_time.pt")
#%%
dataset = {'numbers': [str(i) for i in range(1, 201)],
'number_words': [num2words(i) for i in range(1, 21)],
'cardinal_words': ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'],
'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday'],
'months': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'January'],
'day_prefixes': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
'month_prefixes': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
'letters': list(string.ascii_uppercase)
}
dataset = {k:[' ' + s for s in v] for k,v in dataset.items()}

#%%
if 'pythia' in args.model:
    ckpts = [0, *(2**i for i in range(10)),  *(i * 1000 for i in range(1, 144))]
else:
    api = HfApi()
    refs = api.list_repo_refs(args.model)
    ckpts = [branch.name for branch in refs.branches if 'step' in branch.name]
    ckpts.sort(key=lambda name: int(name.split('-')[0][4:]))

for ckpt in ckpts:
    if ckpt in accuracies:
        continue
    if args.large_model or args.canonical_model:
        model = HookedTransformer.from_pretrained(
            args.model, 
            checkpoint_value=ckpt,
            center_unembed=False,
            center_writing_weights=False,
            fold_ln=False,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            **{"cache_dir": args.cache_dir},
        )
    else:
        ckpt_key = f"step{ckpt}"
        # TODO: Add support for different model seeds
        model = load_model(args.model, args.alt_model, ckpt_key, args.cache_dir)

    # maps from a word in the dataset to that word's vocab / token ID
    word_token_mapping = {}
    for k, v in dataset.items():
        for s in v:
            token = model.tokenizer(s, add_special_tokens=False)['input_ids']
            if len(token) > 1:
                raise ValueError(f'Got multi-token word: {s} ({token}) in {k}')
            token = token[0]
            word_token_mapping[s] = token 

    all_relevant_tokens_ids = list(set(word_token_mapping.items()))

    word_idx_mapping = {word:i for i, (word, _) in enumerate(all_relevant_tokens_ids)}
    idxs = torch.tensor([idx for _, idx in all_relevant_tokens_ids])

    # we index into W_U/W_E using all relevant token ids, so we have to again map from 
    # the words into this indexed portion of W_U/W_E

    idx_dataset = {k: [word_idx_mapping[s] for s in v] for k,v in dataset.items()}

    # getting which words upweight which other words (from SH paper)
    with torch.inference_mode():
        W_E = model.embed.W_E[idxs]
        W_U = model.unembed.W_U[:, idxs]
        mlp0 = model.blocks[0].mlp
        mlp0_W_E = mlp0(W_E.unsqueeze(0)).squeeze(0)
    
    # You could in theory do this in one step, by computing OV for all layers at once! 
    # But this eats up too much GPU memory, as both the hidden dimension and # of layers grow
    total_examples = sum(len(v) - 1 for v in idx_dataset.values())
    successes = torch.ones([model.cfg.n_layers, model.cfg.n_heads, total_examples], device='cuda') * -1
    for layer in range(model.cfg.n_layers):
        with torch.inference_mode():
            OV = model.blocks[layer].attn.OV.AB
            vocab_up_down_weight = einsum(W_U, OV, mlp0_W_E, "hidden_output V_output, head hidden_input hidden_output, V_input hidden_input -> head V_input V_output")
        i = 0
        for k, v in idx_dataset.items():
            input_ids = v[:-1]
            target_ids = v[1:]

            input_tensor = torch.tensor(input_ids)
            target_tensor = torch.tensor(target_ids)

            all_targets_tensor = torch.tensor(list(set(v)))

            target_logits = vocab_up_down_weight[:, input_tensor, target_tensor]
            all_target_logits = vocab_up_down_weight[:, input_tensor][:, :, all_targets_tensor]

            all_targets_max_logits = all_target_logits.max(dim=-1).values
            success = target_logits == all_targets_max_logits
            successes[layer, :, i:i + len(v) - 1] = success
            i += len(v) - 1

    assert not torch.any(successes < 0)
    accuracy = successes.float().mean(-1)
    accuracies['ckpt'] = accuracy
    torch.save(accuracies, f"results/components/{model_folder}/successor_heads_over_time.pt")



    