#%%
import string

import torch
from num2words import num2words
from einops import einsum, rearrange

from transformer_lens import HookedTransformer

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
model = HookedTransformer.from_pretrained(
    'EleutherAI/pythia-1.4b', 
    #checkpoint_value=143000,
    center_unembed=False,
    center_writing_weights=False,
    fold_ln=False,
    dtype=torch.bfloat16,
    **{"cache_dir": '/mnt/hdd-0/circuits-over-time/model_cache'},
    device='cuda:3'
)
#%%
# maps from a word in the dataset to that word's id
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

# we index into W_U/W_E using all relevant token ids, so we have to again map from the words into this indexed portion of W_U/W_E

idx_dataset = {k: [word_idx_mapping[s] for s in v] for k,v in dataset.items()}

#%%
with torch.inference_mode():
    W_E = model.embed.W_E[idxs]
    W_U = model.unembed.W_U[:, idxs]
    mlp0 = model.blocks[0].mlp
    mlp0_W_E = mlp0(W_E.unsqueeze(0)).squeeze(0)
    
# You could in theory do this in one step, by computing OV for all layers at once! 
# But this eats up too much GPU memory, as both the hidden dimension and # of layers grow
total_examples = sum(len(v) - 1 for v in idx_dataset.values())
successes = torch.ones([model.cfg.n_layers, model.cfg.n_heads, total_examples], device='cuda:3') * -1
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
thresh = 0.4 
n_heads = int((accuracy > thresh).float().sum().cpu().item())
print(f"Found {n_heads} successor heads")
if n_heads:
    layer, head = torch.where(accuracy > 0.4)
    heads = [(l,h) for l,h in zip(layer.cpu().tolist(), head.cpu().tolist())]
    print(heads)
    print(accuracy[layer, head].cpu().tolist())
# %%
if False:
    v = idx_dataset['days']
    input_ids = v[:-1]
    target_ids = v[1:]

    input_tensor = torch.tensor(input_ids)
    target_tensor = torch.tensor(target_ids)

    all_targets_tensor = torch.tensor(list(set(v)))

    target_logits = vocab_up_down_weight[:, :, input_tensor, target_tensor]
    all_target_logits = vocab_up_down_weight[:, :, input_tensor][:, :, :, all_targets_tensor]

    all_targets_max_logits = all_target_logits.max(dim=-1).values
    success = target_logits == all_targets_max_logits
