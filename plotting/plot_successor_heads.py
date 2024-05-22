#%%
from pathlib import Path 

import numpy as np
import torch 
import matplotlib.pyplot as plt
#%%
p = Path('/mnt/hdd-0/circuits-over-time/results/components')

fig, ax = plt.subplots()
x_axis = np.array([0, *(2**i for i in range(10)), *(1000 * i for i in range(1, 144))])
for subdir in p.iterdir():
    if subdir.name.count('-') > 1:
        continue 
    successor_head_scores = torch.load(subdir / 'successor_heads_over_time.pt')
    successor_head_idx = successor_head_scores[-1].argmax()
    layer, head = np.unravel_index(successor_head_idx, successor_head_scores[-1].shape)

    head_scores = successor_head_scores[:, layer, head].numpy()
    ax.plot(x_axis, head_scores, label=f"{subdir.name} ({layer}, {head})")

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
ax.set_xlabel('Step')
ax.set_ylabel('Accuracy on Successor Task')
ax.set_title("Successor Head Behavior Over Time")
fig.show()
# %%
p = Path('/mnt/hdd-0/circuits-over-time/results/components')
x_axis = np.array([0, *(2**i for i in range(10)), *(1000 * i for i in range(1, 144))])
print(list(p.iterdir()))
for subdir in p.iterdir():
    #if subdir.name.count('-') > 1:
    #    continue 
    fig, ax = plt.subplots()
    try:
        successor_head_dict = torch.load(subdir / 'successor_heads_over_time.pt')
    except FileNotFoundError:
        print("couldn't find file in", subdir)
        continue
    x_axis = successor_head_dict['checkpoints']
    successor_head_scores = successor_head_dict['data'].cpu()
    all_successor_head_idxs = successor_head_scores.view(successor_head_scores.size(0), -1).argmax(-1)
    layers, heads = np.unravel_index(all_successor_head_idxs, successor_head_scores[-1].shape)
    all_heads = set(zip(layers, heads))
    for layer, head in all_heads:
        head_scores = successor_head_scores[:, layer, head].numpy()
        ax.plot(x_axis, head_scores, label=f"({layer}, {head})")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy on Successor Task')
    ax.set_title(f"Successor Head Behavior Over Time ({subdir.name})")
    fig.show()
    save_dir = Path('results/plots/successor_heads')
    save_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_dir / f'{subdir.name}.pdf')
# %%
