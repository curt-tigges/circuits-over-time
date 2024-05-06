#%%
from pathlib import Path 

import numpy as np
import torch 
import matplotlib.pyplot as plt

#%%
p = Path('results/components')
for subdir in p.iterdir():
    data_file =  subdir / 'components_over_time.pt'
    if not data_file.exists():
        continue
    data = torch.load(data_file)
    steps = list(data.keys())
    steps.sort(key=lambda x: int(x))

    for tertiary_head in ['induction_scores', 'duplicate_token_scores', 'prev_token_scores']:
        head_scores = torch.stack([data[step]['tertiary_head_scores'][tertiary_head].cpu() for step in steps])

        fig, ax = plt.subplots()
        all_head_idxs = head_scores.view(head_scores.size(0), -1).argmax(-1)
        layers, heads = np.unravel_index(all_head_idxs, head_scores[-1].shape)
        all_heads = set(zip(layers, heads))
        for layer, head in all_heads:
            top_head_scores = head_scores[:, layer, head].numpy()
            ax.plot(steps, top_head_scores, label=f"({layer}, {head})")

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5)
        ax.set_xlabel('Step')
        ax.set_ylabel(f'Accuracy on {tertiary_head} Task')
        ax.set_title(f"{tertiary_head} Behavior Over Time ({subdir.name})")
        fig.show()
        save_dir = Path(f'results/plots/{tertiary_head}')
        save_dir.mkdir(exist_ok=True, parents=True)
        fig.savefig(save_dir / f'{subdir.name}.pdf')
# %%
