#%%
from pathlib import Path 
from typing import Optional

import pandas as pd
import torch
from transformers import PreTrainedTokenizer
#%%
def create_dataset(tokenizer: PreTrainedTokenizer, size: Optional[int] = None):
    ML_files = ['sva_raw_data/ML-subj_rel_all.csv',
    'sva_raw_data/ML-prep_inanim_all.csv',
    'sva_raw_data/ML-obj_rel_within_anim_all.csv',
    'sva_raw_data/ML-obj_rel_no_comp_within_inanim_all.csv',
    'sva_raw_data/ML-sent_comp_all.csv',
    'sva_raw_data/ML-obj_rel_no_comp_across_inanim_all.csv',
    'sva_raw_data/ML-obj_rel_within_inanim_all.csv',
    'sva_raw_data/ML-obj_rel_no_comp_within_anim_all.csv',
    'sva_raw_data/ML-obj_rel_across_anim_all.csv',
    'sva_raw_data/ML-obj_rel_across_inanim_all.csv',
    'sva_raw_data/ML-simple_agrmt_all.csv',
    'sva_raw_data/ML-long_vp_coord_all.csv',
    'sva_raw_data/ML-prep_anim_all.csv',
    'sva_raw_data/ML-vp_coord_all.csv',
    'sva_raw_data/ML-obj_rel_no_comp_across_anim_all.csv']

    dfs = []
    for ML_file in ML_files:
        
        df = pd.read_csv(Path(__file__).parent / ML_file)
        d = {'sentence_singular': df['sentence'][df['label']==0].values, 'sentence_plural': df['sentence'][df['label']==1].values, 'group':[ML_file.split('/')[-1].split('.')[0]]*(len(df)//2) }
        new_df = pd.DataFrame.from_dict(d)
        dfs.append(new_df)

    big_df = pd.concat(dfs)

    sing_lens = tokenizer(big_df['sentence_singular'].tolist(), return_tensors='pt', padding='longest').attention_mask.sum(-1)
    plur_lens = tokenizer(big_df['sentence_plural'].tolist(), return_tensors='pt', padding='longest').attention_mask.sum(-1)
    same_lens = sing_lens == plur_lens
    same_lens = same_lens.numpy()
    print(same_lens.sum(), '/', len(big_df))

    big_df = big_df[same_lens]
    sing_enc = tokenizer(big_df['sentence_singular'].tolist(), return_tensors='pt', padding='longest')
    sing_toks = sing_enc.input_ids
    sing_pos = sing_enc.attention_mask.sum(-1) - 1

    plur_enc = tokenizer(big_df['sentence_plural'].tolist(), return_tensors='pt', padding='longest')
    plur_toks = plur_enc.input_ids
    plur_pos = plur_enc.attention_mask.sum(-1) - 1

    max_len = sing_toks.size(1)

    clean = torch.cat((sing_toks, plur_toks), dim=0)
    corrupted = torch.cat((plur_toks, sing_toks), dim=0)
    labels = torch.tensor(([0] * len(sing_toks)) + ([1] * len(plur_toks)))
    pos = torch.cat((sing_pos, plur_pos), dim=0)

    if size is None:
        size = len(clean)

    random_indices = torch.randperm(len(clean))
    clean = clean[random_indices][:size]
    corrupted = corrupted[random_indices][:size]
    labels = labels[random_indices][:size]
    pos = pos[random_indices][:size]

    return clean, corrupted, labels, max_len, pos
    
