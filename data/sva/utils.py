from typing import Optional, List, Union, Literal, Tuple
from pathlib import Path 


import pandas as pd
import torch 

def get_singular_and_plural(model, strict=False) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = model.tokenizer
    tokenizer_length = model.cfg.d_vocab_out

    df: pd.DataFrame = pd.read_csv(Path(__file__).parent / 'combined_verb_list.csv')
    singular = df['sing'].to_list()
    plural = df['plur'].to_list()
    singular_set = set(singular)
    plural_set = set(plural)
    verb_set = singular_set | plural_set
    assert len(singular_set & plural_set) == 0, f"{singular_set & plural_set}"
    singular_indices, plural_indices = [], []

    for i in range(tokenizer_length):
        token = tokenizer._convert_id_to_token(i)
        if token is not None:
            if token[0] == 'Ġ':
                token = token[1:]
                if token in verb_set:    
                    if token in singular_set:
                        singular_indices.append(i)
                    else:  # token in plural_set:
                        idx = plural.index(token)
                        third_person_present = singular[idx]
                        third_person_present_tokenized = tokenizer(f' {third_person_present}', add_special_tokens=False)['input_ids']
                        if len(third_person_present_tokenized) == 1 and third_person_present_tokenized[0] != tokenizer.unk_token_id:
                            plural_indices.append(i)
                        elif not strict:
                            plural_indices.append(i)
               
    return torch.tensor(singular_indices), torch.tensor(plural_indices)
