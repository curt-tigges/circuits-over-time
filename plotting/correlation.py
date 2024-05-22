import os
import json
import glob
import re
import sys
from typing import Dict, List

import torch
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
#import chart_studio
#from chart_studio import plotly as py

from IPython.display import display, clear_output, HTML
'''
sys.path.append("..")
from utils.data_processing import (
    load_edge_scores_into_dictionary,
    compute_weighted_jaccard_similarity,
    compute_node_jaccard_similarity,
    compute_jaccard_similarity,
    compute_weighted_jaccard_similarity_to_reference,
    compute_ewma_weighted_jaccard_similarity,
    generate_in_circuit_df_files,
    load_node_dictionary,
    convert_checkpoint_steps_to_tokens
)

from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import numpy as np
import pandas as pd
import yaml
from typing import Tuple, List, Dict


color_palette = {
  "pythia-70m": "#EE908D",
  "pythia-160m": "#F8D592",
  #"pythia-1b": "#8CD9AF",
  "pythia-410m": "#B2B4D9",
  #"pythia-12b": "#B46F90",
  "pythia-1.4b": "#A7C2D0",
  #"pythia-6.9b": "#BF8271",
  "pythia-2.8b": "#8CD9AF"
}

core_models = list(color_palette.keys())



def generate_in_circuit_df_files(
        graphs_folder: str,
        task: str,
        start_checkpoint: int = 1, 
        limit_to_model: Optional[str] = None, 
        limit_to_task: Optional[str] = None
    ) -> None:
        for model_folder in os.listdir(graphs_folder):
                model_folder_path = os.path.join(graphs_folder, model_folder)
                if os.path.isdir(model_folder_path):  # Check if it's a directory
                        for task_folder in os.listdir(model_folder_path):
                                task_folder_path = os.path.join(model_folder_path, task_folder)
                                if task == "gender_pronoun":
                                        task_folder_path = os.path.join(task_folder_path, 'faithful')
                                # append the subfolder raw to the path
                                task_folder_path = os.path.join(task_folder_path, 'raw')
                                if os.path.isdir(task_folder_path): 
                                        if limit_to_model is not None and model_folder != limit_to_model:
                                                continue
                                        if limit_to_task is not None and task_folder != limit_to_task:
                                                continue
                                        if task == "gender_pronoun":
                                                folder_path = f'{graphs_folder}/{model_folder}/{task_folder}/faithful/raw'
                                        else:
                                                folder_path = f'{graphs_folder}/{model_folder}/{task_folder}/raw'
                                        if folder_path == f'{graphs_folder}/pythia-1.4b/sva/raw':
                                               folder_path = f'{graphs_folder}/{model_folder}/{task_folder}/faithful/raw'
                                        if folder_path == f'{graphs_folder}/pythia-2.8b/sva/raw':
                                               folder_path = f'{graphs_folder}/{model_folder}/{task_folder}/faithful/raw'
                                        dic = load_node_dictionary(folder_path)
                                        zero_dic = dic[dic['num_nodes'] == 0]
                                        return dic
                
        

correlation = {}
for i in core_models:
    correlation[i] = {}
    for j in ['ioi', 'greater_than', 'gender_pronoun', 'sva']:
        correlation[i][j] = 0
def generate_results_nodes(model, TASK, start_checkpoint):
    list_of_df = {}
    for m in model:
        in_circuit_df = generate_in_circuit_df_files('/mnt/hdd-0/circuits-over-time/results/graphs', task = TASK, start_checkpoint=1, limit_to_model=m, limit_to_task=TASK)       
        if in_circuit_df is not None:
            in_circuit_df = in_circuit_df[in_circuit_df['checkpoint'] >= start_checkpoint]
            list_of_df[m] = in_circuit_df  
            in_circuit_df['checkpoint'] = convert_checkpoint_steps_to_tokens(in_circuit_df['checkpoint'])

            correlation[m][TASK] = in_circuit_df['num_nodes'].mean()
    return list_of_df

generate_results_nodes(core_models, 'ioi', 1000)
generate_results_nodes(core_models, 'gender_pronoun', 1000)
generate_results_nodes(core_models, 'greater_than', 1000)
generate_results_nodes(core_models, 'sva', 1000)

print(correlation)
'''

nodes = {'pythia-70m': {'ioi': 28.916666666666668, 'greater_than': 13.24822695035461, 'gender_pronoun': 17.666666666666668, 'sva': 14.965034965034965}, 'pythia-160m': {'ioi': 43.255319148936174, 'greater_than': 17.323943661971832, 'gender_pronoun': 23.696969696969695, 'sva': 34.66433566433567}, 'pythia-410m': {'ioi': 88.2867132867133, 'greater_than': 36.70422535211268, 'gender_pronoun': 39.27272727272727, 'sva': 51.84496124031008}, 'pythia-1.4b': {'ioi': 46.64, 'greater_than': 35.04195804195804, 'gender_pronoun': 56.56, 'sva': 93.93939393939394}, 'pythia-2.8b': {'ioi': 135.44, 'greater_than': 72.75, 'gender_pronoun': 42.333333333333336, 'sva': 97.84848484848484}}

dic = pd.DataFrame.from_dict(nodes).T
print(dic)

dic['size'] = [70000000, 160000000, 410000000, 14000000000, 28000000000]

print(dic['ioi'].corr(dic['size']))

print(dic['greater_than'].corr(dic['size']))

print(dic['gender_pronoun'].corr(dic['size']))

print(dic['sva'].corr(dic['size']))