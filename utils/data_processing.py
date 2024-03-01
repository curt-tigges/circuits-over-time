import os
import json
import glob
import torch
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
#import seaborn as sns
import matplotlib.pyplot as plt
#from joypy import joyplot

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def load_edge_scores_into_dictionary(folder_path):
    file_paths = glob.glob(f'{folder_path}/*.json')

    # Create an empty DataFrame to store all edge scores
    all_edges = pd.DataFrame()

    for i, file_path in enumerate(file_paths):
        print(f'Processing file {i+1}/{len(file_paths)}: {file_path}')
        data = read_json_file(file_path)
        edges = data['edges']
        scores = [edge['score'] for edge in edges.values()]
        circuit_inclusion = [edge['in_graph'] for edge in edges.values()]
        edge_names = [edge for edge in edges.keys()]

        # Extract checkpoint name from the filename
        checkpoint_name = int(os.path.basename(file_path).replace('.json', ''))
        #checkpoint_name = f'step {checkpoint_name}'

        checkpoint_df = pd.DataFrame({'edge': edge_names, 'score': scores, 'in_circuit': circuit_inclusion, 'checkpoint': checkpoint_name})
        all_edges = pd.concat([all_edges, checkpoint_df])

    all_edges = all_edges.sort_values('checkpoint')
    return all_edges


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
            [i * 1000 for i in range(1, 16)]
            + [i * 5000 for i in range(3, 14)]
            + [i * 10000 for i in range(7, 15)]
        )
    else:
        ckpts = [10000, 143000]

    return ckpts


def load_metrics(directory):
    nested_dict = {}

    for model_folder in os.listdir(directory):
        model_path = os.path.join(directory, model_folder)
        if os.path.isdir(model_path) and model_folder != '.ipynb_checkpoints':
            model_key = '-'.join(model_folder.split('-')[:-2])
            metrics_file = os.path.join(model_path, 'metrics.pt')

            if os.path.exists(metrics_file):
                nested_dict[model_key] = torch.load(metrics_file)

                for task_key in nested_dict[model_key].keys():
                    for metric_key in nested_dict[model_key][task_key].keys():
                        metric_values = nested_dict[model_key][task_key][metric_key]
                        # Convert each value to a 1D tensor of regular floats
                        nested_dict[model_key][task_key][metric_key] = torch.tensor([
                            float(value.item() if torch.is_tensor(value) else value) 
                            for value in metric_values.values()
                        ])

    return nested_dict


def aggregate_metrics_to_tensors_step_number(root_folder, shot_name="zero-shot"):
    metric_dictionary = {}
    
    # Regular expression to extract step number from checkpoint name
    step_number_pattern = re.compile(r'_step(\d+)')
    
    # Iterate through the folder structure
    for model_folder in os.listdir(root_folder):
        model_path = os.path.join(root_folder, model_folder)
        if not os.path.isdir(model_path):
            continue  # Skip any files at this level
        
        shot_path = os.path.join(model_path, shot_name)
        if not os.path.exists(shot_path) or not os.path.isdir(shot_path):
            continue  # Skip if the specific shot folder doesn't exist
        
        for filename in os.listdir(shot_path):
            if filename.endswith('.json'):
                file_path = os.path.join(shot_path, filename)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    for task_name, metrics in data['results'].items():
                        for metric_name, value in metrics.items():
                            if model_folder not in metric_dictionary:
                                metric_dictionary[model_folder] = {}
                            if task_name not in metric_dictionary[model_folder]:
                                metric_dictionary[model_folder][task_name] = {}
                            if metric_name not in metric_dictionary[model_folder][task_name]:
                                metric_dictionary[model_folder][task_name][metric_name] = {}
                            
                            # Extract step number from filename
                            match = step_number_pattern.search(filename)
                            if match:
                                step_number = match.group(1)
                                metric_dictionary[model_folder][task_name][metric_name][step_number] = value
    
    # Convert values to PyTorch tensors
    for model_name, tasks in metric_dictionary.items():
        for task_name, metrics in tasks.items():
            for metric_name, checkpoint_values in metrics.items():
                # Convert the dictionary values to tensors, keys remain the same
                tensor_values = torch.tensor(list(checkpoint_values.values()))
                metric_dictionary[model_name][task_name][metric_name] = dict(zip(checkpoint_values.keys(), tensor_values))
    
    return metric_dictionary


def compute_ged(df):
    # Ensure the dataframe is sorted by checkpoint
    df = df.sort_values(by='checkpoint')

    # Get the unique checkpoints
    checkpoints = df['checkpoint'].unique()

    # Initialize a list to store the results
    results = []

    # Iterate over pairs of adjacent checkpoints
    for i in range(len(checkpoints) - 1):
        # Get the sets of edges for each checkpoint
        edges_1 = set(df[(df['checkpoint'] == checkpoints[i]) & (df['in_circuit'] == True)]['edge'])
        edges_2 = set(df[(df['checkpoint'] == checkpoints[i + 1]) & (df['in_circuit'] == True)]['edge'])

        # Calculate the number of additions and deletions
        additions = len(edges_2 - edges_1)
        deletions = len(edges_1 - edges_2)

        # Calculate the total GED
        total_ged = additions + deletions

        # Append the results for this pair of checkpoints
        results.append({
            'checkpoint_1': checkpoints[i],
            'checkpoint_2': checkpoints[i + 1],
            'additions': additions,
            'deletions': deletions,
            'total_ged': total_ged
        })

    # Convert the results to a DataFrame and return
    return pd.DataFrame(results)


def compute_weighted_ged(df):
    # Ensure the dataframe is sorted by checkpoint
    df = df.sort_values(by='checkpoint')

    # Get the unique checkpoints
    checkpoints = df['checkpoint'].unique()

    # Initialize a list to store the results
    results = []

    # Iterate over pairs of adjacent checkpoints
    for i in range(len(checkpoints) - 1):
        # Get the dataframes for each checkpoint
        df1 = df[(df['checkpoint'] == checkpoints[i]) & (df['in_circuit'] == True)]
        df2 = df[(df['checkpoint'] == checkpoints[i + 1]) & (df['in_circuit'] == True)]

        # Calculate the weighted additions and deletions
        additions = df2[~df2['edge'].isin(df1['edge'])]['score'].sum()
        deletions = df1[~df1['edge'].isin(df2['edge'])]['score'].sum()

        # Calculate the total weighted GED
        total_weighted_ged = additions + deletions

        # Append the results for this pair of checkpoints
        results.append({
            'checkpoint_1': checkpoints[i],
            'checkpoint_2': checkpoints[i + 1],
            'weighted_additions': additions,
            'weighted_deletions': deletions,
            'total_weighted_ged': total_weighted_ged
        })

    # Convert the results to a DataFrame and return
    return pd.DataFrame(results)


def compute_gtd(df):
    # Ensure the dataframe is sorted by checkpoint
    df = df.sort_values(by='checkpoint')

    # Get the unique checkpoints
    checkpoints = df['checkpoint'].unique()

    # Initialize a list to store the results
    results = []

    # Iterate over pairs of adjacent checkpoints
    for i in range(len(checkpoints) - 1):
        # Get the dataframes for each checkpoint
        df1 = df[df['checkpoint'] == checkpoints[i]]
        df2 = df[df['checkpoint'] == checkpoints[i + 1]]

        # Merge the dataframes on the edge column
        merged_df = pd.merge(df1, df2, on='edge', suffixes=('_1', '_2'))

        # Calculate the score changes
        merged_df['score_change'] = merged_df['score_2'] - merged_df['score_1']

        # Calculate the weighted additions and deletions
        additions = merged_df[merged_df['score_change'] > 0]['score_change'].sum()
        deletions = -merged_df[merged_df['score_change'] < 0]['score_change'].sum()

        # Calculate the total weighted GTD
        total_weighted_gtd = additions + deletions

        # Append the results for this pair of checkpoints
        results.append({
            'checkpoint_1': checkpoints[i],
            'checkpoint_2': checkpoints[i + 1],
            'summed_increases': additions,
            'summed_decreases': deletions,
            'summed_absolute_change': total_weighted_gtd
        })

    # Convert the results to a DataFrame and return
    return pd.DataFrame(results)


def compute_jaccard_similarity(df):
    # Ensure the dataframe is sorted by checkpoint
    df = df.sort_values(by='checkpoint')

    # Get the unique checkpoints
    checkpoints = df['checkpoint'].unique()

    # Initialize a list to store the results
    results = []

    # Iterate over pairs of adjacent checkpoints
    for i in range(len(checkpoints) - 1):
        # Get the sets of edges for each checkpoint
        edges_1 = set(df[(df['checkpoint'] == checkpoints[i]) & (df['in_circuit'] == True)]['edge'])
        edges_2 = set(df[(df['checkpoint'] == checkpoints[i + 1]) & (df['in_circuit'] == True)]['edge'])

        # Calculate the Jaccard similarity
        intersection = len(edges_1 & edges_2)
        union = len(edges_1 | edges_2)
        jaccard_similarity = intersection / union if union != 0 else 0

        # Append the results for this pair of checkpoints
        results.append({
            'checkpoint_1': checkpoints[i],
            'checkpoint_2': checkpoints[i + 1],
            'jaccard_similarity': jaccard_similarity
        })

    # Convert the results to a DataFrame and return
    return pd.DataFrame(results)


def compute_jaccard_similarity_to_reference(df, reference_checkpoint):
    # Ensure the dataframe is sorted by checkpoint
    df = df.sort_values(by='checkpoint')

    # Get the unique checkpoints
    checkpoints = df['checkpoint'].unique()

    # Initialize a list to store the results
    results = []

    # Get the set of edges for the reference checkpoint
    reference_edges = set(df[(df['checkpoint'] == reference_checkpoint) & (df['in_circuit'] == True)]['edge'])

    # Iterate over all checkpoints
    for checkpoint in checkpoints:
        # Skip the reference checkpoint
        if checkpoint == reference_checkpoint:
            continue

        # Get the set of edges for the current checkpoint
        edges = set(df[(df['checkpoint'] == checkpoint) & (df['in_circuit'] == True)]['edge'])

        # Calculate the Jaccard similarity
        intersection = len(reference_edges & edges)
        union = len(reference_edges | edges)
        jaccard_similarity = intersection / union if union != 0 else 0

        # Append the results for this checkpoint
        results.append({
            'reference_checkpoint': reference_checkpoint,
            'checkpoint': checkpoint,
            'jaccard_similarity': jaccard_similarity
        })

    # Convert the results to a DataFrame and return
    return pd.DataFrame(results)
