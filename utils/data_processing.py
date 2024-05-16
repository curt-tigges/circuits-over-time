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
from typing import Optional, List
#from joypy import joyplot

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def load_faithfulness_scores_into_df(folder_path, seed_name='seed1234'):
    file_paths = glob.glob(f'{folder_path}/*.json')

    # Create an empty DataFrame to store all edge scores
    all_sizes = pd.DataFrame()

    for i, file_path in enumerate(file_paths):
        print(f'Processing file {i+1}/{len(file_paths)}: {file_path}')
        data = read_json_file(file_path)
        sizes = data.keys()
        scores = [data[size] for size in sizes]

        # Extract checkpoint name from the filename
        checkpoint_name = int(os.path.basename(file_path).replace('.json', ''))
        #checkpoint_name = f'step {checkpoint_name}'

        checkpoint_df = pd.DataFrame({'size': sizes, 'faithfulness_score': scores, 'checkpoint': checkpoint_name, 'seed': seed_name})
        all_sizes = pd.concat([all_sizes, checkpoint_df])

        #ensure size and checkpoint are integer columns
        all_sizes['size'] = all_sizes['size'].astype(int)
        all_sizes['checkpoint'] = all_sizes['checkpoint'].astype(int)


    # sort by checkpoint and then by size
    all_sizes = all_sizes.sort_values(by=['seed', 'checkpoint', 'size'])
    return all_sizes


def load_graphs_for_models(target_directory, TASK):
    df_list = []
    for root, dirs, files in os.walk(target_directory):
        for dir in dirs:
            folder_path = f'results/graphs/{dir}/{TASK}'
            if os.path.isdir(folder_path):
                df = load_edge_scores_into_dictionary(folder_path)
                df['subfolder'] = dir  # Add the subfolder name as a new column
                df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df



def load_edge_scores_into_dictionary(folder_path, checkpoint=None):
    file_paths = glob.glob(f'{folder_path}/*.json')

    # Create an empty DataFrame to store all edge scores
    all_edges = pd.DataFrame()

   
    for i, file_path in enumerate(file_paths):
        checkpoint_name = int(os.path.basename(file_path).replace('.json', ''))
        if checkpoint is not None and checkpoint_name != checkpoint:
            continue

        print(f'Processing file {i+1}/{len(file_paths)}: {file_path}')
        data = read_json_file(file_path)
        edges = data['edges']
        scores = [edge['score'] for edge in edges.values()]
        circuit_inclusion = [edge['in_graph'] for edge in edges.values()]
        edge_names = [edge for edge in edges.keys()]

        checkpoint_df = pd.DataFrame({'edge': edge_names, 'score': scores, 'in_circuit': circuit_inclusion, 'checkpoint': checkpoint_name})
        all_edges = pd.concat([all_edges, checkpoint_df])
        print(all_edges)

    all_edges = all_edges.sort_values('checkpoint')

    return all_edges

def load_node_dictionary(folder_path, checkpoint=None):
    print(folder_path)
    file_paths = glob.glob(f'{folder_path}/*.json')
    all_nodes = {}
    all_nodes['checkpoint'] = []
    all_nodes['num_nodes'] = []
    for i, file_path in enumerate(file_paths):
        checkpoint_name = int(os.path.basename(file_path).replace('.json', ''))
        if checkpoint is not None and checkpoint_name != checkpoint:
            continue
        data = read_json_file(file_path)
        nodes = len([i for i in list(data["nodes"].keys()) if data["nodes"][i]])
        all_nodes['checkpoint'].append(checkpoint_name)
        all_nodes['num_nodes'].append(nodes)
    
    return pd.DataFrame.from_dict(all_nodes).sort_values(by = ['checkpoint'])

def get_ckpts(schedule: str) -> List[int]:
    """Get the list of checkpoints to use based on the schedule.
    
    Args:
        schedule (str): The schedule to use.

    Returns:
        List[int]: The list of checkpoints to use.
    """
    print(f"Received schedule: {schedule}")
    if schedule == "all":
        ckpts = (
            [0]
            + [2**i for i in range(10)]
            + [i * 1000 for i in range(1, 144)]
        )
    elif schedule == "linear":
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
    else:
        ckpts = [10000, 143000]

    return ckpts


def convert_checkpoint_steps_to_tokens(checkpoints: List[int]) -> List[int]:
    """Convert checkpoint steps to tokens.

    Args:
        checkpoints (List[int]): The list of checkpoint steps.

    Returns:
        List[int]: The list of tokens.
    """
    return [ckpt * 2097152 for ckpt in checkpoints]


def generate_in_circuit_df_files(
        graphs_folder: str,
        start_checkpoint: int = 4000, 
        limit_to_model: Optional[str] = None, 
        limit_to_task: Optional[str] = None
    ) -> None:
    """Generate in_circuit_edges.feather files for each model and task in the graphs_folder.
    
    This speeds up loading and analysis of the in-circuit edges by pre-filtering the data.

    Args:
        graphs_folder (str): The folder containing the edge score JSON files.
        start_checkpoint (int): The checkpoint number to start from.
        limit_to_model (Optional[str]): If not None, only process this model.
        limit_to_task (Optional[str]): If not None, only process this task.

    Returns:
        None
    """

    for model_folder in os.listdir(graphs_folder):
        model_folder_path = os.path.join(graphs_folder, model_folder)
        if os.path.isdir(model_folder_path):  # Check if it's a directory
            for task_folder in os.listdir(model_folder_path):
                print(f"{model_folder_path}")
                task_folder_path = os.path.join(model_folder_path, task_folder)
                # append the subfolder raw to the path
                task_folder_path = os.path.join(task_folder_path, 'raw')
                if os.path.isdir(task_folder_path): 
                    if limit_to_model is not None and model_folder != limit_to_model:
                        continue
                    if limit_to_task is not None and task_folder != limit_to_task:
                        continue
                    
                    folder_path = f'{graphs_folder}/{model_folder}/{task_folder}/raw'
                    df = load_edge_scores_into_dictionary(folder_path)
                    df = df[df['checkpoint'] >= start_checkpoint]

                    in_circuit_df = df[df['in_circuit'] == True]

                    in_circuit_df.reset_index(drop=True, inplace=True)
                    in_circuit_df.to_feather(f'{graphs_folder}/{model_folder}/{task_folder}/in_circuit_edges.feather')



def load_metrics(directory: str) -> dict:
    """Load the performance metrics generated by get_model_task_performance.

    Args:
        directory (str): The directory containing the metrics.pt files.

    Returns:
        dict: A nested dictionary containing the performance metrics.
    """
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


def aggregate_metrics_to_tensors_step_number(root_folder: str, shot_name: str = "zero-shot") -> dict:
    """Used to aggregate metrics collected by LM Harness during original model training runs
    
    Args:
        root_folder (str): The root folder containing the model folders.
        shot_name (str): The name of the shot folder to look for.

    Returns:
        dict: A nested dictionary containing the performance metrics.
    """
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
        # if checkpoint == reference_checkpoint:
        #     continue

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


def compute_weighted_jaccard_similarity(df):
    # Ensure the dataframe is sorted by checkpoint
    df = df.sort_values(by='checkpoint')

    # Normalize the scores by dividing by the sum of absolute scores within each checkpoint
    df['normalized_abs_score'] = df.groupby('checkpoint')['score'].transform(lambda x: x.abs() / x.abs().sum())

    # Get the unique checkpoints
    checkpoints = df['checkpoint'].unique()

    # Initialize a list to store the results
    results = []

    # Iterate over pairs of adjacent checkpoints
    for i in range(len(checkpoints) - 1):
        # Get the data for each checkpoint
        df_1 = df[(df['checkpoint'] == checkpoints[i]) & (df['in_circuit'] == True)]
        df_2 = df[(df['checkpoint'] == checkpoints[i + 1]) & (df['in_circuit'] == True)]

        # Create dictionaries mapping edges to their normalized absolute scores
        scores_1 = dict(zip(df_1['edge'], df_1['normalized_abs_score']))
        scores_2 = dict(zip(df_2['edge'], df_2['normalized_abs_score']))

        # Calculate the weighted intersection and union
        weighted_intersection = sum(min(scores_1.get(edge, 0), scores_2.get(edge, 0)) for edge in set(scores_1) | set(scores_2))
        weighted_union = sum(max(scores_1.get(edge, 0), scores_2.get(edge, 0)) for edge in set(scores_1) | set(scores_2))

        # Calculate the weighted Jaccard similarity
        weighted_jaccard_similarity = weighted_intersection / weighted_union if weighted_union != 0 else 0

        # Append the results for this pair of checkpoints
        results.append({
            'checkpoint_1': checkpoints[i],
            'checkpoint_2': checkpoints[i + 1],
            'jaccard_similarity': weighted_jaccard_similarity
        })

    # Convert the results to a DataFrame and return
    return pd.DataFrame(results)


def compute_weighted_jaccard_similarity_to_reference(df, reference_checkpoint):
    # Ensure the dataframe is sorted by checkpoint
    df = df.sort_values(by='checkpoint')

    # Normalize the scores by dividing by the sum of absolute scores within each checkpoint
    df['normalized_abs_score'] = df.groupby('checkpoint')['score'].transform(lambda x: x.abs() / x.abs().sum())

    # Get the unique checkpoints
    checkpoints = df['checkpoint'].unique()

    # Initialize a list to store the results
    results = []

    # Get the data for the reference checkpoint
    df_reference = df[(df['checkpoint'] == reference_checkpoint) & (df['in_circuit'] == True)]

    # Create a dictionary mapping edges to their normalized absolute scores for the reference checkpoint
    scores_reference = dict(zip(df_reference['edge'], df_reference['normalized_abs_score']))

    # Iterate over all checkpoints
    for checkpoint in checkpoints:
        # Get the data for the current checkpoint
        df_checkpoint = df[(df['checkpoint'] == checkpoint) & (df['in_circuit'] == True)]

        # Create a dictionary mapping edges to their normalized absolute scores for the current checkpoint
        scores_checkpoint = dict(zip(df_checkpoint['edge'], df_checkpoint['normalized_abs_score']))

        # Calculate the weighted intersection and union
        weighted_intersection = sum(min(scores_reference.get(edge, 0), scores_checkpoint.get(edge, 0)) for edge in set(scores_reference) | set(scores_checkpoint))
        weighted_union = sum(max(scores_reference.get(edge, 0), scores_checkpoint.get(edge, 0)) for edge in set(scores_reference) | set(scores_checkpoint))

        # Calculate the weighted Jaccard similarity
        weighted_jaccard_similarity = weighted_intersection / weighted_union if weighted_union != 0 else 0

        # Append the results for this checkpoint
        results.append({
            'reference_checkpoint': reference_checkpoint,
            'checkpoint': checkpoint,
            'jaccard_similarity': weighted_jaccard_similarity
        })

    # Convert the results to a DataFrame and return
    return pd.DataFrame(results)


def compute_ewma_weighted_jaccard_similarity(df, alpha=0.5):
    # Ensure the dataframe is sorted by checkpoint
    df = df.sort_values(by='checkpoint')

    # Normalize the scores by dividing by the sum of absolute scores within each checkpoint
    df['normalized_abs_score'] = df.groupby('checkpoint')['score'].transform(lambda x: x.abs() / x.abs().sum())

    # Get the unique checkpoints
    checkpoints = df['checkpoint'].unique()

    # Initialize a list to store the results
    results = []

    # Initialize the previous EWMA value
    prev_ewma = 0

    # Iterate over pairs of adjacent checkpoints
    for i in range(len(checkpoints) - 1):
        # Get the data for each checkpoint
        df_1 = df[(df['checkpoint'] == checkpoints[i]) & (df['in_circuit'] == True)]
        df_2 = df[(df['checkpoint'] == checkpoints[i + 1]) & (df['in_circuit'] == True)]

        # Create dictionaries mapping edges to their normalized absolute scores
        scores_1 = dict(zip(df_1['edge'], df_1['normalized_abs_score']))
        scores_2 = dict(zip(df_2['edge'], df_2['normalized_abs_score']))

        # Calculate the weighted intersection and union
        weighted_intersection = sum(min(scores_1.get(edge, 0), scores_2.get(edge, 0)) for edge in set(scores_1) | set(scores_2))
        weighted_union = sum(max(scores_1.get(edge, 0), scores_2.get(edge, 0)) for edge in set(scores_1) | set(scores_2))

        # Calculate the weighted Jaccard similarity
        weighted_jaccard_similarity = weighted_intersection / weighted_union if weighted_union != 0 else 0

        # Calculate the change rate
        change_rate = 1 - weighted_jaccard_similarity

        # Update the EWMA value
        ewma = alpha * change_rate + (1 - alpha) * prev_ewma
        prev_ewma = ewma

        # Append the results for this pair of checkpoints
        results.append({
            'checkpoint_1': checkpoints[i],
            'checkpoint_2': checkpoints[i + 1],
            'jaccard_similarity': weighted_jaccard_similarity,
            'ewma_change_rate': ewma
        })

    # Convert the results to a DataFrame and return
    return pd.DataFrame(results)
