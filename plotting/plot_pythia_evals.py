# %%
import os
import sys
import torch
import json
import re 
import pandas as pd
import plotly.express as px
import transformer_lens.utils as utils

import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

sys.path.append("..")

torch.set_grad_enabled(False)

# %%
def plot_lines(data_series, labels, colors, x_vals, x_title='Iterations', y_title='Accuracy', title=None, log=True, file_name='plot.pdf', show_legend=True, legend_font_size=16):
    fig = go.Figure()

    for series, label, color in zip(data_series, labels, colors):
        fig.add_trace(go.Scatter(x=x_vals, y=series, mode='lines', name=label, line=dict(color=color)))
    
    type = 'log' if log else 'linear'
    fig.update_xaxes(type=type, title_text=x_title, title_font={"size": 22})
    fig.update_yaxes(title_text=y_title, title_font={"size": 22})
    
    # Update layout to adjust margins, optionally show/hide the legend, and increase the legend font size
    fig.update_layout(
        title=title if title is not None else None,  # Set title to None to omit it
        width=700,
        height=400,
        margin=dict(l=50, r=50, t=35, b=50),  # Reduce the top margin (t) to minimize whitespace
        showlegend=show_legend,  # Control display of the legend based on the show_legend parameter
        legend=dict(font=dict(size=legend_font_size))  # Increase the legend font size
    )

    # To display the figure in the notebook or in a browser
    fig.show()

    # To save the figure to a file
    fig.write_image(file_name, format='pdf', width=700, height=400, engine="kaleido")

def clean_json_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Define the regular expressions to remove unwanted lines
    patterns = [
        re.compile(r'^<<<<<<< HEAD.*$', re.MULTILINE),
        re.compile(r'^version https://git-lfs.github.com/spec/v1.*$', re.MULTILINE),
        re.compile(r'^oid sha256:.*$', re.MULTILINE),
        re.compile(r'^size \d+.*$', re.MULTILINE),
        re.compile(r'^=======$', re.MULTILINE),
        re.compile(r'^>>>>>>>.*$', re.MULTILINE)
    ]
    
    # Remove the unwanted lines
    for pattern in patterns:
        content = pattern.sub('', content)
    
    with open(file_path, 'w') as file:
        file.write(content)

# Main function to traverse directories and aggregate metrics
def aggregate_metrics_to_tensors_step_number(root_folder, shot_name="zero-shot"):
    metric_dictionary = {}
    
    # Regular expression to extract step number from checkpoint name
    step_number_pattern = re.compile(r'_step(\d+)')
    
    # Iterate through the folder structure
    for model_folder in os.listdir(root_folder):
        print(model_folder)
        model_path = os.path.join(root_folder, model_folder)
        if not os.path.isdir(model_path):
            continue  # Skip any files at this level
        
        shot_path = os.path.join(model_path, shot_name)
        if not os.path.exists(shot_path) or not os.path.isdir(shot_path):
            continue  # Skip if the specific shot folder doesn't exist
        
        for filename in os.listdir(shot_path):
            if filename.endswith('.json'):
                file_path = os.path.join(shot_path, filename)
                
                # Clean the JSON file before processing
                clean_json_file(file_path)
                #print(file_path)
                
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

def create_new_dict_with_tokens(metric_dictionary, checkpoints):
    # Convert checkpoint steps to tokens
    token_list = convert_checkpoint_steps_to_tokens(checkpoints)
    checkpoint_to_token = dict(zip(checkpoints, token_list))
    
    # Create a new dictionary with token keys
    new_metric_dictionary = {}
    for model_name, tasks in metric_dictionary.items():
        new_metric_dictionary[model_name] = {}
        for task_name, metrics in tasks.items():
            new_metric_dictionary[model_name][task_name] = {}
            for metric_name, checkpoint_values in metrics.items():
                new_checkpoint_values = {str(checkpoint_to_token[step]): value for step, value in checkpoint_values.items()}
                new_metric_dictionary[model_name][task_name][metric_name] = new_checkpoint_values
    
    return new_metric_dictionary

                

# %%
pythia_evals = aggregate_metrics_to_tensors_step_number("../results/task_performance_metrics/pythia-evals/pythia-v1")
pythia_ckpts = [int(k) for k in pythia_evals['pythia-1.4b']['hendrycksTest-abstract_algebra']['acc'].keys()]
pythia_ckpts.sort()

# %%
from utils.data_processing import convert_checkpoint_steps_to_tokens
pythia_tkns = convert_checkpoint_steps_to_tokens(pythia_ckpts)
pythia_evals_tkns = create_new_dict_with_tokens(pythia_evals, [str(c) for c in pythia_ckpts])

# %%
eval_choices = {"sciq": 4, "piqa": 2, "winogrande": 2, "arc_easy": 4}
selected_models = ['pythia-70m', 'pythia-160m', 'pythia-410m', 'pythia-1.4b', 'pythia-2.8b', 'pythia-6.9b', 'pythia-12b']
selected_evals = ['sciq', 'piqa', 'winogrande', 'arc_easy']

task_name = None
metric_name = None
task_counter = 0
for task in selected_evals:
    metric_name = list(pythia_evals['pythia-70m'][task].keys())[0]
    if metric_name != 'acc' or torch.stack([pythia_evals['pythia-12b'][task][metric_name][str(k)] for k in pythia_evals['pythia-12b'][task][metric_name]]).max() < 0.4:
        continue
    task_counter += 1
    data_series = []
    for model in selected_models:
        ckpt_dict = pythia_evals[model][task][metric_name]
        data_series.append(torch.stack([ckpt_dict[str(k)] for k in pythia_tkns]))
        task_name = task
    
    labels = ['70M','160M', '410M', '1.4B', '2.8B', '6.9B', '12B']
    colors = ['green', 'orange', 'purple', 'brown', 'black', 'pink', 'gray']

    #plot_lines(data_series, labels, colors, x_vals=pythia_ckpts, x_title='Steps', y_title='Metric Value', title=f'{task_name.upper()} {metric_name.capitalize()} Over Training Time (Log Scale)')
    plot_lines(data_series, labels, colors, x_vals=pythia_tkns, x_title='Steps', y_title=f'Accuracy (Out of {eval_choices[task]})', title=None, file_name=f'{task_name}-{metric_name}.pdf') #f'{task_name.upper()} {metric_name.capitalize()} Over Training Time (Log Scale)')
    plot_lines(data_series, labels, colors, x_vals=pythia_tkns, x_title='Steps', y_title=f'Accuracy (Out of {eval_choices[task]})', title=None, file_name=f'{task_name}-{metric_name}-nolegend.pdf', show_legend=False) #f'{task_name.upper()} {metric_name.capitalize()} Over Training Time (Log Scale)')

print(task_counter)

# %%
print(pythia_evals_tkns['pythia-12b']['sciq']['acc'].keys())
# %%
