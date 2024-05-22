import os
import pathlib
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
import numpy as np
import pandas as pd
import yaml
from typing import Tuple, List, Dict

import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
import chart_studio
from chart_studio import plotly as py

from utils.visualization import imshow_p

# HELPER FUNCTIONS
def convert_title_to_filename(title: str):
    # replace spaces with dashes, remove parentheses, and make lowercase
    return title.replace(' ', '-').replace('(', '').replace(')', '').lower()


def load_checkpoints(target_directory, filename='all_checkpoints.pt'):
    checkpoints_list = []
    for root, dirs, files in os.walk(target_directory):
        for dir in dirs:
            checkpoint_path = os.path.join(root, dir, filename)
            if os.path.exists(checkpoint_path):
                checkpoints = torch.load(checkpoint_path)
                checkpoints_list.append((dir, checkpoints))
    
    return checkpoints_list

# GRAPH PLOTTING FUNCTIONS
def plot_graph_metric(
        df: pd.DataFrame,
        metric: str,
        perf_metric_dict: Dict[int, float],
        title: Optional[str] = None,
        left_y_title: str = 'Primary Metric',
        right_y_title: str = 'Logit Diff',
        y_ranges: Tuple[Tuple[float, float], Tuple[float, float]] = ((0, 1), (0, 6)),
        fig_size: Tuple[int, int] = (800, 400),
        x_axis_col: str = 'checkpoint',
        log_x: bool = True,
        metric_legend_name: str = "Circuit Edges",
        enable_secondary_y: bool = True,
        legend_font_size: int = 16,
        axis_label_size: int = 16,
        output_path: str = "results/plots/graph_metrics/"
    ) -> None:
    if metric not in df.columns:
        raise ValueError(f"The metric '{metric}' does not exist in the dataframe.")

    if x_axis_col not in df.columns:
        raise ValueError(f"The x_axis column '{x_axis_col}' does not exist in the dataframe.")

    # Prepare the data
    df['perf_metric'] = df[x_axis_col].map(perf_metric_dict).interpolate(method='linear')
    plot_df = df.rename(columns={metric: metric_legend_name})

    # Plotting
    fig = px.line(plot_df, x=x_axis_col, y=[metric_legend_name], log_x=log_x,
                  title=None if title is None else title)

    # Update trace colors and add secondary y-axis trace if enabled
    if enable_secondary_y:
        fig.add_trace(
            go.Scatter(
                x=df[x_axis_col], y=df['perf_metric'], name=right_y_title,
                mode='lines', yaxis='y2',
                line=dict(color='gray', width=2, dash='dot')
            )
        )

    # Layout settings
    fig.update_layout(
        xaxis=dict(title="Training Checkpoint", title_font=dict(size=axis_label_size)),
        yaxis=dict(title=left_y_title, range=y_ranges[0], title_font=dict(size=axis_label_size)),
        yaxis2=dict(
            title=right_y_title, range=y_ranges[1], overlaying='y', side='right', showgrid=False,
            title_font=dict(size=axis_label_size)
        ) if enable_secondary_y else {},
        legend=dict(font=dict(size=legend_font_size), title_text="Metrics"),
        #width=fig_size[0], height=fig_size[1]
    )

    # Display and save the figure
    fig.show()
    if title:
        filename = output_path + convert_title_to_filename(title) + ".pdf"
        fig.write_image(filename, format='pdf', width=fig_size[0], height=fig_size[1], engine="kaleido")


# COMPONENT PLOTTING FUNCTIONS
def plot_head_circuit_scores( 
        checkpoint_dict: Dict[int, np.ndarray],  
        title: str, 
        limit_to_list: List = None,
        y_label='Metric Value',
        range_y=None,
        log_x=False,  # Added parameter to enable log scaling on x-axis
        fig_size: Tuple[int, int] = (700, 400),
        legend_font_size=16, 
        axis_label_size=16, 
        upload=False,
        show_legend=True, 
        disable_title=False,
        output_path: str = "results/plots/components/"
    ) -> pd.DataFrame:
    """
    Plot the circuit metrics scores for attention heads across checkpoints with optional logarithmic x-axis. This can be used to plot any metric that is
    stored in the checkpoint_dict with {checkpoint: numpy array} (shape: n_layers, n_heads) format. This function is more
    general than the others below.

    Args:
        model_name (str): The name of the model for titling the plot.
        checkpoint_dict (Dict[int, np.ndarray]): A dictionary mapping checkpoints to numpy arrays of head attributions.
        title (str): The title of the plot.
        limit_to_list (List): A list of tuples containing the layer and head indices to limit the plot to.
        log_x (bool): Whether to apply logarithmic scaling to the x-axis.
        upload (bool): Whether to upload the plot to Plotly Chart Studio.
        y_label (str): The label for the y-axis.
        show_legend (bool): Whether to display the legend.
        legend_font_size (int): The font size for the legend.
        axis_label_size (int): The font size for the axis labels.
        disable_title (bool): Whether to disable the title.
        range_y (Tuple[float, float]): The range for the y-axis.

    Returns:
        pd.DataFrame: A DataFrame containing the plot data.
    """
    # Define axis title style
    axis_title_style = dict(size=axis_label_size)
    
    # Determine display title based on `disable_title` flag
    display_title = None if disable_title else title

    plot_data = []

    # Iterate through each checkpoint
    for checkpoint, array in checkpoint_dict.items():
        # Iterate through each layer and head in the array
        for layer in range(array.shape[0]):
            for head in range(array.shape[1]):
                condition = True
                if limit_to_list:
                    condition = (layer, head) in limit_to_list
                if array[layer, head] != 0 and condition:
                    # Append the data for plotting
                    plot_data.append({
                        'Checkpoint': checkpoint,
                        'Layer-Head': f'Layer {layer}-Head {head}',
                        'Layer': layer,
                        'Head': head,
                        'Value': array[layer, head]
                    })

    # Convert to DataFrame
    df = pd.DataFrame(plot_data)

    # Plot the data
    fig = px.line(
        df,
        x='Checkpoint',
        y='Value',
        color='Layer-Head',
        range_y=range_y,
        log_x=log_x,  # Apply logarithmic scale if log_x is True
        title=display_title,
        labels={'Checkpoint': 'Checkpoint', 'Value': 'Metric Value'}
    )
    
    fig.update_layout(
        xaxis=dict(
            title="Training Checkpoint", 
            title_font=axis_title_style
        ),
        yaxis=dict( 
            title=y_label, 
            title_font=axis_title_style
        ),
        showlegend=show_legend,
        legend=dict(
            font=dict(size=legend_font_size),
            title_text='Attention Head',
        )
    )

    if upload:
        url = py.plot(fig, filename=title, auto_open=True)
        print(f"Plot uploaded to {url}")
    fig.show()

    filename = output_path + convert_title_to_filename(title) + ".pdf"
    fig.write_image(filename, format='pdf', width=fig_size[0], height=fig_size[1], engine="kaleido")

    return df


def plot_nmh_metrics(model_name: str, component_scores: dict, verbose: bool = False) -> None:
    
    ckpts = list(component_scores.keys())
    ckpts.sort()
    
    copy_scores = dict()
    filtered_copy_scores = dict()
    io_attns = dict()
    io_s1_attn_ratio = dict()
    copy_suppression_scores = dict()
    for ckpt in ckpts:
        if component_scores[ckpt]['direct_effect_scores'] is not None:
            copy_scores[ckpt] = component_scores[ckpt]['direct_effect_scores']['copy_scores']
            #filtered_copy_scores[ckpt] = component_scores[ckpt]['direct_effect_scores']['copy_scores']
            #io_attns[ckpt] = component_scores[ckpt]['direct_effect_scores']['io_attn_scores']
            #io_s1_attn_ratio[ckpt] = component_scores[ckpt]['direct_effect_scores']['io_attn_scores'] / component_scores[ckpt]['direct_effect_scores']['s1_attn_scores']
            #copy_suppression_scores[ckpt] = component_scores[ckpt]['direct_effect_scores']['copy_suppression_scores']
        else:
            if verbose:
                print(f"Skipping {ckpt} for model {model_name} due to absent data (probably no functioning heads of this type at this checkpoint)")
            continue

    # note that most, but not all of these are formally 'NMHs'; if attention to S1 exceeds attention to IO, they are not NMHs
    try:
        all_heads_copy_score = plot_head_circuit_scores(model_name, copy_scores, show_legend=False, title= f'Copy Score Across Checkpoints ({model_name})', disable_title=True)
    except:
        print(f"Error plotting copy scores for {model_name}--likely due to missing data")


def plot_total_cspa_activity(model_shortname: str, checkpoints: Dict[int, np.ndarray]):
    print(f"Subfolder: {model_shortname}")

    cspa_sums = dict()
    checkpoint_keys = list(checkpoints.keys())
    checkpoint_keys.sort()

    for ckpt_key in checkpoint_keys:
        cspa_sums[ckpt_key] = checkpoints[ckpt_key].sum() / 144

    # Convert cspa_sums to DataFrame
    df = pd.DataFrame(list(cspa_sums.items()), columns=['Checkpoint', 'CSPA Score'])

    # Plot using DataFrame
    fig = px.line(df, x='Checkpoint', y='CSPA Score', title=f'CSPA Score Over Checkpoints ({model_shortname})')

    # Set y-axis range and format as percentage
    fig.update_layout(
        yaxis=dict(range=[0, 0.05], tickformat=".1%"),
        title=f'CSPA Score Over Checkpoints ({model_shortname})'
    )

    fig.show()


def display_cspa_grids(model_shortname, checkpoint_schedule):
    checkpoint_dict = torch.load(f'results/cspa/{model_shortname}/all_checkpoints.pt')

    for checkpoint in checkpoint_schedule:
        if checkpoint in checkpoint_dict:
            head_results = checkpoint_dict[checkpoint]
            print(f"Checkpoint {checkpoint}")

            imshow_p(
                head_results * 100,
                title="CSPA Performance Recovered",
                labels={"x": "Head", "y": "Layer", "color": "Performance Recovered"},
                #coloraxis=dict(colorbar_ticksuffix = "%"),
                border=True,
                width=600,
                margin={"r": 100, "l": 100},
                # set max and min for coloraxis
                coloraxis=dict(colorbar_ticksuffix = "%", cmin=-100, cmax=100)
            )


def format_cspa_data_for_plots(checkpoint_dict):
    # Transform the dictionary
    head_values_dict = {}
    for checkpoint, tensor in checkpoint_dict.items():
        for layer_idx, layer in enumerate(tensor):
            for head_idx, value in enumerate(layer):
                head_key = f"L{layer_idx}H{head_idx}"
                if head_key not in head_values_dict:
                    head_values_dict[head_key] = {}
                head_values_dict[head_key][checkpoint] = value

    # Calculate the sum of values for each head across all checkpoints
    head_sums = {head_key: sum(values.values()) for head_key, values in head_values_dict.items()}

    # Sort heads by their sums and select the top 10
    top_heads = sorted(head_sums, key=head_sums.get, reverse=True)[:5]

    # Prepare data for plotting (only for top 10 heads)
    plot_data = []
    for head_key in top_heads:
        sorted_checkpoints = sorted(head_values_dict[head_key].keys())
        for checkpoint in sorted_checkpoints:
            plot_data.append({'Head': head_key, 'Checkpoint': checkpoint, 'CSPA Score': head_values_dict[head_key][checkpoint]})

    # Plot using Plotly Express
    df = pd.DataFrame(plot_data)

    return df


