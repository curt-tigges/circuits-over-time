from sklearn.linear_model import LinearRegression
import numpy as np
import os
import torch
import plotly.express as px

time_step = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000, 65000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 143000]


dic = torch.load("/users/qyu10/lab/circuits-over-time/compiled_metric_dict.pt")

logit_diff_ioi = dic['pythia-70m']['ioi']['mrr']
logit_diff_greater_tjam = dic['pythia-70m']['greater_than']['mrr']
logit_diff_sentiment_cont = dic['pythia-70m']['sentiment_cont']['mrr']
logit_diff_sentiment_class = dic['pythia-70m']['sentiment_class']['mrr']

def moving_average(input_tensor, window_size):
    """Calculate the moving average with a given window size."""
    # Prepend a zero to the input_tensor
    padded_input = torch.cat([torch.zeros(1, device=input_tensor.device), input_tensor])
    cumsum_vec = torch.cumsum(padded_input, dim=0)
    moving_avg = (cumsum_vec[window_size:] - cumsum_vec[:-window_size]) / window_size
    print(moving_avg)
    print(moving_avg.shape)
    return moving_avg


def get_start(input_tensor, START_THRESHOLD, END_THRESHOLD, window_size=5):
    """
    Adjust the function to use the rolling average of the last n "differences" 
    to calculate the end_index.

    Args:
        input_tensor (torch.Tensor): Input tensor.
        START_THRESHOLD (float): Threshold to detect the start.
        END_THRESHOLD (float): Threshold to detect the end.
        window_size (int): Window size for the rolling average calculation.

    Returns:
        tuple: A tuple containing the start and end indices.
    """
    differences = input_tensor[1:] - input_tensor[:-1]
    print(differences)

    # Find start index
    start_indices = torch.nonzero(differences > START_THRESHOLD).view(-1)
    first_index = start_indices[0].item() if len(start_indices) > 0 else None

    if first_index is not None and first_index + window_size <= len(differences):
        rolled_differences = moving_average(differences[first_index:], window_size)
        # Adjust indices to match the original differences tensor
        adjusted_indices = torch.nonzero(rolled_differences < END_THRESHOLD).view(-1) + first_index + window_size - 1
        end_index = adjusted_indices[0].item() if len(adjusted_indices) > 0 else len(input_tensor)
    else:
        end_index = len(input_tensor) - 1  # Adjust to get the actual last index

    return first_index, end_index

s, e = get_start(logit_diff_ioi, 0.01, 0.01)
print(time_step[s], time_step[e])
'''
def get_start(input_tensor, START_THRESHOLD, END_THRESHOLD):
    differences = torch.abs(input_tensor[1:] - input_tensor[:-1])


    # Find indices where the difference is greater than 0.1
    indices = torch.nonzero(differences > START_THRESHOLD)

    # Get the first index where the difference is greater than 0.1 and adjust by 1
    # to account for the shift due to difference calculation
    first_index = indices[0].item() if len(indices) > 0 else None

    new_tensor = differences[first_index:]

    # Find indices where the difference is greater than 0.1
    indices = torch.nonzero(new_tensor < END_THRESHOLD)

    # Get the first index where the difference is greater than 0.1 and adjust by 1
    # to account for the shift due to difference calculation
    end_index = indices[0].item() + first_index if len(indices) > 0 else len(input_tensor)


    return first_index, end_index

def line_with_gradient(tensor, intercept, coefficient, x_start, x_end, renderer=None, width=1200, height=500, **kwargs):
    # Convert tensor to numpy for plotting
    y_values = utils.to_numpy(tensor)
    
    # Create the initial line plot
    fig = px.line(y=y_values, **kwargs)
    
    # Calculate y values for the superimposed line based on the given intercept and coefficient
    x_values = [x_start, x_end]
    y_line = [coefficient * x + intercept for x in x_values]

    print(x_values)
    print(y_line)
    
    # Add the superimposed line to the figure
    fig.add_trace(go.Scatter(x=x_values, y=y_line, mode='lines', name='Superimposed Line'))
    
    # Update layout with specified width and height
    fig.update_layout(
        autosize=False,
        width=width,
        height=height
    )
    
    # Show the figure with the optional renderer
    fig.show(renderer=renderer)

def line_with_gradient(tensor, intercept, coefficient, x_start, x_end, renderer=None, width=1200, height=500, **kwargs):
    # Convert tensor to numpy for plotting
    y_values = utils.to_numpy(tensor)
    
    # Create the initial line plot
    fig = px.line(y=y_values, **kwargs)
    
    # Calculate y values for the superimposed line based on the given intercept and coefficient
    x_values = [x_start, x_end]
    y_line = [coefficient * x + intercept for x in x_values]

    print(x_values)
    print(y_line)
    
    # Add the superimposed line to the figure
    fig.add_trace(go.Scatter(x=x_values, y=y_line, mode='lines', name='Superimposed Line'))
    
    # Update layout with specified width and height
    fig.update_layout(
        autosize=False,
        width=width,
        height=height
    )
    
    # Show the figure with the optional renderer
    fig.show(renderer=renderer)

start_index, end_index = get_start(logit_diff, 0.8, 0.05)
print(time_step[start_index], time_step[end_index])
# Creating a linear regression model
model = LinearRegression()

# Training the model
model.fit(np.array(time_step[start_index:end_index+1]).reshape(-1, 1), logit_diff[start_index:end_index+1])

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)
'''
