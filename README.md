# Circuits Over Time Experimentation Suite

This codebase contains a collection of tools for performing experiments on LLMs (specifically, Pythia) to characterize the nature and evolution of circuits over the course of training.

## Environment
To run this code, you will need a GPU with sufficient GPU RAM (ideally 50 GB or more), and to install the packages listed in `requirements.txt` and `environment.yml`. Install with the following code:
```conda env create -f environment.yml
conda activate your_environment_name
```
And then install the pip packages:
```
pip install -r requirements.txt

```

In addition, you will need to clone this repo https://github.com/hannamw/EAP-IG/tree/10035b88ceecf8bc7e444ba50449107d3e163069 to the `edge_attribution_patching` folder in the root of this code folder.

## Structure
### Key Scripts
Key scripts are all located in the root folder. They rely on settings that are specified in the `configs` folder.
- Circuit graphs are obtained via EAP-IG with `get_circuits_over_time.py`; this can be run with configs specifying different datasets and models.
- Attention head component scores can be obtained with `get_full_model_components_over_time.py`, `get_new_successor_head_scores_over_time.py`, and `get_model_cspa.py`.
- Algorithmic consistency is verified with the `get_ioi_consistency.py` file and variants.
- Behavior/performance can be collected with `get_model_task_performance.py`.

### ./circuit_sketches
This folder contains a collection of notebooks documenting experiments and results that were conducted in order to characterize the IOI circuit, mostly in models that have already completed training. This was done to establish a baseline for each model, as well as to confirm that the circuit is similar to that implemented in GPT-2 and that it uses similar subcomponents across model sizes and random seeds.

### ./plotting
This folder contains various scripts/notebooks for plotting results.

### ./utils
Contains all the key functions for running most of the experiments.

## Outside Code
With written permission from the authors, we include CSPA code produced by Arthur Comny and Callum McDougall in the utils. This code is used to obtain the CSPA score for each attention head.
