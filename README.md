# Circuits Over Time Experimentation Suite

This codebase contains a collection of tools for performing experiments on LLMs (specifically, Pythia) to characterize the nature and evolution of circuits over the course of training. Currently, the primary circuit examined is the Indirect Object Identification circuit, though we intend to add others.

## Structure

### get_model_perf_only.py
This script should be run with a config specified, e.g.: `python get_model_perf_only.py -c ./configs/pythia-160m.yml`. For the model specified in the config, it will collect performance metrics for (currently) an IOI dataset, including accuracy, logit difference, and rank 0 rate (for the correct answer). It will do this for checkpoints across training according to the schedule name specified in the config file. The results will be saved to the `results` folder in the appropriate subfolder.

### get_model_results.py
The script specified here will collect performance metrics and also conduct path patching to look at the circuit over time. Currently under heavy revision.

### ./circuit_sketches
This folder contains a collection of notebooks documenting experiments and results that were conducted in order to characterize the IOI circuit, mostly in models that have already completed training. This was done to establish a baseline for each model, as well as to confirm that the circuit is similar to that implemented in GPT-2 and that it uses similar subcomponents across model sizes and random seeds.

### ./configs
Contains configs for each model, specifying checkpoint schedule, model, and the directory where the models should be saved (can get very storage-intensive very fast). Note the following:
- `model_name` is the HuggingFace name of the model, and will be used to download it.
- `model_tl_name` is the name of the architecture that will be instantiated in TransformerLens, which is one of our main tools for analysis. The model downloaded from HuggingFace will be loaded into a TransformerLens HookedTransformer of this class.
- Current available checkpoint schedules are:
    - `linear`: Every checkpoint will be downloaded and checked.
    - `exponential`: The checkpoint schedule will follow an exponential progression (1, 2, 4, 8, 16, etc.).
    - `exponential_plus_detail`: The schedule starts exponential, but at step 1000 it starts evaluating every 1,000th checkpoint, and at step 15,000 it starts evaluating every 5,000th checkpoint, and finally at step 7,0000 it starts evaluating every 10,000th checkpoint.
