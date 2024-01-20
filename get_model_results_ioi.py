import os
import pickle
import torch
import argparse
from collections import namedtuple
import json

import utils.circuit_utils as cu
from utils.model_utils import load_model, clear_gpu_memory
from utils.data_utils import generate_data_and_caches
from utils.metrics import _logits_to_mean_logit_diff, _logits_to_mean_accuracy, _logits_to_rank_0_rate, ave_logit_diff
from torchtyping import TensorType as TT

# Set up argument parser
parser = argparse.ArgumentParser(description='Run model with specified settings')
parser.add_argument('model_name', type=str, help='Name of the model to load')
parser.add_argument('circuit_file', type=str, help='Filename for the circuit dictionary')

# Parse arguments
args = parser.parse_args()

# Use the parsed model name and circuit file
model_name = args.model_name
circuit_file = args.circuit_file

# Settings
if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"
#
#DO_SLOW_RUNS = True
#
model_full_name = f"EleutherAI/{model_name}"
#
cache_dir = "model_cache"
##cache_dir = "/media/curttigges/project-files/projects/circuits"
#
## load model
model = load_model(
    model_full_name,model_full_name, "step143000", cache_dir=cache_dir
)

CircuitComponent = namedtuple(
    "CircuitComponent", ["heads", "position", "receiver_type"]
)

# Load the circuit dictionary from the specified file
circuit_root = "results/circuits/"
with open(circuit_root + circuit_file, 'rb') as f:
    circuit = pickle.load(f)

# set up data
N = 70
ioi_dataset, abc_dataset, ioi_cache, abc_cache, ioi_metric_noising = generate_data_and_caches(model, N, verbose=True)

# get baselines
#clean_logits, clean_cache = model.run_with_cache(ioi_dataset.toks)
#corrupted_logits, corrupted_cache = model.run_with_cache(abc_dataset.toks)
#
#clean_logit_diff = _logits_to_mean_logit_diff(clean_logits, ioi_dataset)
#print(f"Clean logit diff: {clean_logit_diff:.4f}")
#
#corrupted_logit_diff = _logits_to_mean_logit_diff(corrupted_logits, ioi_dataset)
#print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")
#
#clean_logit_accuracy = _logits_to_mean_accuracy(clean_logits, ioi_dataset).item()
#print(f"Clean logit accuracy: {clean_logit_accuracy:.4f}")
#
#corrupted_logit_accuracy = _logits_to_mean_accuracy(corrupted_logits, ioi_dataset).item()
#print(f"Corrupted logit accuracy: {corrupted_logit_accuracy:.4f}")
#
#clean_logit_rank_0_rate = _logits_to_rank_0_rate(clean_logits, ioi_dataset)
#print(f"Clean logit rank 0 rate: {clean_logit_rank_0_rate:.4f}")
#
#corrupted_logit_rank_0_rate = _logits_to_rank_0_rate(corrupted_logits, ioi_dataset)
#print(f"Corrupted logit rank 0 rate: {corrupted_logit_rank_0_rate:.4f}")
#
#CLEAN_BASELINE = clean_logit_diff
#CORRUPTED_BASELINE = corrupted_logit_diff


clear_gpu_memory(model)

# get values over time
# ckpts = [round((2**i) / 1000) * 1000 if 2**i > 1000 else 2**i for i in range(18)]
ckpts = [142000, 143000]

with torch.no_grad():
    clean_logits = model(ioi_dataset.toks)
    corrupt_logits = model(abc_dataset.toks)
    clean_logit_diff = ave_logit_diff(clean_logits, 
    io_tokenIDs = ioi_dataset.io_tokenIDs, 
    s_tokenIDs = ioi_dataset.s_tokenIDs,
    word_idx = ioi_dataset.word_idx['end']).item()

    corrupt_logit_diff = ave_logit_diff(corrupt_logits, 
    io_tokenIDs = abc_dataset.io_tokenIDs, 
    s_tokenIDs = abc_dataset.s_tokenIDs,
    word_idx = abc_dataset.word_idx['end']).item()


threshold = 0.02

nodes, edges, logits = cu.get_acdcpp_circuits(
    model_full_name,
    clean_logit_diff = clean_logit_diff, 
    corrupt_logit_diff = -corrupt_logit_diff,
    cache_dir = cache_dir,
    ckpts = ckpts,
    clean_data=ioi_dataset,
    corrupted_data=abc_dataset,
    threshold = threshold,
    batch_size = 9,
)

graph = {}
for k,v in edges.items():
    if v.in_graph:
        graph[k] = float(v.score)

with open(f"results/{model_name}-acdcpp-{threshold}.json", 'w') as f:
    json.dump(graph, f)

#a = torch.stack((torch.tensor(ioi_dataset.io_tokenIDs), torch.tensor(ioi_dataset.s_tokenIDs)), dim = -1)

'''
results_dict = cu.get_chronological_circuit_data(
    model_full_name,
    cache_dir = cache_dir,
    ckpts = ckpts,
    circuit=circuit,
    clean_tokens=ioi_dataset.toks,
    corrupted_tokens=abc_dataset.toks,
    answer_token_indices = a.cuda()
)

# save results
os.makedirs(f"results/{model_name}-no-dropout", exist_ok=True)
torch.save(
    results_dict["logit_diffs"], f"results/{model_name}-no-dropout/overall_perf.pt"
)
torch.save(
    results_dict["clean_baselines"],
    f"results/{model_name}-no-dropout/clean_baselines.pt",
)
torch.save(
    results_dict["corrupted_baselines"],
    f"results/{model_name}-no-dropout/corrupted_baselines.pt",
)
torch.save(
    results_dict["attn_head_vals"], f"results/{model_name}-no-dropout/attn_head_perf.pt"
)
torch.save(
    results_dict["value_patch_vals"], f"results/{model_name}-no-dropout/value_perf.pt"
)
with open(f"results/{model_name}-no-dropout/circuit_vals.pkl", "wb") as f:
    pickle.dump(results_dict["circuit_vals"], f)
with open(f"results/{model_name}-no-dropout/knockout_drops.pkl", "wb") as f:
    pickle.dump(results_dict["knockout_drops"], f)

'''