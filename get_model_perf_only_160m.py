import os
from collections import namedtuple

import torch
import pickle

from torchtyping import TensorType as TT

from model_utils import load_model, clear_gpu_memory
import circuit_utils as cu

# Settings
if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"

torch.set_grad_enabled(False)
DO_SLOW_RUNS = True

# define the model names
model_name = "pythia-160m"
model_tl_name = "pythia-125m"

model_full_name = f"EleutherAI/{model_name}"
model_tl_full_name = f"EleutherAI/{model_tl_name}"

# cache_dir = "/fsx/home-curt/saved_models"
cache_dir = "/media/curttigges/project-files/projects/circuits"

# load model
model = load_model(
    model_full_name, model_tl_full_name, "step143000", cache_dir=cache_dir
)

# set up data
prompts = [
    "When John and Mary went to the shops, John gave the bag to",
    "When John and Mary went to the shops, Mary gave the bag to",
    "When Tom and James went to the park, James gave the ball to",
    "When Tom and James went to the park, Tom gave the ball to",
    "When Dan and Sid went to the shops, Sid gave an apple to",
    "When Dan and Sid went to the shops, Dan gave an apple to",
    "After Martin and Amy went to the park, Amy gave a drink to",
    "After Martin and Amy went to the park, Martin gave a drink to",
]

answers = [
    (" Mary", " John"),
    (" John", " Mary"),
    (" Tom", " James"),
    (" James", " Tom"),
    (" Dan", " Sid"),
    (" Sid", " Dan"),
    (" Martin", " Amy"),
    (" Amy", " Martin"),
]


clean_tokens, corrupted_tokens, answer_token_indices = cu.set_up_data(
    model, prompts, answers
)

# get baselines
clean_logits, clean_cache = model.run_with_cache(clean_tokens)
corrupted_logits, corrupted_cache = model.run_with_cache(corrupted_tokens)

clean_logit_diff = cu.get_logit_diff(clean_logits, answer_token_indices).item()
print(f"Clean logit diff: {clean_logit_diff:.4f}")

corrupted_logit_diff = cu.get_logit_diff(corrupted_logits, answer_token_indices).item()
print(f"Corrupted logit diff: {corrupted_logit_diff:.4f}")

CLEAN_BASELINE = clean_logit_diff
CORRUPTED_BASELINE = corrupted_logit_diff

clean_baseline_ioi = cu.ioi_metric(
    clean_logits, CLEAN_BASELINE, CORRUPTED_BASELINE, answer_token_indices
)
corrupted_baseline_ioi = cu.ioi_metric(
    corrupted_logits, CLEAN_BASELINE, CORRUPTED_BASELINE, answer_token_indices
)

print(
    f"Clean Baseline is 1: {cu.ioi_metric(clean_logits, CLEAN_BASELINE, CORRUPTED_BASELINE, answer_token_indices).item():.4f}"
)
print(
    f"Corrupted Baseline is 0: {cu.ioi_metric(corrupted_logits, CLEAN_BASELINE, CORRUPTED_BASELINE, answer_token_indices).item():.4f}"
)

clear_gpu_memory(model)

# get values over time
ckpts = [round((2**i) / 1000) * 1000 if 2**i > 1000 else 2**i for i in range(18)]
# ckpts = [142000, 143000]
results_dict = cu.get_chronological_circuit_performance(
    model_full_name,
    model_tl_full_name,
    cache_dir,
    ckpts,
    clean_tokens=clean_tokens,
    corrupted_tokens=corrupted_tokens,
    answer_token_indices=answer_token_indices,
)

# save results
os.makedirs(f"results/{model_name}-no-dropout", exist_ok=True)
torch.save(
    results_dict["logit_diffs"], f"results/{model_name}-no-dropout/logit_diffs.pt"
)
# torch.save(
#     results_dict["clean_baselines"],
#     f"results/{model_name}-no-dropout/clean_baselines.pt",
# )
# torch.save(
#     results_dict["corrupted_baselines"],
#     f"results/{model_name}-no-dropout/corrupted_baselines.pt",
# )
