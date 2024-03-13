import os
import time
import torch
#import circuitsvis as cv
import pickle
import warnings
from tdqm.auto import tdqm
from transformers import AutoModelForCausalLM

from transformer_lens import HookedTransformer
from IPython.display import display, clear_output, HTML
from utils.visualization import imshow_p

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

#from transformer_lens.cautils.notebook import *
torch.set_grad_enabled(False)

from utils.cspa_functions import (
    get_cspa_results_batched,
    get_result_mean,
    get_performance_recovered
)
from utils.cspa_extra_utils import (
    process_webtext,
)


def load_model_for_cspa(
        base_model: str = "pythia-160m", 
        variant: str = None, 
        checkpoint: int = 143000, 
        cache: str = "model_cache", 
        device: torch.device = torch.device("cuda")
    ) -> HookedTransformer:
    """
    Load a transformer model from a pretrained base model or variant.

    Args:
        BASE_MODEL (str): The name of the base model.
        VARIANT (str): The name of the model variant (if applicable).
        CHECKPOINT (int): The checkpoint value for the model.
        CACHE (str): The directory to cache the model.
        device (torch.device): The device to load the model onto.

    Returns:
        HookedTransformer: The loaded transformer model.
    """
    if not variant:
        model = HookedTransformer.from_pretrained(
            base_model,
            checkpoint_value=checkpoint,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            #refactor_factored_attn_matrices=False,
            #dtype=torch.bfloat16,
            **{"cache_dir": cache},
        )
    else:
        revision = f"step{checkpoint}"
        source_model = AutoModelForCausalLM.from_pretrained(
           variant, revision=revision, cache_dir=cache
        ).to(device) #.to(torch.bfloat16)
        print(f"Loaded model {variant} at {revision}; now loading into HookedTransformer")
        model = HookedTransformer.from_pretrained(
            base_model,
            hf_model=source_model,
            center_unembed=True,
            center_writing_weights=True,
            fold_ln=True,
            #refactor_factored_attn_matrices=False,
            #dtype=torch.bfloat16,
            **{"cache_dir": cache},
        )

    model.cfg.use_split_qkv_input = False
    model.cfg.use_attn_result = True
    #model.cfg.use_hook_mlp_in = True
    return model


def prepare_data(model, use_semanticity=True, batch_size=500, seq_len=1000, seed=6, verbose=False, return_indices=True):

    DATA_TOKS, DATA_STR_TOKS_PARSED, indices = process_webtext(
        seed=seed, 
        batch_size=batch_size, 
        seq_len=seq_len, 
        model=model, 
        verbose=verbose, 
        return_indices=return_indices)

    if use_semanticity:
        cspa_semantic_dict = pickle.load(open("cspa/cspa_semantic_dict_full.pkl", "rb"))

    else:
        warnings.warn("Not using semanticity unlike old notebook versions!")
        cspa_semantic_dict = {}

    return DATA_TOKS, DATA_STR_TOKS_PARSED, cspa_semantic_dict, indices


def get_cspa_per_checkpoint(base_model, variant, cache, device, checkpoints, start_layer, overwrite=False):
    
    model_shortname = base_model if not variant else variant[11:]
    
    filename = f'results/cspa/{model_shortname}/all_checkpoints.pt'
    directory = os.path.dirname(filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Load the dictionary from disk if it exists, otherwise create a new one
    if os.path.exists(filename):
        checkpoint_dict = torch.load(filename)
    else:
        checkpoint_dict = {} 

    for checkpoint in tdqm(checkpoints):
        if checkpoint in checkpoint_dict and not overwrite:
            print(f"Skipping checkpoint {checkpoint} as it already exists in the dictionary.")
            continue
        else:
            print(f"Processing checkpoint {checkpoint}...")

        # Your existing code
        model = load_model_for_cspa(base_model, variant, checkpoint, cache, device)
        head_results = get_cspa_for_model(model, start_layer=start_layer)

        # Save results to the dictionary and resave to disk
        checkpoint_dict[checkpoint] = head_results
        torch.save(checkpoint_dict, filename)
        clear_output()
        


def get_cspa_for_model(model, start_layer=2):
    DATA_TOKS, DATA_STR_TOKS_PARSED, cspa_semantic_dict, indices = prepare_data(model)
    head_results = torch.zeros((12, 12))

    current_batch_size = 17 # Smaller values so we can check more checkpoints in a reasonable amount of time
    current_seq_len = 61

    for layer in tdqm(range(start_layer, 12)):
        for head in range(12):
            start = time.time()
            result_mean = get_result_mean([(layer, head)], DATA_TOKS[:100, :], model, verbose=True)
            cspa_results_qk_ov = get_cspa_results_batched(
                model=model,
                toks=DATA_TOKS[:current_batch_size, :current_seq_len],
                max_batch_size=1,  # 50,
                negative_head=(layer, head),
                interventions=["ov", "qk"],
                only_keep_negative_components=True,
                K_unembeddings=0.05,  # most interesting in range 3-8 (out of 80)
                K_semantic=1,  # either 1 or up to 8 to capture all sem similar
                semantic_dict=cspa_semantic_dict,
                result_mean=result_mean,
                use_cuda=True,
                verbose=True,
                compute_s_sstar_dict=False,
                computation_device="cpu",  # device
            )
            head_results[layer, head] = get_performance_recovered(cspa_results_qk_ov)

            print(f"Layer {layer}, head {head} done. Performance: {head_results[layer, head]}")

    return head_results

    

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