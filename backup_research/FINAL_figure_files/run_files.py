from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, create_layered_scatter, replace_output_hook, prepare_dataset
# %% Constants
from GOOD_self_repair_graph_generator import write_result, write_result_dicts
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#FOLDER_TO_WRITE_GRAPHS_TO = "figures/new_self_repair_graphs/"
#FOLDER_TO_STORE_PICKLES = "pickle_storage/new_graph_pickle/"
import sys
sys.path.append("../..")
#from utils.model_utils import load_model

# %% Import the Model
from transformers import LlamaForCausalLM, LlamaTokenizer
#from constants import LLAMA_MODEL_PATH # change LLAMA_MODEL_PATH to the path of your llama model weights
import sys
sys.path.append("..")
from utils.backup_analysis import (
    load_model,
)
checkpoints = [256, 512, 1000, 2000, 3000, 5000, 10000, 30000, 60000, 90000, 143000]
torch.set_grad_enabled(False)

BASE_MODEL = "pythia-160m"
VARIANT = "EleutherAI/pythia-160m-hiddendropout"
MODEL_SHORTNAME = BASE_MODEL if not VARIANT else VARIANT[11:]
CACHE = "model_cache"

for k in checkpoints:
    #model = HookedTransformer.from_pretrained('pythia-160m')
    model = load_model(BASE_MODEL, VARIANT, k, CACHE, 'cuda')
    model.set_use_attn_result(False)

    FOLDER_TO_WRITE_GRAPHS_TO = f'sr_over_time/plots/{MODEL_SHORTNAME}/'
    FOLDER_TO_STORE_DICTS= f"sr_over_time/data/{MODEL_SHORTNAME}/"
    os.makedirs(FOLDER_TO_WRITE_GRAPHS_TO, exist_ok = True) 
    os.makedirs(FOLDER_TO_STORE_DICTS, exist_ok = True)
    ABLATION_TYPE = 'sample'
    write_result_dicts(model, ABLATION_TYPE, MODEL_SHORTNAME, k, FOLDER_TO_WRITE_GRAPHS_TO, FOLDER_TO_STORE_DICTS)