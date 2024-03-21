from imports import *
from path_patching import act_patch
from GOOD_helpers import is_notebook, shuffle_owt_tokens_by_batch, return_item, get_correct_logit_score, collect_direct_effect, create_layered_scatter, replace_output_hook, prepare_dataset
# %% Constants
from GOOD_self_repair_graph_generator import write_result
in_notebook_mode = is_notebook()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FOLDER_TO_WRITE_GRAPHS_TO = "figures/new_self_repair_graphs/"
FOLDER_TO_STORE_PICKLES = "pickle_storage/new_graph_pickle/"
import sys
sys.path.append("../..")
from utils.model_utils import load_model

# %% Import the Model
from transformers import LlamaForCausalLM, LlamaTokenizer
#from constants import LLAMA_MODEL_PATH # change LLAMA_MODEL_PATH to the path of your llama model weights
import sys
sys.path.append("..")
from utils.backup_analysis import (
    load_model,
)
checkpoints = [143000]
torch.set_grad_enabled(False)
for k in checkpoints:
    model = HookedTransformer.from_pretrained('pythia-160m')
    #model = load_model('pythia-160m', None, k, "model_cache", 'cuda')
    #model_name = 'pythia-160m '
    model.set_use_attn_result(False)

    FOLDER_TO_WRITE_GRAPHS_TO = f'self_repair/figures/new_self_repair_graph/pythia-160m/'
    FOLDER_TO_STORE_PICKLES= f"self_repair_pickle_storage/new_graph_pickle/pythia-160m/"
    os.makedirs(FOLDER_TO_WRITE_GRAPHS_TO, exist_ok = True) 
    os.makedirs(FOLDER_TO_STORE_PICKLES, exist_ok = True)
    ABLATION_TYPE = 'sample'
    write_result(model, ABLATION_TYPE, 'pythia-160m', FOLDER_TO_WRITE_GRAPHS_TO, FOLDER_TO_STORE_PICKLES)