import os
import torch
import torch.nn.functional as F
from torch import Tensor
from jaxtyping import Float

from transformers import PreTrainedTokenizer

from path_patching_cm.ioi_dataset import IOIDataset
from data.greater_than_dataset import get_year_indices


if torch.cuda.is_available():
    device = int(os.environ.get("LOCAL_RANK", 0))
else:
    device = "cpu"


def _logits_to_mean_logit_diff(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset, per_prompt=False):
    '''
    Returns logit difference between the correct and incorrect answer.

    If per_prompt=True, return the array of differences rather than the average.
    '''

    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    return answer_logit_diff if per_prompt else answer_logit_diff.mean()


def _logits_to_mean_accuracy(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset):
    '''
    Returns accuracy of the model on the IOI dataset.
    '''
    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object / subject tokens respectively
    io_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.io_tokenIDs]
    s_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"], ioi_dataset.s_tokenIDs]
    # Find logit difference
    answer_logit_diff = io_logits - s_logits
    # Find accuracy
    return (answer_logit_diff > 0).float().mean()


def _logits_to_rank_0_rate(logits: Float[Tensor, "batch seq d_vocab"], ioi_dataset: IOIDataset):
    '''
    Returns rate of the model ranking the correct answer as the most probable.
    '''
    # Only the final logits are relevant for the answer
    # Get the logits corresponding to the indirect object tokens
    final_logits: Float[Tensor, "batch"] = logits[range(logits.size(0)), ioi_dataset.word_idx["end"]]
    
    return (final_logits.argmax(dim=-1) == torch.tensor(ioi_dataset.io_tokenIDs).to(device)).float().mean()


def _ioi_metric_noising(
        logits: Float[Tensor, "batch seq d_vocab"],
        clean_logit_diff: float,
        corrupted_logit_diff: float,
        ioi_dataset: IOIDataset,
    ) -> float:
        '''
        We calibrate this so that the value is 0 when performance isn't harmed (i.e. same as IOI dataset),
        and -1 when performance has been destroyed (i.e. is same as ABC dataset).
        '''
        patched_logit_diff = _logits_to_mean_logit_diff(logits, ioi_dataset)
        return ((patched_logit_diff - clean_logit_diff) / (clean_logit_diff - corrupted_logit_diff)).item()


def get_positional_logits(logits, positions=None):
    """Gets the logits at the provided positions. If no positions are provided, the final logits are returned.

    Args:
        logits (torch.Tensor): Logits to use.
        positions (torch.Tensor): Positions to get logits at.

    Returns:
        torch.Tensor: Logits at the provided positions.
    """
    if positions is None:
        return logits[:, -1, :]
    return logits[range(logits.size(0)), positions]


def compute_logit_diff(
        logits: Float[Tensor, "batch seq d_vocab"], 
        answer_token_indices: Float[Tensor, "batch num_answers"],
        positions: Float[Tensor, "batch"] = None,
        per_prompt=False
)-> Float[Tensor, "batch num_answers"]:
    """Computes the difference between a correct and incorrect logit (or mean of a group of logits) for each item in the batch.

    Takes the full logits, and the indices of the tokens to compare. These indices can be of multiple types, either specifying
    a correct and incorrect token index (in which cases the tensor should be of shape (batch_size, 2)), or a group of correct 
    and incorrect token indices (in which case the tensor should be of shape (batch_size, 2, num_members)).

    Args:
        logits (torch.Tensor): Logits to use.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.
        positions (torch.Tensor): Positions to get logits at. Should be one position per batch item.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    logits = get_positional_logits(logits, positions)
    if len(answer_token_indices.shape) == 2:
        answer_token_indices = answer_token_indices.unsqueeze(-1)
    correct_logits = logits[range(logits.size(0)), answer_token_indices[:, 0]]
    incorrect_logits = logits[range(logits.size(0)), answer_token_indices[:, 1]]
    logit_diff = correct_logits - incorrect_logits
    return logit_diff if per_prompt else logit_diff.mean()


def compute_prob_diff(
        logits: Float[Tensor, "batch seq d_vocab"], 
        answer_token_indices: Float[Tensor, "batch num_answers"],
        positions: Float[Tensor, "batch"] = None,
        per_prompt=False
)-> Float[Tensor, "batch num_answers"]:
    """Computes the difference between a correct and incorrect probability (or mean of a group of probabilities) for each item in the batch.

    Takes the full logits, and the indices of the tokens to compare. These indices can be of multiple types, either specifying
    a correct and incorrect token index (in which cases the tensor should be of shape (batch_size, 2)), or a group of correct
    and incorrect token indices (in which case the tensor should be of shape (batch_size, 2, num_members)).

    Args:
        logits (torch.Tensor): Logits to use.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.
        positions (torch.Tensor): Positions to get logits at. Should be one position per batch item.

    Returns:
        torch.Tensor: Difference between the probabilities of the provided tokens.
    """
    logits = get_positional_logits(logits, positions)
    if len(answer_token_indices.shape) == 2:
        answer_token_indices = answer_token_indices.unsqueeze(-1)
    correct_probs = torch.softmax(logits, dim=-1)[range(logits.size(0)), answer_token_indices[:, 0]]
    incorrect_probs = torch.softmax(logits, dim=-1)[range(logits.size(0)), answer_token_indices[:, 1]]
    prob_diff = correct_probs - incorrect_probs
    return prob_diff if per_prompt else prob_diff.mean()

    



def ioi_metric(logits, clean_baseline, corrupted_baseline, answer_token_indices):
    """Computes the IOI metric for a given set of logits, baselines, and answer token indices. Metric is relative to the
    provided baselines.

    Args:
        logits (torch.Tensor): Logits to use.
        clean_baseline (float): Baseline for the clean model.
        corrupted_baseline (float): Baseline for the corrupted model.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: IOI metric.
    """
    return (get_logit_diff(logits, answer_token_indices) - corrupted_baseline) / (
        clean_baseline - corrupted_baseline
    )


def get_prob_diff(tokenizer: PreTrainedTokenizer):
    year_indices = get_year_indices(tokenizer) 
    def prob_diff(logits, per_prompt, years):
        # Prob diff (negative, since it's a loss)
        probs = torch.softmax(logits[:, -1], dim=-1)[:, year_indices]
        diffs = []
        for prob, year in zip(probs, years):
            diffs.append(prob[year + 1 :].sum() - prob[: year + 1].sum())
        return -torch.stack(diffs).mean().to('cuda')
    return prob_diff