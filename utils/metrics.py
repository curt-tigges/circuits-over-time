import os
import torch
from torch import Tensor
from jaxtyping import Float


from path_patching_cm.ioi_dataset import IOIDataset


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

def get_logit_diff(logits, answer_token_indices, per_prompt=False):
    """Gets the difference between the logits of the provided tokens (e.g., the correct and incorrect tokens in IOI)

    Args:
        logits (torch.Tensor): Logits to use.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    if len(logits.shape) == 3:
        # Get final logits only
        logits = logits[:, -1, :]
    correct_logits = logits.gather(1, answer_token_indices[:, 0].unsqueeze(1))
    incorrect_logits = logits.gather(1, answer_token_indices[:, 1].unsqueeze(1))
    if per_prompt:
        return (correct_logits - incorrect_logits).squeeze()
    else:
        return (correct_logits - incorrect_logits).mean()


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