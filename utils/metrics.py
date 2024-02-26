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
    answer_token_indices = answer_token_indices.cuda()
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


def ave_logit_diff(
    logits: Float[Tensor, 'batch seq d_vocab'],
    word_idx,
    io_tokenIDs,
    s_tokenIDs,
    per_prompt: bool = False
):
    '''
        Return average logit difference between correct and incorrect answers
    '''
    # Get logits for indirect objects
    io_logits = logits[range(logits.size(0)), word_idx, io_tokenIDs]
    s_logits = logits[range(logits.size(0)), word_idx, s_tokenIDs]
    # Get logits for subject
    logit_diff = io_logits - s_logits
    return logit_diff if per_prompt else logit_diff.mean()

def ioi_metric(
    logits: Float[Tensor, "batch seq_len d_vocab"],
    word_idx,
    io_tokenIDs,
    s_tokenIDs,
    corrupt_logit_diff,
    clean_logit_diff,
 ):
    patched_logit_diff = ave_logit_diff(logits, word_idx, io_tokenIDs, s_tokenIDs)
    return (patched_logit_diff - corrupt_logit_diff) / (clean_logit_diff - corrupt_logit_diff)


class CircuitMetric:
    """ General wrapper for metric functions

        Functions for which this is to be used as a wrapper should accept logits and a per_prompt argument.
        Other arguments can be included through args or kwargs. If normalization_fn is not None, it should
        accept logits, metric_fn, and per_prompt as arguments, and any other arguments can be included through
        args or kwargs.

        Args:
            name (str): Name of the metric.
            metric_fn (function): Function to be wrapped.
            normalization_fn (function): Function to be used for normalization.

        Returns:
            function: Wrapped metric function.
    """

    def __init__(self, name, metric_fn, normalization_fn=None, eap=False):
            self.name = name
            self.metric_fn = metric_fn
            self.normalization_fn = normalization_fn
            self.eap = eap

    def __call__(self, logits, *args, loss=False, **kwargs):
        if self.eap:
            if self.name != 'kl_divergence':
                args = args[1:]

        multiplier = -1 if loss and self.name != 'kl_divergence' else 1
        if self.normalization_fn is not None:
            return self.normalization_fn(logits, self.metric_fn, *args, **kwargs)
        return multiplier * self.metric_fn(logits, *args, **kwargs)

def get_positional_logits(
        logits: Float[Tensor, "batch seq d_vocab"],
        positions: Float[Tensor, "batch"] = None
)-> Float[Tensor, "batch d_vocab"]:
    """Gets the logits at the provided positions. If no positions are provided, the final logits are returned.

    Args:
        logits (torch.Tensor): Logits to use.
        positions (torch.Tensor): Positions to get logits at. This should be a tensor of shape (batch_size,).

    Returns:
        torch.Tensor: Logits at the provided positions.
    """
    if positions is None:
        return logits[:, -1, :]

    return logits[range(logits.size(0)), positions, :]


def compute_logit_diff(
        logits: Float[Tensor, "batch seq d_vocab"], 
        answer_token_indices: Float[Tensor, "batch num_answers"],
        positions: Float[Tensor, "batch"] = None,
        flags_tensor: torch.Tensor = None,
        per_prompt=False,
        mode="simple"
)-> Float[Tensor, "batch num_answers"]:
    """Computes the difference between a correct and incorrect logit (or mean of a group of logits) for each item in the batch.

    Takes the full logits, and the indices of the tokens to compare. These indices can be of multiple types as follows:

    - Simple: The tensor should be of shape (batch_size, 2), where the first index in the third dimension is the correct token index,
        and the second index is the incorrect token index.

    - Pairs: In this mode, answer_token_indices is a 3D tensor of shape (batch, num_pairs, 2). For each pair, you'll need to compute 
             the difference between the logits at the two indices, then average these differences across each pair for every batch item.

    - Groups: Here, answer_token_indices is also a 3D tensor of shape (batch, num_tokens, 2). The third dimension indicates group membership 
              (correct or incorrect). The mean logits for each group are calculated and then subtracted from each other.
              

    Args:
        logits (torch.Tensor): Logits to use.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.
        positions (torch.Tensor): Positions to get logits at. Should be one position per batch item.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    #print(f"positions: {positions}")
    #print(f"flags_tensor: {flags_tensor}")
    logits = get_positional_logits(logits, positions)
    
    # Mode 1: Simple
    if mode == "simple":
        correct_logits = logits[torch.arange(logits.size(0)), answer_token_indices[:, 0]]
        incorrect_logits = logits[torch.arange(logits.size(0)), answer_token_indices[:, 1]]
        logit_diff = correct_logits - incorrect_logits

    # Mode 2: Pairs
    elif mode == "pairs":
        pair_diffs = logits[torch.arange(logits.size(0))[:, None], answer_token_indices[..., 0]] - \
                     logits[torch.arange(logits.size(0))[:, None], answer_token_indices[..., 1]]
        logit_diff = pair_diffs.mean(dim=1)

    # Mode 3: Groups
    elif mode == "groups":
        assert flags_tensor is not None
        logit_diff = torch.zeros(logits.size(0), device=logits.device)

        for i in range(logits.size(0)):
            selected_logits = logits[i, answer_token_indices[i]]

            # Calculate the logit difference using the correct/incorrect flags
            correct_logits = selected_logits[flags_tensor[i] == 1]
            incorrect_logits = selected_logits[flags_tensor[i] == -1]

            # Handle cases where there are no correct or incorrect logits
            if len(correct_logits) > 0:
                correct_mean = correct_logits.mean()
            else:
                correct_mean = 0

            if len(incorrect_logits) > 0:
                incorrect_mean = incorrect_logits.mean()
            else:
                incorrect_mean = 0

            logit_diff[i] = correct_mean - incorrect_mean

    else:
        raise ValueError("Invalid mode specified")

    return logit_diff.mean() if not per_prompt else logit_diff


def compute_probability_diff(
        logits: torch.Tensor, 
        answer_token_indices: torch.Tensor,
        positions: torch.Tensor = None,
        flags_tensor: torch.Tensor = None,
        per_prompt=False,
        mode="simple"
) -> torch.Tensor:
    """Computes the difference between probability of a correct and incorrect logit (or mean of a group of logits) for each item in the batch.

    Takes the full logits, and the indices of the tokens to compare. These indices can be of multiple types as follows:

    - Simple: The tensor should be of shape (batch_size, 2), where the first index in the third dimension is the correct token index,
        and the second index is the incorrect token index.

    - Pairs: In this mode, answer_token_indices is a 3D tensor of shape (batch, num_pairs, 2). For each pair, you'll need to compute 
             the difference between the probabilities at the two indices, then average these differences across each pair for every batch item.

    - Groups: Here, answer_token_indices is also a 3D tensor of shape (batch, num_tokens, 2). The third dimension indicates group membership 
              (correct or incorrect). The mean probabilities for each group are calculated and then subtracted from each other.
              

    Args:
        logits (torch.Tensor): Logits to use.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.
        positions (torch.Tensor): Positions to get logits at. Should be one position per batch item.

    Returns:
        torch.Tensor: Difference between the logits of the provided tokens.
    """
    logits = get_positional_logits(logits, positions)
    probabilities = torch.softmax(logits, dim=-1)  # Applying softmax to logits
    #print(f"probabilities={probabilities.shape}")

    # Mode 1: Simple
    if mode == "simple":
        correct_probs = probabilities[torch.arange(logits.size(0)), answer_token_indices[:, 0]]
        incorrect_probs = probabilities[torch.arange(logits.size(0)), answer_token_indices[:, 1]]
        prob_diff = correct_probs - incorrect_probs

    # Mode 2: Pairs
    elif mode == "pairs":
        pair_diffs = probabilities[torch.arange(logits.size(0))[:, None], answer_token_indices[..., 0]] - \
                     probabilities[torch.arange(logits.size(0))[:, None], answer_token_indices[..., 1]]
        prob_diff = pair_diffs.mean(dim=1)

    # Mode 3: Groups
    elif mode == "groups":
        # Initialize tensors to store the probability differences for each batch item
        assert flags_tensor is not None
        prob_diff = torch.zeros(logits.size(0), device=logits.device)

        for i in range(logits.size(0)):
            # Select the probabilities for the token IDs of this batch item
            selected_probs = probabilities[i, answer_token_indices[i]]

            # Calculate the probability difference using the correct/incorrect flags
            correct_probs = selected_probs[flags_tensor[i] == 1]
            incorrect_probs = selected_probs[flags_tensor[i] == -1]

            # Handle cases where there are no correct or incorrect tokens
            if len(correct_probs) > 0:
                correct_mean = correct_probs.mean()
            else:
                correct_mean = 0

            if len(incorrect_probs) > 0:
                incorrect_mean = incorrect_probs.mean()
            else:
                incorrect_mean = 0

            prob_diff[i] = correct_mean - incorrect_mean

    # Mode 4: Group Sum
    elif mode == "group_sum":
        assert flags_tensor is not None
        prob_diff = torch.zeros(logits.size(0), device=logits.device)

        for i in range(logits.size(0)):
            selected_probs = probabilities[i, answer_token_indices[i]]

            # Calculate the sum of probabilities using the correct/incorrect flags
            correct_sum = selected_probs[flags_tensor[i] == 1].sum()
            incorrect_sum = selected_probs[flags_tensor[i] == -1].sum()

            prob_diff[i] = correct_sum - incorrect_sum

    else:
        raise ValueError("Invalid mode specified")

    return prob_diff.mean() if not per_prompt else prob_diff


def compute_probability_mass(
        logits: torch.Tensor, 
        answer_token_indices: torch.Tensor,
        positions: torch.Tensor = None,
        flags_tensor: torch.Tensor = None,
        group="correct",
        mode="simple"
) -> torch.Tensor:
    logits = get_positional_logits(logits, positions)
    probabilities = torch.softmax(logits, dim=-1)

    # Determine the flag value based on the specified group
    flag_value = 1 if group == "correct" else -1

    # Mode logic
    if mode == "simple":
        selected_indices = answer_token_indices[:, 0] if group == "correct" else answer_token_indices[:, 1]
        group_probs = probabilities[torch.arange(logits.size(0)), selected_indices]

    elif mode == "pairs":
        group_probs = torch.zeros(logits.size(0), device=logits.device)
        for i in range(logits.size(0)):
            for pair in answer_token_indices[i]:
                selected_index = pair[0] if group == "correct" else pair[1]
                group_probs[i] += probabilities[i, selected_index]
            group_probs[i] /= answer_token_indices.size(1)

    elif mode in ["groups", "group_sum"]:
        assert flags_tensor is not None
        batch_size = logits.size(0)
        group_mass = torch.zeros(batch_size, device=logits.device)

        for i in range(batch_size):
            selected_probs = probabilities[i, answer_token_indices[i]]
            group_mass[i] = selected_probs[flags_tensor[i] == flag_value].sum()

        # For "group_sum" mode, return the sum of the group mass across the batch
        if mode == "group_sum":
            return group_mass.sum()

    else:
        raise ValueError("Invalid mode specified")

    return group_probs.mean()


def compute_rank_0_rate(
        logits: torch.Tensor, 
        answer_token_indices: torch.Tensor,
        positions: torch.Tensor = None,
        flags_tensor: torch.Tensor = None,
        group="correct",
        mode="simple"
) -> torch.Tensor:
    logits = get_positional_logits(logits, positions)
    probabilities = torch.softmax(logits, dim=-1)

    # Mode logic
    if mode == "simple":
        top_rank_indices = probabilities.argmax(dim=-1)
        correct_indices = answer_token_indices[:, 0] if group == "correct" else answer_token_indices[:, 1]
        rank_0_rate = (top_rank_indices == correct_indices).float().mean()

    elif mode == "pairs":
        rank_0_rate = torch.zeros(logits.size(0), device=logits.device)
        for i in range(logits.size(0)):
            for pair in answer_token_indices[i]:
                top_rank_index = probabilities[i].argmax()
                correct_index = pair[0] if group == "correct" else pair[1]
                rank_0_rate[i] += (top_rank_index == correct_index).float()
            rank_0_rate[i] /= answer_token_indices.size(1)

    elif mode == "groups":
        assert flags_tensor is not None
        rank_0_rate = torch.zeros(logits.size(0), device=logits.device)

        for i in range(logits.size(0)):
            selected_probs = probabilities[i, answer_token_indices[i]]
            top_rank_id = selected_probs.argmax()
            rank_0_rate[i] = (flags_tensor[i, top_rank_id] == 1).float() if group == "correct" else \
                             (flags_tensor[i, top_rank_id] == -1).float()

    else:
        raise ValueError("Invalid mode specified")

    return rank_0_rate.mean()


def compute_accuracy(
        logits: torch.Tensor,
        answer_token_indices: torch.Tensor,
        positions: torch.Tensor = None,
        flags_tensor: torch.Tensor = None,
        mode="simple"
) -> float:
    """
    Calculates the accuracy based on logits and answer token indices.

    Args:
        logits (torch.Tensor): Logits from the model.
        answer_token_indices (torch.Tensor): Indices of the tokens to compare.
        positions (torch.Tensor, optional): Positions to get logits at. Should be one position per batch item.
        flags_tensor (torch.Tensor, optional): Tensor for flags in 'groups' mode.
        mode (str, optional): The mode to use in compute_logit_diff function. Defaults to "simple".

    Returns:
        float: The accuracy as the proportion of cases where correct logits are greater than incorrect logits.
    """
    # Compute logit differences
    logit_diffs = compute_logit_diff(
        logits, 
        answer_token_indices, 
        positions, 
        flags_tensor, 
        per_prompt=True,  # Get per-prompt logit differences
        mode=mode
    )

    # Calculate accuracy
    correct_predictions = (logit_diffs > 0).float()  # Positive logit diff indicates correct prediction
    accuracy = correct_predictions.mean().item()  # Mean of correct predictions

    return accuracy

