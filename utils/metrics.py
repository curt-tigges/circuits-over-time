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


# ======= NEW METRICS =======
# We want to be able to compute the following metrics:
# - Logit difference between the correct and incorrect answer, or pairs or groups of answers
# - Probability difference between the correct and incorrect answer, or pairs or groups of answers;
#   in the groups case, we want to be able to average the probabilities or sum them
# - Probability mass of the correct or incorrect answer, or pairs or groups of answers
# - Rank 0 rate of the correct or incorrect answer, or pairs or groups of answers
# The old metrics are still available in metrics_old.py, but they are deprecated and will be removed in the future.


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
    def __init__(self, name, metric_fn, normalization_fn=None):
        self.name = name
        self.metric_fn = metric_fn
        self.normalization_fn = normalization_fn

    def __call__(self, logits, *args, **kwargs):
        if self.normalization_fn is not None:
            return self.normalization_fn(logits, self.metric_fn, *args, **kwargs)
        return self.metric_fn(logits, *args, **kwargs)
    

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

