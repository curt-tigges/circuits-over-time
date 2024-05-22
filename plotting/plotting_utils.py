from typing import List, Union
color_palette = {
  "pythia-70m": "#EE908D",
  "pythia-160m": "#F8D592",
  "pythia-1b": "#8F948D",
  "pythia-410m": "#B2B4D9",
  "pythia-12b": "#B46F90",
  "pythia-1.4b": "#A7C2D0",
  "pythia-6.9b": "#BF8271",
  "pythia-2.8b": "#8CD9AF"
}

core_models = list(color_palette.keys())

def steps2tokens(checkpoints: Union[int, List[int]]) -> Union[int, List[int]]:
  """Convert checkpoint steps to tokens.

  Args:
      checkpoints (List[int]): The list of checkpoint steps.

  Returns:
      List[int]: The list of tokens.
  """
  if isinstance(checkpoints, list):
    return [ckpt * 2097152 for ckpt in checkpoints]
  else:
    return checkpoints * 2097152