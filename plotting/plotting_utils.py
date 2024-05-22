color_palette = {
  "pythia-70m": "#EE908D",
  "pythia-160m": "#F8D592",
  "pythia-1b": "#8CD9AF",
  "pythia-410m": "#B2B4D9",
  "pythia-1.4b": "#A7C2D0",
  "pythia-2.8b": "#8F948D",
  "pythia-6.9b": "#BF8271",
  "pythia-12b": "#B46F90",
  
}

core_models = list(color_palette.keys())

def steps2tokens(step):
    return int(step) * 2097152