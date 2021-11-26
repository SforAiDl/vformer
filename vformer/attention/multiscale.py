import numpy

import torch
import torch.nn as nn

def pool_attention(input, thw, pool, norm):
  dim = input.dim()
  if dim == 3:
    input =  input.unsqueeze(1)
  elif dim != 4:
    raise NotImplementedError(f"Unsupported input dimension {input.shape}")
  T,H,W = thw
  B,N,L,C = input.shape
