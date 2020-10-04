import megengine as meg
import megengine.module as M
import megengine.functional as F
from typing import Iterable


def mean(inputs: meg.Tensor,
         axis: Iterable[int],
         keepdims=False) -> meg.Tensor:
  inp = inputs
  if keepdims:
    for ax in axis:
      inp = F.mean(inp, ax, keepdims=keepdims)
  else:
    axis = sorted(axis)
    for i, ax in enumerate(axis):
      inp = F.mean(inp, ax - i, keepdims=keepdims)
  return inp
