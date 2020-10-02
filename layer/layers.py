import megengine.functional as F
import megengine.module as M
import megengine as meg
from typing import Callable


class Flatten(M.Module):
  def __init__(self, start_axis: int = 1, end_axis: int = -1):
    """ Flatten layer equal keras Flatten.

    Args:
        start_axis (int, optional): Defaults to 1.
        end_axis (int, optional): Defaults to -1.
    """
    super().__init__()
    self.start_axis = start_axis
    self.end_axis = end_axis

  def forward(self, inputs: meg.Tensor):
    return F.flatten(inputs, self.start_axis, self.end_axis)


class Lambda(M.Module):
  def __init__(self, lmb: Callable[..., meg.Tensor]):
    """ Lambda layer equal keras Lambda.

    """
    super().__init__()
    assert callable(lmb)
    self.lmb = lmb

  def forward(self, *args, **kwargs):
    return self.lmb(*args, **kwargs)
