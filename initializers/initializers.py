from scipy.stats import truncnorm
import megengine.module.init as minit
from megengine import Tensor, Graph


def truncated_normal_(tensor: Tensor, mean=0.0, std=1.0):
  """ use truncated_normal init parameter inplace

  PT doesn't have truncated normal.
  https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18

  Args:
      tensor (meg.Tensor): parameter
      mean (float, optional): Defaults to 0.0.
      std (float, optional): Defaults to 1.0.
  """
  values = truncnorm.rvs(-2, 2, size=tensor.shape)
  values = mean + std * values
  with Graph(eager_evaluation=True):
    tensor.set_value(values)
