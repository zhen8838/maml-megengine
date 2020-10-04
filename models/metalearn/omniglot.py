import megengine as meg
import megengine.module as M
import megengine.functional as F
import initializers as init
import megengine.module.init as minit
import layers as kl
import functions as kf


def fc_init_(module: M.Linear):
  if hasattr(module, 'weight') and module.weight is not None:
    init.truncated_normal_(module.weight, mean=0.0, std=0.01)
  if hasattr(module, 'bias') and module.bias is not None:
    minit.zeros_(module.bias)


def maml_init_(module: M.Conv2d):
  minit.xavier_uniform_(module.weight, gain=1.0)
  minit.zeros_(module.bias)


class LinearBlock(M.Module):

  def __init__(self, input_size: int, output_size: int):
    super(LinearBlock, self).__init__()
    self.relu = M.ReLU()
    self.normalize = M.BatchNorm1d(
        output_size,
        affine=True,
        momentum=0.999,
        eps=1e-3,
        track_running_stats=False,
    )
    self.linear = M.Linear(input_size, output_size)
    fc_init_(self.linear)

  def forward(self, x):
    x = self.linear(x)
    x = self.normalize(x)
    x = self.relu(x)
    return x


class ConvBlock(M.Module):

  def __init__(self,
               in_channels: int,
               out_channels: int,
               kernel_size: int,
               max_pool=True,
               max_pool_factor=1.0):
    super(ConvBlock, self).__init__()
    stride = (int(2 * max_pool_factor), int(2 * max_pool_factor))
    if max_pool:
      self.max_pool = M.MaxPool2d(
          kernel_size=stride,
          stride=stride)
      stride = (1, 1)
    else:
      self.max_pool = lambda x: x
    self.normalize = M.BatchNorm2d(
        out_channels,
        affine=True)
    minit.uniform_(self.normalize.weight)
    self.relu = M.ReLU()

    self.conv = M.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=1,
        bias=True,
    )
    maml_init_(self.conv)

  def forward(self, x):
    x = self.conv(x)
    x = self.normalize(x)
    x = self.relu(x)
    x = self.max_pool(x)
    return x


class ConvBase(M.Sequential):

  # NOTE:
  #     Omniglot: hidden=64, channels=1, no max_pool
  #     MiniImagenet: hidden=32, channels=3, max_pool

  def __init__(self,
               output_size,
               hidden=64,
               channels=1,
               max_pool=False,
               layers=4,
               max_pool_factor=1.0):
    core = [ConvBlock(channels,
                      hidden,
                      (3, 3),
                      max_pool=max_pool,
                      max_pool_factor=max_pool_factor),
            ]
    for _ in range(layers - 1):
      core.append(ConvBlock(hidden,
                            hidden,
                            kernel_size=(3, 3),
                            max_pool=max_pool,
                            max_pool_factor=max_pool_factor))
    super(ConvBase, self).__init__(*core)


class OmniglotFC(M.Module):
  """

  [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models.py)

  **Description**

  The fully-connected network used for Omniglot experiments, as described in Santoro et al, 2016.

  **References**

  1. Santoro et al. 2016. “Meta-Learning with Memory-Augmented Neural Networks.” ICML.

  **Arguments**

  * **input_size** (int) - The dimensionality of the input.
  * **output_size** (int) - The dimensionality of the output.
  * **sizes** (list, *optional*, default=None) - A list of hidden layer sizes.

  **Example**
  ~~~python
  net = OmniglotFC(input_size=28**2,
                   output_size=10,
                   sizes=[64, 64, 64])
  ~~~

  """

  def __init__(self, input_size, output_size, sizes=None):
    super().__init__()
    if sizes is None:
      sizes = [256, 128, 64, 64]
    layers = [LinearBlock(input_size, sizes[0]), ]
    for s_i, s_o in zip(sizes[:-1], sizes[1:]):
      layers.append(LinearBlock(s_i, s_o))
    layers = M.Sequential(*layers)
    self.features = M.Sequential(
        kl.Flatten(),
        layers,
    )
    self.classifier = M.Linear(sizes[-1], output_size)
    fc_init_(self.classifier)
    self.input_size = input_size

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x


class OmniglotCNN(M.Module):
  """

  [Source](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/models.py)

  **Description**

  The convolutional network commonly used for Omniglot, as described by Finn et al, 2017.

  This network assumes inputs of shapes (1, 28, 28).

  **References**

  1. Finn et al. 2017. “Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.” ICML.

  **Arguments**

  * **output_size** (int) - The dimensionality of the network's output.
  * **hidden_size** (int, *optional*, default=64) - The dimensionality of the hidden representation.
  * **layers** (int, *optional*, default=4) - The number of convolutional layers.

  **Example**
  ~~~python
  model = OmniglotCNN(output_size=20, hidden_size=128, layers=3)
  ~~~

  """

  def __init__(self, output_size=5, hidden_size=64, layers=4):
    super().__init__()
    self.hidden_size = hidden_size
    self.base = ConvBase(output_size=hidden_size,
                         hidden=hidden_size,
                         channels=1,
                         max_pool=False,
                         layers=layers)
    self.features = M.Sequential(
        kl.Lambda(lambda x: F.reshape(x, (-1, 1, 28, 28))),
        self.base,
        kl.Lambda(lambda x: kf.mean(x, axis=[2, 3])),
        kl.Flatten(),
    )
    self.classifier = M.Linear(hidden_size, output_size, bias=True)

  def forward(self, x):
    x = self.features(x)
    x = self.classifier(x)
    return x
