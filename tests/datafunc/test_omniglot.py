import sys
import os
sys.path.insert(0, os.getcwd())

from datafuncs.omniglot import OmniglotDataset
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import megengine.data.dataset as mds
import megengine.data.sampler as msp
import megengine.functional as F
import megengine.module as M
from megengine.module.module import _expand_structure, _is_parameter
from megengine.data.dataloader import DataLoader
from models import OmniglotFC, OmniglotCNN
import megengine as meg
import megengine.optimizer as optim
from typing import List, Dict, Tuple
from algorithms import MAML
from megengine.jit import trace


def build_dataset(root=Path('/home/zqh/data/omniglot-py'),
                  nway=5, kshot=1, kquery=1):
  ds = OmniglotDataset(root, nway, kshot, kquery)
  return ds


def build_dataloader(root=Path('/home/zqh/data/omniglot-py'),
                     nway=5, kshot=1, kquery=1, batch_size=32):
  train_ds = OmniglotDataset(root, nway, kshot, kquery, mode='train')
  train_smp = msp.SequentialSampler(train_ds,
                                    drop_last=True,
                                    batch_size=batch_size)
  train_loader = DataLoader(train_ds,
                            sampler=train_smp,
                            num_workers=4)

  return train_loader


def test_OmniglotDataset_sample_image_path_label():
  ds = build_dataset()
  image_paths, labels = ds.sample_image_path_label(1)
  images = ds.read_image_path(image_paths)
  images.shape
  plt.imshow(images[0][0])
  plt.imshow(images[0][1])
  plt.imshow(images[1][0])
  plt.imshow(images[1][1])


def test_OmniglotDataset_batch():
  ds = build_dataset()
  sampler = msp.SequentialSampler(dataset=ds, batch_size=32, drop_last=True)
  for indices in sampler:
    print(indices)
    break


def test_OmniglotDataset_loader():
  ds = build_dataset()
  sampler = msp.SequentialSampler(dataset=ds, batch_size=32, drop_last=True)
  dataload = DataLoader(dataset=ds, sampler=sampler, num_workers=4)
  for im_s, lb_s, im_q, lb_q in dataload:
    im_s.shape  # [ 32 ,5 ,1 ,105, 105, 1]
    lb_s.shape  # (32, 5, 1, 5)
    break


def test_OmniglotFC():
  net = OmniglotFC(28 * 28, 5)
  x = meg.tensor(np.random.randn(5, 28 * 28), dtype='float32')
  y: meg.Tensor = net(x)

  assert y.shape == (5, 5)


def test_OmniglotCNN():
  nway = 20
  net = OmniglotCNN(output_size=nway, hidden_size=128, layers=3)
  x = meg.tensor(np.random.randn(5, 28, 28), dtype='float32')
  y: meg.Tensor = net(x)

  assert y.shape == (5, nway)


def test_Clone_model():
  # 必须要将新参数clone到另一个模型中，才可以继续
  train_loader = build_dataloader()
  image_support = meg.tensor(dtype='float32')
  label_support = meg.tensor(dtype="int32")

  model = OmniglotFC(28 * 28, 5)

  model.train()
  loss_fn = F.cross_entropy_with_softmax
  optimizer = optim.SGD(model.parameters(), lr=0.05)
  iters = iter(train_loader)

  (images_support, labels_support, images_query, labels_query) = next(iters)
  i = 0
  image_support.set_value(images_support[i])
  label_support.set_value(labels_support[i])
  image_support = F.remove_axis(image_support, 1)
  label_support = F.remove_axis(label_support, 1)

  support_out = model.forward(image_support)
  support_loss = loss_fn(support_out, label_support)

  # 对需要梯度更新的参数进行更新
  params = list(model.parameters(requires_grad=True))
  params[0] = meg.tensor(np.ones((5)), dtype='float32')

  grads = F.grad(support_loss, params, use_virtual_grad=False)

  fast_weights = [p - 0.5 * g for g, p in zip(grads, params)]


def test_class_name_assign():
  class CLS:
    def __init__(self):
      self.name = "title"
      self.addr = "http://xxx.com"
      self.bais = meg.tensor(np.ones(5), dtype='float32')

  c = CLS()
  d = dict(name=c.__dict__)
  d['name']['name'] = 'new_title'
  d['name']['bais'] = meg.tensor(np.zeros(5), dtype='float32')
  # 利用一个dict保存key：dict即可。


def test_maml_update_var():
  model = OmniglotFC(28 * 28, 5)
  model.train()
  loss_fn = F.cross_entropy_with_softmax
  old_params = list(model.parameters())
  maml = MAML(model)
  params = list(maml.named_parameters.values())
  optimizer = optim.SGD(old_params, lr=0.05)
  optimizer.zero_grad()
  support_out = model.forward(meg.tensor(np.random.randn(5, 28 * 28), dtype='float32'))
  support_loss = loss_fn(support_out, meg.tensor(np.random.randint(0, 5, (5)), dtype='int32'))
  optimizer.backward(support_loss)
  optimizer.step()
  assert id(old_params[0]) == id(params[0])
  # 手动update

  grads = F.grad(support_loss, params, use_virtual_grad=False)
  fast_weights = [p - 0.5 * g for g, p in zip(grads, params)]
  named_update = dict(zip(maml.named_parameters.keys(), fast_weights))
  named_old = dict(zip(maml.named_parameters.keys(), old_params))
  maml.replace_parameter(maml.module_table, named_update)
  # 被替换为新的值后就无法通过model.parameters()找到了。
  after_params = list(model.parameters())
  maml.module_table['classifier'].bias
  named_update['classifier.bias']
  mods = list(model.modules())
  mods[1].bias

  maml.replace_parameter(maml.module_table, named_old)


def replace_parameter(module_table,
                      named_updates):
  for key, value in named_updates.items():
    module_name, param_name = key.rsplit('.', 1)
    exec(f"module_table['{module_name}'].{param_name}.set_value(value)")


def test_grad_twice():
  # model define
  model = M.Sequential(M.Linear(10, 20),
                       M.Linear(20, 10),
                       M.Linear(10, 5))
  model.train()
  named_param = dict(list(model.named_parameters(requires_grad=True)))
  named_module = dict(list(model.named_children()))
  name_keys = list(named_param.keys())
  params = list(named_param.values())
  loss_fn = F.cross_entropy_with_softmax
  optimizer = optim.SGD(params, lr=0.003)

  # forward once
  optimizer.zero_grad()
  x1 = meg.tensor(np.random.randn(5, 10), dtype='float32')
  y1 = meg.tensor(np.random.randint(0, 5, (5)), dtype='int32')
  loss = loss_fn(model(x1), y1)
  grads = F.grad(loss, params,
                 use_virtual_grad=False,
                 return_zero_for_nodep=False)
  fast_weights = [p - 0.5 * g for g, p in zip(grads, params)]

  # manual update params
  replace_parameter(named_module, dict(zip(name_keys, fast_weights)))

  # forward twice
  x2 = meg.tensor(np.random.randn(5, 10), dtype='float32')
  y2 = meg.tensor(np.random.randint(0, 5, (5)), dtype='int32')
  loss2 = loss_fn(model(x2), y2)
  # got error
  replace_parameter(named_module, named_param)
  optimizer.backward(loss2)
  optimizer.step()


class CustomModel(M.Module):
  def __init__(self) -> None:
    super().__init__()
    self.l1 = M.Linear(10, 20)
    self.l2 = M.Linear(20, 10)
    self.l3 = M.Linear(10, 5)

  def forward(self, x, weights: Dict[str, meg.Tensor] = None):
    x = self.l1._calc_linear(x, weights['l1.weight'],
                             weights['l1.bias'])
    x = self.l2._calc_linear(x, weights['l2.weight'],
                             weights['l2.bias'])
    x = self.l3._calc_linear(x, weights['l3.weight'],
                             weights['l3.bias'])
    return x


@trace
def train_func(x1, y1, x2, y2, *, loss_fn, opt, net, keys, params):
  # 此处data和label不再需要先创建tensor然后通过set_value赋值，这些操作在trace内部完成
  logits = net(x1, weights=dict(zip(keys, params)))
  loss = loss_fn(logits, y1)
  grads = F.grad(loss, params, use_virtual_grad=False)
  fast_weights = [p - 0.5 * g for g, p in zip(grads, params)]

  # forward twice
  loss2 = loss_fn(net(x2, weights=dict(zip(keys, fast_weights))), y2)
  opt.backward(loss2)


def test_grad_twice_method_2():
  # model define
  model = CustomModel()
  model.train()
  named_param = dict(list(model.named_parameters(requires_grad=True)))
  name_keys = list(named_param.keys())
  params = list(named_param.values())

  loss_fn = F.cross_entropy_with_softmax
  optimizer = optim.SGD(params, lr=0.003)

  # forward once
  optimizer.zero_grad()
  x1 = meg.tensor(np.random.randn(5, 10), dtype='float32')
  y1 = meg.tensor(np.random.randint(0, 5, (5)), dtype='int32')
  x2 = meg.tensor(np.random.randn(5, 10), dtype='float32')
  y2 = meg.tensor(np.random.randint(0, 5, (5)), dtype='int32')

  train_func(x1, y1, x2, y2, loss_fn=loss_fn, opt=optimizer,
             net=model, keys=name_keys, params=params)
  optimizer.step()


class CustomModel3(M.Module):
  def __init__(self) -> None:
    super().__init__()
    self.l1 = M.Linear(10, 20)
    self.l2 = M.Linear(20, 10)
    self.l3 = M.Linear(10, 5)

  def forward(self, x, weights: Dict[str, meg.Tensor] = None):
    x = self.l1._calc_linear(x, weights[1], weights[0])
    x = self.l2._calc_linear(x, weights[3], weights[2])
    x = self.l3._calc_linear(x, weights[5], weights[4])
    return x


@trace
def train_func3(x1, y1, x2, y2, *, loss_fn, opt, net, params):
  loss = loss_fn(net(x1, weights=params), y1)
  grads = F.grad(loss, params, use_virtual_grad=False)
  fast_weights = [p - 0.5 * g for g, p in zip(grads, params)]
  # forward twice
  loss2 = loss_fn(net(x2, weights=fast_weights), y2)
  opt.backward(loss2)


def test_grad_twice_method_3():
  # model define
  model = CustomModel3()
  model.train()
  named_param = dict(list(model.named_parameters(requires_grad=True)))
  params = list(named_param.values())
  external_params = [meg.Parameter(np.random.normal(size=p.shape), dtype='float32') for p in params]

  loss_fn = F.cross_entropy_with_softmax
  optimizer = optim.SGD(external_params, lr=0.003)

  # forward once
  optimizer.zero_grad()
  x1 = meg.tensor(np.random.randn(5, 10), dtype='float32')
  y1 = meg.tensor(np.random.randint(0, 5, (5)), dtype='int32')
  x2 = meg.tensor(np.random.randn(5, 10), dtype='float32')
  y2 = meg.tensor(np.random.randint(0, 5, (5)), dtype='int32')

  train_func3(x1, y1, x2, y2, loss_fn=loss_fn, opt=optimizer,
              net=model, params=external_params)
  optimizer.step()


test_grad_twice_method_3()
