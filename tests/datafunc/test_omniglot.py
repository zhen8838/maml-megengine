from datafunc.omniglot import OmniglotDataset
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import megengine.data.dataset as mds
import megengine.data.sampler as msp
from megengine.data.dataloader import DataLoader
from models import OmniglotFC
import megengine as meg


def build_dataset(root=Path('/home/zqh/data/omniglot-py'),
                  nway=5, kshot=1, kquery=1):

  ds = OmniglotDataset(root, nway, kshot, kquery)
  return ds


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
    im_s.shape  # [ 32 ,5 ,1 ,105, 105, 3]
    lb_s.shape  # (32, 5, 1, 5)
    break


def test_OmniglotFC():
  net = OmniglotFC(28 * 28, 5)
  x = meg.tensor(np.random.randn(5, 28 * 28), dtype='float32')
  y: meg.Tensor = net(x)

  assert y.shape == (5, 5)
