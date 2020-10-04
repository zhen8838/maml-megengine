import megengine as meg
from numpy.core.numeric import outer
import megengine.data.dataset as mds
from megengine.data.transform import Transform
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import cv2


class OmniglotDataset(mds.Dataset):
  def __init__(self, root: Path, nway: int, kshot: int, kquery: int, mode: str = 'train', out_size=(28, 28), out_format: Optional[str] = 'fc') -> None:
    super().__init__()
    if mode.lower() == 'train':
      self.root = root / 'images_background'
    else:
      self.root = root / 'images_evaluation'

    self.nway = nway
    self.kshot = kshot
    self.kquery = kquery
    self.char_fold, self.image_list = self.get_char_fold(self.root, shuffle=True)
    self.char_fold_idx = np.arange(len(self.char_fold))
    self.out_size = out_size
    self.im_process = {'fc': self.im_process_fc,
                       'cnn': self.im_process_cnn}[out_format]

  def get_char_fold(self, root: Path, shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """get all omniglot sub sub dir, eg. 

      ['root_dir/Anglo-Saxon_Futhorc/character22'
       'root_dir/Japanese_(katakana)/character46']

    Args:
        root (Path): root
        shuffle (bool, optional): Defaults to True.

    Returns:
        np.ndarray: char fold list, shape [n]
        np.ndarray: inner image list, shape [n,m]
    """
    char_fold = []
    image_list = []
    for sub in root.iterdir():
      for ssub in sub.iterdir():
        char_fold.append(str(ssub))
        image_list.append([str(path) for path in ssub.iterdir()])
    char_fold = np.array(char_fold)
    image_list = np.array(image_list)
    if shuffle:
      fold_idx = np.arange(len(char_fold))
      np.random.shuffle(fold_idx)
      char_fold = char_fold[fold_idx]
      image_list = image_list[fold_idx]
    return char_fold, image_list

  def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray,
                                             np.ndarray, np.ndarray]:
    image_paths, labels = self.sample_image_path_label(index)
    images = self.read_image_path(image_paths)
    # labels = self.to_categorical(labels)
    return self.split_support_query(images, labels)

  def __len__(self) -> int:
    return len(self.char_fold)

  def sample_image_path_label(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
    """ get images,labels. 
      image_paths : shape = [nway,kshot + kquery]
      labels : shape = [nway,kshot + kquery], int32 

    Args:
        index (int): i

    Returns:
        Tuple[np.ndarray,np.ndarray]: image_paths,labels
    """
    sampled_char_fold: np.ndarray = np.random.choice(self.char_fold_idx,
                                                     self.nway, replace=False)
    labels = np.arange(self.nway)
    out_labels, out_images = [], []
    for label, fold in zip(labels, sampled_char_fold):
      out_images.append(
          np.random.choice(self.image_list[fold], self.kshot + self.kquery, replace=True))
      out_labels.append(
          np.ones([self.kshot + self.kquery], dtype='int32') * label)
    return np.array(out_images), np.array(out_labels)

  def im_resize(self, x):
    return cv2.resize(x, self.out_size).astype('float32')

  def im_read(self, x):
    return np.expand_dims(cv2.imread(x, cv2.IMREAD_GRAYSCALE), -1)

  def im_norm(self, x, mean=127.5, std=127.5):
    return (x - mean) / std

  def im_trans(self, x):
    return np.ascontiguousarray(np.rollaxis(x, 2, start=0))

  def im_process_cnn(self, x: np.ndarray):
    return self.im_trans(self.im_norm(
        np.expand_dims(self.im_resize(x), -1)))

  def im_process_fc(self, x: np.ndarray):
    return np.ravel(self.im_norm(self.im_resize(x), -1))

  def read_image_path(self, image_paths: np.ndarray) -> np.ndarray:
    """read image paths

    Args:
        image_paths (np.ndarray): images , shape = [nway,kshot + kquery,height, weight, channel]

    Returns:
        np.ndarray: image_paths,labels
    """

    images = [[self.im_process(im) for im in map(self.im_read, nway)] for nway in image_paths]
    return np.array(images)

  def to_categorical(self, labels: np.ndarray) -> np.ndarray:
    """ labels to_categorical

    Args:
        labels (np.ndarray): [nway,kshot+kquery]

    Returns:
        np.ndarray: [nway,kshot+kquery,nway]
    """
    return np.eye(self.nway, dtype='float32')[labels]

  def split_support_query(self, images: np.ndarray,
                          labels: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    """ split_support_query

    Args:
        images (np.ndarray): 
        labels (np.ndarray): 

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 

        `(images_support,labels_support,images_query,labels_query)`
    """
    return (images[:, :self.kshot, ...],
            labels[:, :self.kshot, ...],
            images[:, self.kshot:, ...],
            labels[:, self.kshot:, ...])
