import sys
import os
sys.path.insert(0, os.getcwd())
from datafuncs import OmniglotDataset
from megengine.data import DataLoader, SequentialSampler
from models import OmniglotFC
from algorithms import MAML
import megengine as meg
from pathlib import Path
import megengine.functional as F
import megengine.module as M
import megengine.optimizer as optim


def build_dataset(root=Path('/home/zqh/data/omniglot-py'),
                  nway=5, kshot=1, kquery=1, batch_size=32):

  train_ds = OmniglotDataset(root, nway, kshot, kquery, mode='train')
  train_smp = SequentialSampler(train_ds,
                                drop_last=True,
                                batch_size=batch_size)
  train_loader = DataLoader(train_ds,
                            sampler=train_smp,
                            num_workers=4)

  val_ds = OmniglotDataset(root, nway, kshot, kquery, mode='val')
  val_smp = SequentialSampler(train_ds,
                              drop_last=True,
                              batch_size=batch_size)
  val_loader = DataLoader(val_ds,
                          sampler=val_smp,
                          num_workers=4)

  return train_loader, val_loader


def main():
  nway = 5
  batch_size = 32
  train_loader, val_loader = build_dataset(nway=nway, batch_size=batch_size)
  model = OmniglotFC(28 * 28, nway)
  model.train()
  maml = MAML(model)

  loss_fn = F.cross_entropy_with_softmax
  optimizer = optim.SGD(maml.trainable_params, lr=0.05)
  accuracy = F.accuracy
  adapt_data = meg.tensor(dtype='float32')
  adapt_label = meg.tensor(dtype='int32')
  eval_data = meg.tensor(dtype='float32')
  eval_label = meg.tensor(dtype='int32')

  for iteration, (images_support, labels_support,
                  images_query, labels_query) in enumerate(train_loader):
    optimizer.zero_grad()
    meta_train_error = 0.0
    meta_train_accuracy = 0.0
    for i in range(batch_size):
      (image_support, label_support,
       image_query, label_query) = (images_support[i], labels_support[i],
                                    images_query[i], labels_query[i])
      adapt_data.set_value(image_support)
      adapt_data = F.remove_axis(adapt_data, 1)
      adapt_label.set_value(label_support)
      adapt_label = F.remove_axis(adapt_label, 1)
      loss = loss_fn(model.forward(adapt_data), adapt_label)
      gradients = F.grad(loss,
                         maml.trainable_params,
                         use_virtual_grad=False)

      fast_weights = [p - 0.5 * g for p, g in
                      zip(maml.trainable_params, gradients)]

      maml.replace_fast_parameter(fast_weights)
      # Evaluate the adapted model
      eval_data.set_value(image_query)
      eval_data = F.remove_axis(eval_data, 1)
      eval_label.set_value(label_query)
      eval_label = F.remove_axis(eval_label, 1)

      predictions = model.forward(eval_data)
      valid_error = loss_fn(predictions, eval_label)
      valid_accuracy = accuracy(predictions, eval_label)
      optimizer.backward(valid_error)
      meta_train_error += valid_error.numpy().item()
      meta_train_accuracy += valid_accuracy.numpy().item()

    for p in maml.trainable_params:
      p.grad = p.grad * (1.0 / batch_size)
    optimizer.step()
    print('\n')
    print('Iteration', iteration)
    print('Meta Train Error', meta_train_error / batch_size)
    print('Meta Train Accuracy', meta_train_accuracy / batch_size)


if __name__ == "__main__":
  main()
