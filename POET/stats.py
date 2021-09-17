import numpy as np


def compute_ranks(x):
  """
  Rank of each element in x [0, len(x)]

  :param x: input np.array
  :return: ranks [0, len(x)]
  """
  assert x.ndim == 1
  ranks = np.empty(len(x), dtype=int)
  ranks[x.argsort()] = np.arange(len(x))
  return ranks


def compute_centered_ranks(x):
  """
  Given a ranked vector, normalize the data between [-0.5, +0.5], i.e. range=1, mean=0

  :param x: ranked vector
  :return: centered rank vector
  """
  y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
  y /= (x.size - 1)
  y -= 0.5
  return y
