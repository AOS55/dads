from POET.novelty import euclidian_distance
import numpy as np


def test_euclidian_distance():
  x = np.array([0.4, 0.2, -0.3])
  y = np.array([0.3, -0.15, 0.9])
  dist = euclidian_distance(x, y)
  assert type(dist) == np.float64
