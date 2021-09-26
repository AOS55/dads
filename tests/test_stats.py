from POET.stats import compute_centered_ranks, compute_ranks
from POET.novelty import euclidian_distance
import numpy as np


def test_compute_ranks():
  test_list = np.array([0.1, -0.2, 0.5, 0.3])
  rank_list = compute_ranks(test_list)
  assert rank_list.all() == np.array([1, 0, 3, 2]).all()


def test_centered_ranks():
  test_list = np.array([0.1, -0.2, 0.5, 0.3])
  centered_rank_list = compute_centered_ranks(test_list)
  assert centered_rank_list.all() == np.array([-0.16666666, -0.5,  0.5,  0.16666669], dtype=np.float32).all()


def test_novelty(k=5):
  distances = []
  candidate = np.array([0.1, 0.1, 0.5, 0.3, 0.4, 0.9])
  agent_1 = np.array([0.3, 0.5, 0.45, 0.4, 0.6, 0.2])
  agent_2 = np.array([0.3, 0.2, 0.4, 0.6, 0.2, 0.1])
  agent_3 = np.array([0.1, 0.2, 0.1, 0.4, 0.3, 0.6])
  agent_4 = np.array([0.15, 0.4, 0.5, 0.2, 0.3, 0.1])
  agent_5 = np.array([0.05, 0.1, 0.1, 0.4, 0.8, 0.2])
  agent_6 = np.array([0.3, 0.4, 0.2, 0.3, 0.6, 0.15])
  agent_list = [agent_1, agent_2, agent_3, agent_4, agent_5, agent_6]
  for agent in agent_list:
    distances.append(euclidian_distance(candidate, agent))
  distances = np.array(distances)
  top_k_indices = distances.argsort()[:k]
  top_k = distances[top_k_indices]
  assert type(top_k.mean()) == np.float64
