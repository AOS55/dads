import numpy as np
from POET.stats import compute_centered_ranks


def euclidian_distance(x, y):
  """
  Given 2 numpy arrays calculate the euclidian distance

  :param x: first np array
  :param y: second np array
  :return: euclidian distance
  """
  n, m = len(x), len(y)
  if n > m:
    a = np.linalg.norm(y - x[:m])
    b = np.linalg.norm(y[-1] - x[m:])
  else:
    a = np.linalg.norm(x - y[:n])
    b = np.linalg.norm(x[-1] - y[n:])
  return np.sqrt(a**2 + b**2)


# TODO: Understand how this method works, look at diagram in report and track data
def compute_novelty_vs_archive(ea_pairs, candidate_env, k, low, high):
  distances = []
  candidate_env.update_pata_ec(ea_pairs.archived_agents, ea_pairs.active_agents, low, high)
  for agent in ea_pairs.archived_agents:
    distances.append(euclidian_distance(agent.pata_ec, candidate_env.pata_ec))

  for agent in ea_pairs.active_agents:
    distances.append(euclidian_distance(agent.pata_ec, candidate_env.pata_ec))

  distances = np.array(distances)
  top_k_indices = distances.argsort()[:k]
  top_k = distances[top_k_indices]
  return top_k.mean()
