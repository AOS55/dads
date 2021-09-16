import numpy as np


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


def compute_novelty_vs_archive(archived_agents, active_agents, candidate_env, k, low, high):
  distances = []
  candidate_env.update_pata_ec(archived_agents, active_agents, low, high)
  for point in archived_agents['performance'][0]:
    distances.append(euclidian_distance())