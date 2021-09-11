import os
from POET.utils import read_perf_json, read_json
from POET.reproduce_ops import Reproducer

def eligible_to_reproduce(env_path, min_performance):
  perf = read_perf_json(env_path)
  if perf.tot_return >= min_performance:
    return True
  return False


def env_reproduce(setup_child, parent_list, max_children, master_seed, env_categories, log_dir):
  reproducer = Reproducer(master_seed, env_categories)
  children_added = 0
  children_list = {}
  env_tree = read_json(os.path.join(log_dir, 'tree.json'))
  while max_children < children_added:
    for parent in parent_list:
      child = reproducer.mutate(parent)
      score, env_tree = setup_child(child, log_dir, parent, env_tree)
      children_list[child] = score
      # if mc_satisfied(score):
      #   # TODO: Calculate novelty score
  return children_list


def mc_satisfied(score, mc_low=-1000, mc_high=2000):
  if score < mc_low or score > mc_high:
    return False
  else:
    return True


def rank_by_novelty(child_list):
  # TODO: Implement a rank by novelty (PATA-EC) may be best here?
  return None


def evaluate_candidates(env_list, target_env):
  # TODO: Find a method to evaluate each agent on the given environment then select appropriate one
  return None


def remove_oldest(env_list, num_removals):
  # TODO: Remove where appropriate (PATA-EC may be used here)
  return None



