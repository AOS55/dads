import os
import pickle as pkl
from dads_OOP import EnvPairs


def get_poet_vals():
  log_dir = 'log_dir/bipedal_walker_custom'
  poet_log_file = os.path.join(log_dir, 'poet_vals.pkl')
  with open(poet_log_file, 'rb') as f:
    ea_pairs, max_poet_iters = pkl.load(f)
  return ea_pairs, max_poet_iters


if __name__ == '__main__':
  ea_pairs, max_poet_iters = get_poet_vals()
  print(len(ea_pairs.pairs))
  print(len(ea_pairs.archived_pairs))
  print(max_poet_iters)

