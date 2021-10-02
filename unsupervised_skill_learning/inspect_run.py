import os
import pickle as pkl
from dads_OOP import EnvPairs

log_dir = 'log_dir/bipedal_walker_custom'
poet_log_file = os.path.join(log_dir, 'poet_vals.pkl')
with open(poet_log_file, 'rb') as f:
  ea_pairs, max_poet_iters = pkl.load(f)

print(ea_pairs.pairs)
print(ea_pairs.archived_pairs)
