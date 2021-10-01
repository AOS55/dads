from POET.reproducer import Reproducer
from POET.novelty import compute_novelty_vs_archive


class Mutator:
  """
  Mutate environment agent pairs
  """
  def __init__(self,
               max_admitted,
               capacity,
               min_performance,
               mc_low,
               mc_high,
               reproducer_config):
    self.reproducer = Reproducer(env_categories=reproducer_config['env_categories'],
                                 master_seed=reproducer_config['master_seed'],
                                 max_children=reproducer_config['max_children'])

    # Mutator hyper-parameters
    self.max_admitted = max_admitted
    self.max_capacity = capacity
    self.min_performance = min_performance
    self.mc_low = mc_low
    self.mc_high = mc_high

  def mutate_env(self, ea_pairs):
    """
    Mutate the environment-agent pairs

    :param ea_pairs: instance of EnvPairs
    :return: ea_pairs
    """
    parent_list = []
    ea_pair_list = ea_pairs.pairs
    archived_pairs = ea_pairs.archived_pairs

    # Check if parent is eligible to reproduce for offspring
    for ea_pair in ea_pair_list:
      if self._eligible_to_reproduce(ea_pair):
        parent_list.append(ea_pair)

    # Generate child_list and rank by novelty, remove if mc_criteria not satisfied
    # TODO: Include a check for environment already existing
    child_list = self.reproducer.mutate_list(parent_list, ea_pairs)
    child_novelty_list = []
    for child in child_list:
      parent_agent_dir = [(pair.agent_config['log_dir'], pair.agent_config['save_dir']) for pair in parent_list
                          if pair.env_name == child.parent]
      score = ea_pairs.evaluate_agent_on_env(parent_agent_dir[0][0], parent_agent_dir[0][1], child.env_config)
      child = child._replace(agent_score=score)
      # if len(ea_pairs.pairs) >= self.max_capacity - self.reproducer.max_children:
      if not self._mc_satisfied(child.agent_score[0]):
        # child_list.remove(child)
        print(f'Child removal score: {child.agent_score[0]}')
        continue
      else:
        novelty_score = compute_novelty_vs_archive(ea_pairs=ea_pairs,
                                                   candidate_env=child.env_config,
                                                   k=5,
                                                   low=self.min_performance,
                                                   high=-self.min_performance)
        child_novelty_list.append((child, novelty_score))
    child_list = [x[0] for x in child_novelty_list]
    # Evaluate which policy to use on the new environment and ensure it satisfies the mc_criteria
    admitted = 0
    for child in child_list:
      agent_config, agent_score = ea_pairs.evaluate_transfer(candidate_env_config=child.env_config)
      agent_config['num_epochs'] = 0
      child = child._replace(agent_config=agent_config, agent_score=agent_score)
      if self._mc_satisfied(child.agent_score[0]):
        ea_pairs.pairs.append(child)
        admitted += 1
      if admitted >= self.max_capacity:
        break

    if len(ea_pairs.pairs) > self.max_capacity:
      num_removals = len(ea_pairs.pairs) - self.max_capacity
      ea_pair_list, archived_pairs = self._remove_oldest(ea_pairs.pairs, ea_pairs.archived_pairs, num_removals)
    return ea_pair_list, archived_pairs

  def _eligible_to_reproduce(self, pair):
    if pair.agent_score[0] >= self.min_performance:
      return True
    else:
      return False

  def _mc_satisfied(self, score):
    if score < self.mc_low or score > self.mc_high:
      return False
    else:
      return True

  @staticmethod
  def _remove_oldest(pairs, archived_pairs, num_removals):
    """
    Remove oldest pairs to limit size of active env buffers
    :param num_removals: the number of elements to remove from the end of the list
    :return: None
    """
    for pair in reversed(pairs):
      if num_removals > 0:
        pairs.remove(pair)
        archived_pairs.append(pair)
        num_removals -= 1
      else:
        break
    return pairs, archived_pairs
