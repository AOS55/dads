from reproducer import Reproducer


class Mutator:
  """
  Mutate Environment agent pairs
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
    Mutate the environment agent-pairs into children and add to list of ea_pairs

    :param ea_pairs:
    :return: ea_pairs
    """
    parent_list = []
    for pair in ea_pairs:
      if self._eligible_to_reproduce(pair):
        parent_list.append(pair)
    # TODO: need to evaluate an mc_score for each child environment to test mc_criterion
    child_list = self.reproducer.mutate_list(parent_list)
    for child in child_list:
      if not self._mc_satisfied(child['performance'][0]):
        child_list.remove(child)  # remove child if too easy or too hard with parent agent (i.e. kills behaviours)
    child_list = self._rank_by_novelty(ea_pairs, child_list)

  def _eligible_to_reproduce(self, pair):
    if pair['performance'][0] >= self.min_performance:
      return True
    else:
      return False

  def _mc_satisfied(self, score):
    if score < self.mc_low or score > self.mc_high:
      return False
    else:
      return True

  @staticmethod
  def _rank_by_novelty(ea_pairs, child_list):
    # TODO: Implement a PATA-EC form of of novelty ranking, look @ UBER POET implementation
    return child_list
