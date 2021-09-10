from POET.reproduce_ops import Reproducer
from POET.model import evaluate_candidates


class Mutate:
  """
  Mutate environment agent pairs
  """

  def __init__(self, model, reproducer_config: dict, max_admitted: int, max_capacity: int,
               mc_crit: tuple, passing_score: float) -> None:
    """
    Mutate environment constructor

    :param model: the environment-policy model object
    :param reproducer_config: configuration dictionary for Reproducer class
    :param max_admitted: max admitted children per mutation
    :param max_capacity: max capacity of ea_list
    :param mc_crit: max and minimum return before moving on to another environment
    :param passing_score: minimum env performance to be eligable to reproduce
    """
    self.model = model  # feed through model object from POET
    self.reproducer = Reproducer(land_list=reproducer_config['land_list'],
                                 max_children=reproducer_config['max_children'])

    # Mutate hyperparameters
    self.max_admitted = max_admitted
    self.max_capacity = max_capacity
    self.mc_crit_low = mc_crit[0]
    self.mc_crit_high = mc_crit[1]
    self.passing_score = passing_score

    # Initialize list attributes to be used throughout mutations
    self.parent_list = []
    self.child_list = []
    self.ea_list = []
    self.archived_policies = []

  def mutate_env(self, ea_list: list) -> list:
    """
    Mutate the environment agent pairs into children and add to ea_list

    :param ea_list: list of env, action pairs [(E as env_config, agent as policy state_dict), ...]
    :return: ea_list with new mutated envs and dropped old pairs if reaching max capacity
    """
    self.ea_list = ea_list
    self.parent_list = []
    self.list_elgible_parents()
    self.child_list = self.reproducer.reproduce_family(self.parent_list)
    print(f'number of children before mc removal: {len(self.child_list)}')
    self.check_child_mc()
    print(f'number of children after mc removal: {len(self.child_list)}')
    self.rank_by_novelty()
    self.add_to_ea_list()
    len_pairs = len(self.ea_list)
    if len_pairs > self.max_capacity:
      num_removals = len_pairs - self.max_capacity
      self.remove_oldest(num_removals)
    return self.ea_list

  def list_elgible_parents(self):
    """
    If a pair in the ea list is eligible to reproduce add it to parent list

    :return: None
    """
    eligible_parents = [pair for pair in self.ea_list if
                        self.model.eligible_to_reproduce.remote(self.model, passing_score=self.passing_score,
                                                                pair=pair)]

    self.parent_list.extend(eligible_parents)
    print(f'Elgible parents are: {self.parent_list}')

    # for pair in self.ea_list:
    #     if self.model.eligible_to_reproduce(passing_score=self.passing_score, pair=pair):
    #         self.parent_list.append(pair)

  def check_child_mc(self) -> None:
    """
    for the list of new children check the mc criteria is satisfied for each

    :return: None
    """
    child_list = []
    child_bool = [self.mc_satisfied(child_pair[0], child_pair[1]) for child_pair in self.child_list]
    for idx, conditional in enumerate(child_bool):
      if conditional:
        child_list.append(self.child_list[idx])
    del child_bool  # free from memory
    self.child_list = child_list

    # child_list = []
    # # print(f'child_list is: {len(self.child_list)} long')
    # # idx = 0
    # for child_pair in self.child_list:
    #     # idx += 1
    #     # print(f'idx is: {idx}, result: {self.mc_satisfied(child_pair[0], child_pair[1])},'
    #     #       f' length: {len(self.child_list)}')
    #     if self.mc_satisfied(child_pair[0], child_pair[1]):
    #         child_list.append(child_pair)  # keep if mc_satisfied
    # self.child_list = child_list

  def rank_by_novelty(self) -> None:
    """
    Rank the child_list by their novelty compared to all previous trialed policies.

    ...
    The most novel env will score the lowest as all policies previously scored are poorly suited to the new env

    :return: None
    """
    pair_score = []  # initialize pair score with zero
    for idx, child_pair in enumerate(self.child_list):
      reward_eval = 0
      # Eval on archived env policies
      for policy in self.archived_policies:
        reward_eval += self.model.simulate_env(n_episodes=5, env_config=child_pair[0], policy=policy)
      # Eval on current policies
      for ea_pair in self.ea_list:
        reward_eval += self.model.simulate_env(n_episodes=5, env_config=child_pair[0], policy=ea_pair[1])
      # TODO: Add k-means nn based ranking as in POET paper, use L2 norm too, current might be ok though?
      pair_score.append(reward_eval)
    print(sorted(pair_score, reverse=False))
    self.child_list = [x for _, x in sorted(zip(pair_score, self.child_list), reverse=False)]

  def add_to_ea_list(self):
    """
    Add each ea pair to the master ea list and assign it the best policy from all possible parents

    :return: None
    """
    num_admitted = 0
    for pair in self.child_list:  # test the diversity of the child_list distro
      # Investigate transfer of each current policy in ea pair list
      child_policy = evaluate_candidates([x[1] for x in self.ea_list], pair[0], pair[1], self.model)
      # Assign original policy if candidate already had best policy
      if not child_policy:
        child_policy = pair[1]
      if self.mc_satisfied(pair[0], child_policy):
        self.ea_list.append((pair[0], child_policy))
        num_admitted += 1
        if num_admitted >= self.max_admitted:
          break

  def mc_satisfied(self, env, policy) -> bool:
    """
    Check if Minimal Criterion (MC) 'score' is satisfied by investigating performance of each under policy

    :param env: env config
    :param policy: policy network
    :return: boolean, true if criteria satisfied
    """
    reward = self.model.simulate_env(n_episodes=5, env_config=env, policy=policy)
    print(f'low_crit: {self.mc_crit_low}, high_crit: {self.mc_crit_high}, reward: {reward}')
    return self.mc_crit_low < reward < self.mc_crit_high

  def remove_oldest(self, num_removals: int) -> None:
    """
    Remove oldest pairs to limit size of active env buffers

    :param num_removals: the number of elements to remove from the end of the list
    :return: None
    """
    for pair in reversed(self.ea_list):
      if num_removals > 0:
        self.ea_list.remove(pair)
        self.archived_policies.append(pair[1])
      else:
        break