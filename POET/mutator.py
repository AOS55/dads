

def eligible_to_reproduce(env):
  # TODO: Insert eligibility criteria
  return True


def env_reproduce(parent_list, max_children):
  # TODO: Use reproducer operations to generate a list of children
  return None


def mc_satisfied(child_list):
  # TODO: Implement a method to remove if minimal criterion is no longer satisfied
  return None


def rank_by_novelty(child_list):
  # TODO: Implement a rank by novelty (PATA-EC) may be best here?
  return None


def evaluate_candidates(env_list, target_env):
  # TODO: Find a method to evaluate each agent on the given environment then select appropriate one
  return None


def remove_oldest(env_list, num_removals):
  # TODO: Remove where appropriate (PATA-EC may be used here)
  return None


def mutate_env(env_list, max_children=3, max_admitted=3, capacity=10):
  parent_list = []
  for env in env_list:
    if eligible_to_reproduce(env):
      parent_list.append(env)
  child_list = env_reproduce(parent_list, max_children)
  child_list = mc_satisfied(child_list)
  child_list = rank_by_novelty(child_list)
  admitted = 0
  for child in child_list:
    target_agent = evaluate_candidates(env_list, child)
    if mc_satisfied(target_agent):
      # TODO: add a way to encode appropriate agent into env (will be dir copy)
      env_list.append(child)
      admitted += 1
      if admitted >= max_admitted:
        break
  env_list_size = len(env_list)
  if env_list_size > capacity:
    num_removals = env_list_size - capacity
    env_list = remove_oldest(env_list, num_removals)
  return env_list
