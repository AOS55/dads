from tests.test_bipedal import TestBipedalCustom
from POET.reproducer import Reproducer
from envs.bipedal_walker_custom import Env_config


class TestReproducer:
  def __init__(self):
    env_categories = ['pit', 'stump', 'stair']
    self.max_children = 10
    self.reproducer = Reproducer(42, env_categories, self.max_children)
    self.test_bipedal = TestBipedalCustom()

  def test_group(self, num_children=100):
    parent = Env_config(
      name='init_parent',
      ground_roughness=0,
      pit_gap=[],
      stump_width=[],
      stump_height=[],
      stump_float=[],
      stair_height=[],
      stair_width=[],
      stair_steps=[]
    )
    for idx in range(num_children):
      child = self.reproducer.reproduce(parent)
      self.test_bipedal.env.set_env_config(child)
      self.test_bipedal.test_reset()
      self.test_bipedal.test_step()
      if idx % self.max_children and idx > 0:
        parent = child

  def test_ea_list(self, num_children=100):

    def _remove_oldest(pairs, archived_pairs, num_removals):
      for pair in reversed(pairs):
        if num_removals > 0:
          pairs.remove(pair)
          archived_pairs.append(pair)
          num_removals -= 1
        else:
          break
      return pairs, archived_pairs

    parent = Env_config(
      name='init_parent',
      ground_roughness=0,
      pit_gap=[],
      stump_width=[],
      stump_height=[],
      stump_float=[],
      stair_height=[],
      stair_width=[],
      stair_steps=[]
    )
    ea_list = [parent]
    archived_list = []
    removals = 90
    for idx in range(num_children):
      child = self.reproducer.reproduce(parent)
      if len(list(filter(lambda x: (x.name == child.name), ea_list))) == 0:
        ea_list.append(child)
      if idx % self.max_children and idx > 0:
        parent = child

    ea_list, archived_list = _remove_oldest(ea_list, archived_list, removals)
    assert len(archived_list) == removals


def test_reproducer():
  test_class = TestReproducer()
  test_class.test_group(1000)
  test_class.test_ea_list(1000)
