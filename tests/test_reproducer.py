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


def test_reproducer():
  test_class = TestReproducer()
  test_class.test_group(1000)
