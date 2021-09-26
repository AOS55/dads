from envs.bipedal_walker_custom import BipedalWalkerCustom, Env_config
import numpy as np


class TestBipedalCustom:
  def __init__(self):
    test_env_config = Env_config(
      name='test_env',
      ground_roughness=0,
      pit_gap=[],
      stump_width=[],
      stump_height=[],
      stump_float=[],
      stair_height=[],
      stair_width=[],
      stair_steps=[]
    )
    self.env = BipedalWalkerCustom(test_env_config)

  def test_reset(self):
    init_output = self.env.reset()
    assert type(init_output) == np.ndarray

  def test_step(self):
    action = [0, 0, 0, 0]
    output_step = self.env.step(action)
    assert type(output_step[0]) == np.ndarray
    assert type(output_step[1]) == np.float64
    assert type(output_step[2]) == bool
    assert type(output_step[3]) == dict

  def test_pit_gap(self):
    test_env_config = Env_config(
      name='pit_gap',
      ground_roughness=0,
      pit_gap=[0, 1],
      stump_width=[],
      stump_height=[],
      stump_float=[],
      stair_height=[],
      stair_width=[],
      stair_steps=[]
    )
    self.env.set_env_config(test_env_config)
    self.test_reset()
    self.test_step()

  def test_stumps(self):
    test_env_config = Env_config(
      name='stumps',
      ground_roughness=0,
      pit_gap=[],
      stump_width=[1, 2],
      stump_height=[0, 0.4],
      stump_float=[0, 1],
      stair_height=[],
      stair_width=[],
      stair_steps=[]
    )
    self.env.set_env_config(test_env_config)
    self.test_reset()
    self.test_step()

  def test_stairs(self):
    test_env_config = Env_config(
      name='stairs',
      ground_roughness=0,
      pit_gap=[],
      stump_width=[],
      stump_height=[],
      stump_float=[],
      stair_height=[0, 0.4],
      stair_width=[4, 5],
      stair_steps=[1, 2]
    )
    self.env.set_env_config(test_env_config)
    self.test_reset()
    self.test_step()

  def test_all(self):
    test_env_config = Env_config(
      name='all',
      ground_roughness=0,
      pit_gap=[0, 1],
      stump_width=[1, 2],
      stump_height=[0, 0.4],
      stump_float=[0, 1],
      stair_height=[0, 0.4],
      stair_width=[4, 5],
      stair_steps=[1, 2]
    )
    self.env.set_env_config(test_env_config)
    self.test_reset()
    self.test_step()


def test_bipedal_custom():
  test_class = TestBipedalCustom()
  test_class.test_reset()
  test_class.test_step()
  test_class.test_pit_gap()
  test_class.test_stumps()
  test_class.test_stairs()
  test_class.test_all()
