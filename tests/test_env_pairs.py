from unsupervised_skill_learning.dads_OOP import EnvPairs, setup_top_dirs, setup_agent_dir
from envs.bipedal_walker_custom import Env_config
from POET.reproducer import Reproducer
import tensorflow as tf
import pyvirtualdisplay
import numpy as np

test_env_config = Env_config(
    name='test_env',
    ground_roughness=0,
    pit_gap=[4, 5],
    stump_width=[],
    stump_height=[],
    stump_float=[],
    stair_height=[],
    stair_width=[],
    stair_steps=[]
  )


def setup_config(model_dir, save_dir, test_env_config):
  test_config = {
      'env_name': 'bipedal_walker_custom',
      'env_config': test_env_config,
      'log_dir': model_dir,
      'num_skills': 2,
      'skill_type': 'cont_uniform',
      'random_skills': 100,
      'min_steps_before_resample': 2000,
      'resample_prob': 0.02,
      'max_env_steps': 200,
      'observation_omit_size': 0,
      'reduced_observation': 0,
      'hidden_layer_size': 512,
      'save_dir': save_dir,
      'skill_dynamics_observation_relabel_type': 'importance_sampling',
      'skill_dynamics_relabel_type': 'importance_sampling',
      'is_clip_eps': 1,
      'normalize_data': 1,
      'graph_type': 'default',
      'num_components': 4,
      'fix_variance': 1,
      'skill_dynamics_lr': 3e-4,
      'agent_lr': 3e-4,
      'agent_gamma': 0.99,
      'agent_entropy': 0.1,
      'debug': 0,
      'collect_policy_type': 'default',
      'replay_buffer_capacity': 100000,
      'train_skill_dynamics_on_policy': 0,
      'initial_collect_steps': 2000,
      'collect_steps': 1000,
      'action_clipping': 1.,
      'num_epochs': 10,
      'save_model': 'dads',
      'save_freq': 2,
      'clear_buffer_every_iter': 0,
      'skill_dynamics_train_steps': 8,
      'skill_dynamics_batch_size': 256,
      'num_samples_for_relabelling': 1,
      'debug_skill_relabelling': 0,
      'agent_train_steps': 64,
      'agent_relabel_type': 'importance_sampling',
      'agent_batch_size': 256,
      'record_freq': 100,
      'vid_name': 'skill',
      'deterministic_eval': 0,
      'num_evals': 3,
      'restore_training': True
    }
  return test_config


class TestEnvPairs:
  def __init__(self, env_config=test_env_config):
    root_dir, log_dir, save_dir = setup_top_dirs('log_dir', 'bipedal_walker_custom')
    log_dir, model_dir, save_dir = setup_agent_dir(log_dir, 'test_env')
    test_config = setup_config(model_dir, save_dir, env_config)
    self.ea_pairs = EnvPairs(test_config, log_dir)
    self.reproducer = Reproducer(42, ['stump', 'stairs'], 5)

  def add_children(self):
    children = self.reproducer.mutate_list(self.ea_pairs.pairs, self.ea_pairs)
    for child in children:
      self.ea_pairs.pairs.append(child)

  def test_update_pata_ec(self):
    self.add_children()
    candidate_env = Env_config(
      name='test_env',
      ground_roughness=0,
      pit_gap=[2, 3],
      stump_width=[],
      stump_height=[],
      stump_float=[],
      stair_height=[],
      stair_width=[],
      stair_steps=[]
    )
    lower_bound = -4000
    upper_bound = +4000
    pata_ec = self.ea_pairs.update_pata_ec(candidate_env, lower_bound, upper_bound)
    assert type(pata_ec) == np.ndarray
    top_k_indices = pata_ec.argsort()[:5]
    top_k = pata_ec[top_k_indices]
    novelty = top_k.mean()
    assert type(novelty) == np.float32


def test_env_pairs():
  pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
  # Setup tf values
  tf.compat.v1.enable_resource_variables()
  tf.compat.v1.disable_eager_execution()

  # Setup logging
  test = TestEnvPairs()
  test.test_update_pata_ec()
