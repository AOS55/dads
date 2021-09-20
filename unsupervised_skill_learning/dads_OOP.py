from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle as pkl
import os
import io
from absl import logging, flags
import functools

import copy
import sys
sys.path.append(os.path.abspath('./'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.environments import suite_mujoco
from tf_agents.trajectories import time_step as ts
from tf_agents.environments.suite_gym import wrap_env
from tf_agents.trajectories.trajectory import from_transition, to_transition
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.policies import ou_noise_policy
from tf_agents.trajectories import policy_step
# from tf_agents.policies import py_tf_policy
# from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils

from unsupervised_skill_learning import dads_agent

from envs import skill_wrapper
from envs import video_wrapper
from envs.gym_mujoco import ant
from envs.gym_mujoco import half_cheetah
from envs.gym_mujoco import humanoid
from envs.gym_mujoco import point_mass

from envs import dclaw
from envs import dkitty_redesign
from envs import hand_block

from envs import bipedal_walker
from envs import bipedal_walker_custom
from envs.bipedal_walker_custom import Env_config

from POET.mutator import Mutator
from POET.utils import EAPair
from POET.stats import compute_centered_ranks

import pyvirtualdisplay

from lib import py_tf_policy
from lib import py_uniform_replay_buffer

# Global parameters
FLAGS = flags.FLAGS
nest = tf.nest

# general hyperparameters
flags.DEFINE_string('logdir', '~/tmp/dads', 'Directory for saving experiment data')

# environment hyperparameters
flags.DEFINE_string('environment', 'point_mass', 'Name of the environment')
flags.DEFINE_integer('max_env_steps', 200,
                     'Maximum number of steps in one episode')
flags.DEFINE_integer('reduced_observation', 0,
                     'Predict dynamics in a reduced observation space')
flags.DEFINE_integer(
    'min_steps_before_resample', 50,
    'Minimum number of steps to execute before resampling skill')
flags.DEFINE_float('resample_prob', 0.,
                   'Creates stochasticity timesteps before resampling skill')

# need to set save_model and save_freq
flags.DEFINE_string(
    'save_model', None,
    'Name to save the model with, None implies the models are not saved.')
flags.DEFINE_integer('save_freq', 100, 'Saving frequency for checkpoints')
flags.DEFINE_string(
    'vid_name', None,
    'Base name for videos being saved, None implies videos are not recorded')
flags.DEFINE_integer('record_freq', 100,
                     'Video recording frequency within the training loop')

# final evaluation after training is done
flags.DEFINE_integer('run_eval', 0, 'Evaluate learnt skills')

# evaluation type
flags.DEFINE_integer('num_evals', 0, 'Number of skills to evaluate')
flags.DEFINE_integer('deterministic_eval', 0, 'Evaluate all skills, only works for discrete skills')

# training
flags.DEFINE_integer('run_train', 0, 'Train the agent')
flags.DEFINE_integer('num_epochs', 500, 'Number of training epochs')

# skill latent space
flags.DEFINE_integer('num_skills', 2, 'Number of skills to learn')
flags.DEFINE_string('skill_type', 'cont_uniform',
                    'Type of skill and the prior over it')
# network size hyperparameter
flags.DEFINE_integer(
    'hidden_layer_size', 512,
    'Hidden layer size, shared by actors, critics and dynamics')

# reward structure
flags.DEFINE_integer(
    'random_skills', 0,
    'Number of skills to sample randomly for approximating mutual information')

# optimization hyperparameters
flags.DEFINE_integer('replay_buffer_capacity', int(1e6),
                     'Capacity of the replay buffer')
flags.DEFINE_integer(
    'clear_buffer_every_iter', 0,
    'Clear replay buffer every iteration to simulate on-policy training, use larger collect steps and train-steps'
)
flags.DEFINE_integer(
    'initial_collect_steps', 2000,
    'Steps collected initially before training to populate the buffer')
flags.DEFINE_integer('collect_steps', 200, 'Steps collected per agent update')

# relabelling
flags.DEFINE_string('agent_relabel_type', None,
                    'Type of skill relabelling used for agent')
flags.DEFINE_integer(
    'train_skill_dynamics_on_policy', 0,
    'Train skill-dynamics on policy data, while agent train off-policy')
flags.DEFINE_string('skill_dynamics_relabel_type', None,
                    'Type of skill relabelling used for skill-dynamics')
flags.DEFINE_integer(
    'num_samples_for_relabelling', 100,
    'Number of samples from prior for relabelling the current skill when using policy relabelling'
)
flags.DEFINE_float(
    'is_clip_eps', 0.,
    'PPO style clipping epsilon to constrain importance sampling weights to (1-eps, 1+eps)'
)
flags.DEFINE_float(
    'action_clipping', 1.,
    'Clip actions to (-eps, eps) per dimension to avoid difficulties with tanh')
flags.DEFINE_integer('debug_skill_relabelling', 0,
                     'analysis of skill relabelling')

# skill dynamics optimization hyperparamaters
flags.DEFINE_integer('skill_dyn_train_steps', 8,
                     'Number of discriminator train steps on a batch of data')
flags.DEFINE_float('skill_dynamics_lr', 3e-4,
                   'Learning rate for increasing the log-likelihood')
flags.DEFINE_integer('skill_dyn_batch_size', 256,
                     'Batch size for discriminator updates')
# agent optimization hyperparameters
flags.DEFINE_integer('agent_batch_size', 256, 'Batch size for agent updates')
flags.DEFINE_integer('agent_train_steps', 128,
                     'Number of update steps per iteration')
flags.DEFINE_float('agent_lr', 3e-4, 'Learning rate for the agent')

# SAC hyperparameters
flags.DEFINE_float('agent_entropy', 0.1, 'Entropy regularization coefficient')
flags.DEFINE_float('agent_gamma', 0.99, 'Reward discount factor')
flags.DEFINE_string(
    'collect_policy', 'default',
    'Can use the OUNoisePolicy to collect experience for better exploration')

# skill-dynamics hyperparameters
flags.DEFINE_string(
    'graph_type', 'default',
    'process skill input separately for more representational power')
flags.DEFINE_integer('num_components', 4,
                     'Number of components for Mixture of Gaussians')
flags.DEFINE_integer('fix_variance', 1,
                     'Fix the variance of output distribution')
flags.DEFINE_integer('normalize_data', 1, 'Maintain running averages')

# debug
flags.DEFINE_integer('debug', 0, 'Creates extra summaries')

# DKitty
flags.DEFINE_integer('expose_last_action', 1, 'Add the last action to the observation')
flags.DEFINE_integer('expose_upright', 1, 'Add the upright angle to the observation')
flags.DEFINE_float('upright_threshold', 0.9, 'Threshold before which the DKitty episode is terminated')
flags.DEFINE_float('robot_noise_ratio', 0.05, 'Noise ratio for robot joints')
flags.DEFINE_float('root_noise_ratio', 0.002, 'Noise ratio for root position')
flags.DEFINE_float('scale_root_position', 1, 'Multiply the root coordinates the magnify the change')
flags.DEFINE_integer('run_on_hardware', 0, 'Flag for hardware runs')
flags.DEFINE_float('randomize_hfield', 0.0, 'Randomize terrain for better DKitty transfer')
flags.DEFINE_integer('observation_omission_size', 2, 'Dimensions to be omitted from policy input')

# Manipulation Environments
flags.DEFINE_integer('randomized_initial_distribution', 1, 'Fix the initial distribution or not')
flags.DEFINE_float('horizontal_wrist_constraint', 1.0, 'Action space constraint to restrict horizontal motion of the wrist')
flags.DEFINE_float('vertical_wrist_constraint', 1.0, 'Action space constraint to restrict vertical motion of the wrist')

# MPC hyperparameters
flags.DEFINE_integer('planning_horizon', 1, 'Number of primitives to plan in the future')
flags.DEFINE_integer('primitive_horizon', 1, 'Horizon for every primitive')
flags.DEFINE_integer('num_candidate_sequences', 50, 'Number of candidates sequence sampled from the proposal distribution')
flags.DEFINE_integer('refine_steps', 10, 'Number of optimization steps')
flags.DEFINE_float('mppi_gamma', 10.0, 'MPPI weighting hyperparameter')
flags.DEFINE_string('prior_type', 'normal', 'Uniform or Gaussian prior for candidate skill(s)')
flags.DEFINE_float('smoothing_beta', 0.9, 'Smooth candidate skill sequences used')
flags.DEFINE_integer('top_primitives', 5, 'Optimization parameter when using uniform prior (CEM style)')


def get_environment(env_name='point_mass', env_config=None):
  """
  Get the gym environment to use for algorithm itself

  :param env_name: name of the environment to use as a string
  :param env_config: configuration file for the environment, should be an OrderedTuple custom type
  :return: env
  """
  if env_name == 'Ant-v1':
    env = ant.AntEnv(
        expose_all_qpos=True,
        task='motion')
  elif env_name == 'Ant-v1_goal':
    return wrap_env(
        ant.AntEnv(
            task='goal',
            goal=env_config.goal,
            expose_all_qpos=True),
        max_episode_steps=env_config.max_env_steps)
  elif env_name == 'Ant-v1_foot_sensor':
    env = ant.AntEnv(
        expose_all_qpos=True,
        model_path='ant_footsensor.xml',
        expose_foot_sensors=True)
  elif env_name == 'HalfCheetah-v1':
    env = half_cheetah.HalfCheetahEnv(expose_all_qpos=True, task='motion')
  elif env_name == 'Humanoid-v1':
    env = humanoid.HumanoidEnv(expose_all_qpos=True)
  elif env_name == 'point_mass':
    env = point_mass.PointMassEnv(expose_goal=False, expose_velocity=False)
  elif env_name == 'bipedal_walker':
    # pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
    env = bipedal_walker.BipedalWalker()
  elif env_name == 'bipedal_walker_custom':
    if env_config is None:
      env_config = Env_config(
        name='default_env',
        ground_roughness=0,
        pit_gap=[],
        stump_width=[],
        stump_height=[],
        stump_float=[],
        stair_height=[],
        stair_width=[],
        stair_steps=[]
      )
    env = bipedal_walker_custom.BipedalWalkerCustom(env_config)
  else:
    # note this is already wrapped, no need to wrap again
    env = suite_mujoco.load(env_name)
  return env


def setup_top_dirs(root_dir, env_name):
  """
  Setup top level dirs to save model into

  :param root_dir: root directory used for logging passed in flags as logdir!
  :param env_name: environment name used to generate the log directory
  :return: paths to root_dir, log_dir, save_dir
  """
  root_dir = os.path.abspath(os.path.expanduser(root_dir))
  if not tf.io.gfile.exists(root_dir):
    tf.io.gfile.makedirs(root_dir)
  log_dir = os.path.join(root_dir, env_name)

  if not tf.io.gfile.exists(log_dir):
    tf.io.gfile.makedirs(log_dir)
  save_dir = os.path.join(log_dir, 'models')
  if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)
  return root_dir, log_dir, save_dir


def setup_agent_dir(log_dir, env_name):
  """
  If using multiple environment configs in a run create a new dir under the environment log_dir

  :param log_dir: core env_name directory
  :param env_name: name of this environment configuration
  :return: log_dir, model_dir, save_dir
  """
  model_dir = os.path.join(log_dir, env_name)
  if not tf.io.gfile.exists(model_dir):
    tf.io.gfile.makedirs(model_dir)
  save_dir = os.path.join(model_dir, 'models')
  if not tf.io.gfile.exists(save_dir):
    tf.io.gfile.makedirs(save_dir)
  return log_dir, model_dir, save_dir


class EnvPairs:
  def __init__(self, init_config, log_dir):
    self.config = init_config
    self.log_dir = log_dir
    self.pairs = []  # list of active pairs being tracked of type EAPair
    self.archived_pairs = []  # list of pairs that have been archived of type EAPair
    self.initialize_agent()

  def initialize_agent(self):
    """
    Train the initial instant of the agent configuration

    :return: None
    """
    init_agent = self._create_agent(self.config)
    perf = init_agent.train_agent()
    del init_agent
    tf.keras.backend.clear_session()
    init_ea_pair = EAPair(env_name=self.config['env_name'],
                          env_config=self.config['env_config'],
                          agent_config=self.config,
                          agent_score=perf,
                          parent=None,
                          pata_ec=0.5)

    self.pairs.append(copy.deepcopy(init_ea_pair))

  def train_agent(self, pair):
    """
    Train an ea_list agent for a further number of steps

    :param pair: current environment-agent config
    :return: pair with updated performance & model
    """
    log_dir, model_dir, save_dir = setup_agent_dir(self.log_dir, pair.env_config.name)
    self.config['name'] = pair.env_config.name
    self.config['env_config'] = pair.env_config
    self.config['log_dir'] = model_dir
    self.config['save_dir'] = save_dir
    agent = self._create_agent(self.config)
    perf = agent.train_agent()
    del agent
    pair._replace(agent_score=perf)
    return pair

  def train_on_new_env(self, env_config):
    """
    Train a new agent on an existing env config
    :param env_config:
    :return: None
    """
    log_dir, model_dir, save_dir = setup_agent_dir(self.log_dir, env_config.name)
    self.config['name'] = env_config.name
    self.config['env_config'] = env_config
    self.config['log_dir'] = model_dir
    self.config['save_dir'] = save_dir
    agent = self._create_agent(self.config)
    perf = agent.train_agent()
    del agent
    tf.keras.backend.clear_session()
    ea_pair = EAPair(env_name=self.config['env_name'],
                     env_config=self.config['env_config'],
                     agent_config=self.config,
                     agent_score=perf,
                     parent=None,
                     pata_ec=None)
    self.pairs.append(copy.deepcopy(ea_pair))

  def evaluate_agent_on_env(self, log_dir, save_dir, env_config):
    """
    Given a previously trained agent evaluate the performance of a different agent on this env

    :param log_dir: directory of agent log_dir to train agent on
    :param save_dir: directory of agent model_dir to train agent on
    :param env_config:
    :return:
    """
    # self.config['name'] = env_config.name
    self.config['env_config'] = env_config
    self.config['log_dir'] = log_dir  # use the agent model dir to train on
    self.config['save_dir'] = save_dir
    self.config['num_epochs'] = 5  # eval over 5 epochs of training (sample by training on env)
    self.config['record_freq'] = 10  # set record_freq to be higher than num_epochs to evaluate on
    self.config['save_freq'] = 10  # set save_freq to be higher than num_epochs to evaluate on
    self.config['restore_training'] = False
    # TODO: Check this is not overwriting the saved model, should just deploy the model for evaluation
    agent = self._create_agent(self.config)
    perf = agent.train_agent()
    del agent
    tf.keras.backend.clear_session()
    return perf

  def update_pata_ec(self, candidate_env_config, lower_bound, upper_bound):
    """
    Based on the current group of agents calculate the PATA-EC (Performance of All Trained Agents)

    :param candidate_env_config: configuration for candidate environments
    :param lower_bound: lower bound for agent
    :param upper_bound: upper bound for agent
    :return: pata_ec score
    """
    def _cap_score(score, lower, upper):
      if score < lower:
        score = lower
      elif score > upper:
        score = upper
      return score

    raw_scores = []
    for agent in self.pairs:
      score = self.evaluate_agent_on_env(agent.agent_config['log_dir'], agent.agent_config['save_model'],
                                         candidate_env_config)
      raw_scores.append(_cap_score(score[0], lower_bound, upper_bound))
      del agent
    for agent in self.archived_pairs:
      score = self.evaluate_agent_on_env(agent.agent_config['log_dir'], agent.agent_config['save_model'],
                                         candidate_env_config)
      raw_scores.append(_cap_score(score[0], lower_bound, upper_bound))
      del agent
    if len(raw_scores) > 1:
      pata_ec = compute_centered_ranks(np.array(raw_scores))
    else:
      pata_ec = [0.5]
    return pata_ec

  def evaluate_transfer(self, candidate_env_config):
    """
    Given an env_config find which active agent is best suited for the env

    :param candidate_env_config: env_config of the candidate env
    :return: best_agent to use with candidate_env
    """

    best_score = None
    best_agent = None

    for agent in self.pairs:
      score = self.evaluate_agent_on_env(agent.agent_config['log_dir'],
                                         agent.agent_config['save_dir'], candidate_env_config)
      if best_score is None or best_score[0] < score[0]:
        best_agent = copy.deepcopy(agent.agent_config)
        best_score = score
      del agent

    return best_agent, best_score

  def update_ea_pair(self, pair, parent_name, env_config):
    log_dir, model_dir, save_dir = setup_agent_dir(self.log_dir, env_config.name)
    new_pair = EAPair(env_name=env_config.name,
                      env_config=env_config,
                      agent_config=copy.deepcopy(pair.agent_config),  # need to deepcopy to prevent mutating parent
                      agent_score=None,
                      parent=parent_name,
                      pata_ec=0.5)
    new_agent_config = new_pair.agent_config
    new_agent_config['log_dir'] = model_dir
    new_agent_config['save_dir'] = save_dir
    new_agent_config['env_name'] = env_config.name
    new_agent_config['env_config'] = env_config
    new_pair = new_pair._replace(agent_config=new_agent_config)
    return new_pair

  @staticmethod
  def _get_agent_config(current_dads) -> dict:
    """
    Get values used to construct a copy of a dads object

    :param current_dads: dads object
    :return: dictionary of the objects implementation
    """
    full_dict = current_dads.__dict__
    constructor_config = {
      'env_name': full_dict['env_name'],
      'env_config': full_dict['env_config'],
      'log_dir': full_dict['log_dir'],
      'num_skills': full_dict['num_skills'],
      'skill_type': full_dict['skill_type'],
      'random_skills': full_dict['random_skills'],
      'min_steps_before_resample': full_dict['min_steps_before_resample'],
      'resample_prob': full_dict['resample_prob'],
      'max_env_steps': full_dict['max_env_steps'],
      'observation_omit_size': full_dict['observation_omit_size'],
      'reduced_observation': full_dict['reduced_observation'],
      'hidden_layer_size': full_dict['hidden_layer_size'],
      'save_dir': full_dict['save_dir'],
      'skill_dynamics_observation_relabel_type': full_dict['skill_dynamics_observation_relabel_type'],
      'skill_dynamics_relabel_type': full_dict['skill_dynamics_relabel_type'],
      'is_clip_eps': full_dict['is_clip_eps'],
      'normalize_data': full_dict['normalize_data'],
      'graph_type': full_dict['graph_type'],
      'num_components': full_dict['num_components'],
      'fix_variance': full_dict['fix_variance'],
      'skill_dynamics_lr': full_dict['skill_dynamics_lr'],
      'agent_lr': full_dict['agent_lr'],
      'agent_gamma': full_dict['agent_gamma'],
      'agent_entropy': full_dict['agent_entropy'],
      'debug': full_dict['debug'],
      'collect_policy_type': full_dict['collect_policy_type'],
      'replay_buffer_capacity': full_dict['replay_buffer_capacity'],
      'train_skill_dynamics_on_policy': full_dict['train_skill_dynamics_on_policy'],
      'initial_collect_steps': full_dict['initial_collect_steps'],
      'collect_steps': full_dict['collect_steps'],
      'action_clipping': full_dict['action_clipping'],
      'num_epochs': full_dict['num_epochs'],
      'save_model': full_dict['save_model'],
      'save_freq': full_dict['save_freq'],
      'clear_buffer_every_iter': full_dict['clear_buffer_every_iter'],
      'skill_dynamics_train_steps': full_dict['skill_dynamics_train_steps'],
      'skill_dynamics_batch_size': full_dict['skill_dynamics_batch_size'],
      'num_samples_for_relabelling': full_dict['num_samples_for_relabelling'],
      'debug_skill_relabelling': full_dict['debug_skill_relabelling'],
      'agent_train_steps': full_dict['agent_train_steps'],
      'agent_relabel_type': full_dict['agent_relabel_type'],
      'agent_batch_size': full_dict['agent_batch_size'],
      'record_freq': full_dict['record_freq'],
      'vid_name': full_dict['vid_name'],
      'deterministic_eval': full_dict['deterministic_eval'],
      'num_evals': full_dict['num_evals'],
      'restore_training': full_dict['restore_training']
    }
    del full_dict
    return constructor_config

  @staticmethod
  def _create_agent(dads_config):
    """
    Given a DADS config file make an instance of a DADS object

    :param dads_config:
    :return:
    """
    agent = DADS(env_name=dads_config['env_name'],
                 env_config=dads_config['env_config'],
                 log_dir=dads_config['log_dir'],
                 num_skills=dads_config['num_skills'],
                 skill_type=dads_config['skill_type'],
                 random_skills=dads_config['random_skills'],
                 min_steps_before_resample=dads_config['min_steps_before_resample'],
                 resample_prob=dads_config['resample_prob'],
                 max_env_steps=dads_config['max_env_steps'],
                 observation_omit_size=dads_config['observation_omit_size'],
                 reduced_observation=dads_config['reduced_observation'],
                 hidden_layer_size=dads_config['hidden_layer_size'],
                 save_dir=dads_config['save_dir'],
                 skill_dynamics_observation_relabel_type=dads_config['skill_dynamics_observation_relabel_type'],
                 skill_dynamics_relabel_type=dads_config['skill_dynamics_relabel_type'],
                 is_clip_eps=dads_config['is_clip_eps'],
                 normalize_data=dads_config['normalize_data'],
                 graph_type=dads_config['graph_type'],
                 num_components=dads_config['num_components'],
                 fix_variance=dads_config['fix_variance'],
                 skill_dynamics_lr=dads_config['skill_dynamics_lr'],
                 agent_lr=dads_config['agent_lr'],
                 agent_gamma=dads_config['agent_gamma'],
                 agent_entropy=dads_config['agent_entropy'],
                 debug=dads_config['debug'],
                 collect_policy_type=dads_config['collect_policy_type'],
                 replay_buffer_capacity=dads_config['replay_buffer_capacity'],
                 train_skill_dynamics_on_policy=dads_config['train_skill_dynamics_on_policy'],
                 initial_collect_steps=dads_config['initial_collect_steps'],
                 collect_steps=dads_config['collect_steps'],
                 action_clipping=dads_config['action_clipping'],
                 num_epochs=dads_config['num_epochs'],
                 save_model=dads_config['save_model'],
                 save_freq=dads_config['save_freq'],
                 clear_buffer_every_iter=dads_config['clear_buffer_every_iter'],
                 skill_dynamics_train_steps=dads_config['skill_dynamics_train_steps'],
                 skill_dynamics_batch_size=dads_config['skill_dynamics_batch_size'],
                 num_samples_for_relabelling=dads_config['num_samples_for_relabelling'],
                 debug_skill_relabelling=dads_config['debug_skill_relabelling'],
                 agent_train_steps=dads_config['agent_train_steps'],
                 agent_relabel_type=dads_config['agent_relabel_type'],
                 agent_batch_size=dads_config['agent_batch_size'],
                 record_freq=dads_config['record_freq'],
                 vid_name=dads_config['vid_name'],
                 deterministic_eval=dads_config['deterministic_eval'],
                 num_evals=dads_config['num_evals'],
                 restore_training=dads_config['restore_training'])
    return agent


def get_agent_config(current_dads) -> dict:
  """
  Get values used to construct a copy of a dads object

  :param current_dads: dads object
  :return: dictionary of the objects implementation
  """
  full_dict = current_dads.__dict__
  constructor_config = {
    'env_name': full_dict['env_name'],
    'env_config': full_dict['env_config'],
    'log_dir': full_dict['log_dir'],
    'num_skills': full_dict['num_skills'],
    'skill_type': full_dict['skill_type'],
    'random_skills': full_dict['random_skills'],
    'min_steps_before_resample': full_dict['min_steps_before_resample'],
    'resample_prob': full_dict['resample_prob'],
    'max_env_steps': full_dict['max_env_steps'],
    'observation_omit_size': full_dict['observation_omit_size'],
    'reduced_observation': full_dict['reduced_observation'],
    'hidden_layer_size': full_dict['hidden_layer_size'],
    'save_dir': full_dict['save_dir'],
    'skill_dynamics_observation_relabel_type': full_dict['skill_dynamics_observation_relabel_type'],
    'skill_dynamics_relabel_type': full_dict['skill_dynamics_relabel_type'],
    'is_clip_eps': full_dict['is_clip_eps'],
    'normalize_data': full_dict['normalize_data'],
    'graph_type': full_dict['graph_type'],
    'num_components': full_dict['num_components'],
    'fix_variance': full_dict['fix_variance'],
    'skill_dynamics_lr': full_dict['skill_dynamics_lr'],
    'agent_lr': full_dict['agent_lr'],
    'agent_gamma': full_dict['agent_gamma'],
    'agent_entropy': full_dict['agent_entropy'],
    'debug': full_dict['debug'],
    'collect_policy_type': full_dict['collect_policy_type'],
    'replay_buffer_capacity': full_dict['replay_buffer_capacity'],
    'train_skill_dynamics_on_policy': full_dict['train_skill_dynamics_on_policy'],
    'initial_collect_steps': full_dict['initial_collect_steps'],
    'collect_steps': full_dict['collect_steps'],
    'action_clipping': full_dict['action_clipping'],
    'num_epochs': full_dict['num_epochs'],
    'save_model': full_dict['save_model'],
    'save_freq': full_dict['save_freq'],
    'clear_buffer_every_iter': full_dict['clear_buffer_every_iter'],
    'skill_dynamics_train_steps': full_dict['skill_dynamics_train_steps'],
    'skill_dynamics_batch_size': full_dict['skill_dynamics_batch_size'],
    'num_samples_for_relabelling': full_dict['num_samples_for_relabelling'],
    'debug_skill_relabelling': full_dict['debug_skill_relabelling'],
    'agent_train_steps': full_dict['agent_train_steps'],
    'agent_relabel_type': full_dict['agent_relabel_type'],
    'agent_batch_size': full_dict['agent_batch_size'],
    'record_freq': full_dict['record_freq'],
    'vid_name': full_dict['vid_name'],
    'deterministic_eval': full_dict['deterministic_eval'],
    'num_evals': full_dict['num_evals'],
    'restore_training': full_dict['restore_training']
  }
  del full_dict
  return constructor_config


class DADS:
  """Core class to implement a single DADS EA pair object"""
  def __init__(self,
               env_name,
               env_config,
               log_dir,
               num_skills,
               skill_type,
               random_skills,
               min_steps_before_resample,
               resample_prob,
               max_env_steps,
               observation_omit_size,
               reduced_observation,
               hidden_layer_size,
               save_dir,
               skill_dynamics_observation_relabel_type,
               skill_dynamics_relabel_type,
               is_clip_eps,
               normalize_data,
               graph_type,
               num_components,
               fix_variance,
               skill_dynamics_lr,
               agent_lr,
               agent_gamma,
               agent_entropy,
               debug,
               collect_policy_type,
               replay_buffer_capacity,
               train_skill_dynamics_on_policy,
               initial_collect_steps,
               collect_steps,
               action_clipping,
               num_epochs,
               save_model,
               save_freq,
               clear_buffer_every_iter,
               skill_dynamics_train_steps,
               skill_dynamics_batch_size,
               num_samples_for_relabelling,
               debug_skill_relabelling,
               agent_train_steps,
               agent_relabel_type,
               agent_batch_size,
               record_freq,
               vid_name,
               deterministic_eval,
               num_evals,
               restore_training
               ):
    """
    Constructor for DADS procedures

    :param env_name: name of gym environment
    :param env_config: configuration (leave None for non-custom environment)
    :param log_dir: logging directory for *this* object
    :param num_skills: number of skills to learn p(z) ~ Z
    :param skill_type: type of skills encoded by z ~ Z
    :param random_skills: skills to use for prior samples
    :param min_steps_before_resample: minimum steps to take before resampling the environment
    :param resample_prob: likelihood of resampling the environment
    :param max_env_steps: maximum number of steps to take within a given environment
    :param observation_omit_size: whether to omit the size of observations
    :param reduced_observation: whether to reduce the size of the observation
    :param hidden_layer_size: size of networks hidden layers
    :param save_dir: directory used to save the envs checkpoints (should link to log_dir)
    :param skill_dynamics_observation_relabel_type: type of relabelling to apply to skill dynamics observation
    :param skill_dynamics_relabel_type: type of skill relabelling e.g. importance sampling
    :param is_clip_eps: size of eps clip
    :param normalize_data: boolean to normalize_data
    :param graph_type: type of graph used
    :param num_components: number of components
    :param fix_variance: boolean for fixed variance
    :param skill_dynamics_lr: learning rate for skill dynamics
    :param agent_lr: learning rate for agent itself
    :param agent_gamma: gamma (discount factor) for agent
    :param agent_entropy: entropy value for agent
    :param debug: debug level
    :param collect_policy_type: type of collection policy
    :param replay_buffer_capacity: size of replay buffer
    :param train_skill_dynamics_on_policy: skill dynamics training policy
    :param initial_collect_steps: initial collection steps
    :param collect_steps: collection steps
    :param action_clipping: size of action clipping (+ve)
    :param num_epochs: number of epochs to use
    :param save_model: boolean, whether to save model or not
    :param save_freq: frequency at which to save the environment key params
    :param clear_buffer_every_iter: boolean, whether to clear the buffer @ every iteration
    :param skill_dynamics_train_steps: number of steps to use to train each of the skill dynamics
    :param skill_dynamics_batch_size: size of skill dynamics batch
    :param num_samples_for_relabelling: number of samples to use for each samples relabelling
    :param debug_skill_relabelling: boolean, whether to use skill relabelling (currently depreceated)
    :param agent_train_steps: number of steps to use to train the SAC agent
    :param agent_relabel_type: type of relabelling to use (e.g. 'Importance_Sampling')
    :param agent_batch_size: size of the agent batch
    :param record_freq: how often to record env information parameters
    :param vid_name: general name to use for the video
    :param deterministic_eval: boolean, whether to use deterministic evaluation
    :param num_evals: number of evalutions to use for non-deterministic evaluation (pi is a distro NOT fixed outcome)
    """

    # Initialize tensorboard logging
    self.train_summary_writer = tf.compat.v2.summary.create_file_writer(os.path.join(log_dir, 'train', 'in_graph_data'),
                                                                        flush_millis=10 * 1000)
    self.train_summary_writer.set_as_default()

    # Initialize environment
    self.env_name = env_name
    self.env_config = env_config
    self.env = get_environment(env_name, env_config)

    # Initialize a global step type
    self.global_step = tf.compat.v1.train.get_or_create_global_step()  # graph in which to create global step tensor

    # Initialize env parameters
    self.episode_size_buffer = []
    self.episode_return_buffer = []
    self.num_skills = num_skills
    self.skill_type = skill_type
    self.random_skills = random_skills
    self.max_env_steps = max_env_steps
    self.min_steps_before_resample = min_steps_before_resample
    self.resample_prob = resample_prob
    self.py_env = self.wrap_env()

    # Initialize spec attributes
    self.observation_omit_size = observation_omit_size
    self.reduced_observation = reduced_observation
    self.py_action_spec = None
    self.tf_action_spec = None
    self.env_obs_spec = None
    self.py_env_time_step_spec = None
    self.agent_obs_spec = None
    self.py_agent_time_step_spec = None
    self.tf_agent_time_step_spec = None
    self.skill_dynamics_observation_size = None
    self.define_spec()

    # Initialize networks
    self.actor_net = None
    self.critic_net = None
    self.hidden_layer_size = hidden_layer_size
    self.initialize_networks(self._normal_projection_net)

    # Initialize DADS agent
    self.save_dir = save_dir
    # self.skill_dynamics_observation_size = skill_dynamics_observation_size
    # TODO: skill_dynamics_observation_relabel_type and skill_dynamics_relabel_type are the same variable?
    self.skill_dynamics_observation_relabel_type = skill_dynamics_observation_relabel_type
    self.skill_dynamics_relabel_type = skill_dynamics_relabel_type
    self.is_clip_eps = is_clip_eps
    self.normalize_data = normalize_data
    self.graph_type = graph_type
    self.num_components = num_components
    self.fix_variance = fix_variance
    self.skill_dynamics_lr = skill_dynamics_lr
    self.agent_lr = agent_lr
    self.agent_gamma = agent_gamma
    self.agent_entropy = agent_entropy
    # self.reward_scale_factor = reward_scale_factor
    self.debug = debug
    self.agent = self.get_dads_agent()

    # Initialize policies
    self.collect_policy_type = collect_policy_type
    self.eval_policy, self.collect_policy, self.relabel_policy = self.initialize_policies(self.collect_policy_type)

    # Initialize replay buffer
    self.policy_step_spec, self.trajectory_spec = self.define_buffer_spec()
    self.replay_buffer_capacity = replay_buffer_capacity
    self.train_skill_dynamics_on_policy = train_skill_dynamics_on_policy
    self.initial_collect_steps = initial_collect_steps
    self.collect_steps = collect_steps
    self.rbuffer, self.on_buffer = self.initialize_buffer()

    # Initialize agent methods for sess graphs
    self.agent.build_agent_graph()
    self.agent.build_skill_dynamics_graph()
    self.agent.create_savers()

    # Save current setup & start sessions
    self.train_checkpointer, self.policy_checkpointer, self.rb_checkpointer = self.initialize_checkpoints(
      self.global_step)
    self.sess = self.initialize_session()

    # Parameters for training
    self.num_epochs = num_epochs
    self.action_clipping = action_clipping
    self.log_dir = log_dir
    self.save_model = save_model
    self.save_freq = save_freq
    self.restore_training = restore_training
    self.clear_buffer_every_iter = clear_buffer_every_iter
    self.skill_dynamics_train_steps = skill_dynamics_train_steps
    self.skill_dynamics_batch_size = skill_dynamics_batch_size
    self.num_samples_for_relabelling = num_samples_for_relabelling
    self.debug_skill_relabelling = debug_skill_relabelling
    self.agent_train_steps = agent_train_steps
    self.agent_relabel_type = agent_relabel_type
    self.agent_batch_size = agent_batch_size
    self.record_freq = record_freq
    self.vid_name = vid_name
    self.deterministic_eval = deterministic_eval
    self.num_evals = num_evals

  def __del__(self):
    return

  def wrap_env(self):
    """
    Wrap an environment with the skill wrapper

    :return: None
    """
    py_env = wrap_env(
      skill_wrapper.SkillWrapper(
        self.env,
        num_latent_skills=self.num_skills,
        skill_type=self.skill_type,
        preset_skill=None,
        min_steps_before_resample=self.min_steps_before_resample,
        resample_prob=self.resample_prob),
      max_episode_steps=self.max_env_steps
    )
    return py_env

  def define_spec(self):
    """
    Define the spec required for policies and buffers

    :param observation_omit_size: size of observations to omit (i.e. min due noise)
    :param reduced_observation: size of reduced observation, None if unknown
    :param num_skills: number of skills to use to calculate reduced observation size
    :return:
    """
    self.py_action_spec = self.py_env.action_spec()
    self.tf_action_spec = tensor_spec.from_spec(self.py_action_spec)  # policy & critic action spec
    self.env_obs_spec = self.py_env.observation_spec()
    self.py_env_time_step_spec = ts.time_step_spec(self.env_obs_spec)
    if self.observation_omit_size > 0:
      self.agent_obs_spec = array_spec.BoundedArraySpec(
        (self.env_obs_spec.shape[0] - self.observation_omit_size,),
        self.env_obs_spec.dtype,
        minimum=self.env_obs_spec.minimum,
        maximum=self.env_obs_spec.maximum,
        name=self.env_obs_spec.name
      )
    else:
      self.agent_obs_spec = self.env_obs_spec
    self.py_agent_time_step_spec = ts.time_step_spec(self.agent_obs_spec)
    self.tf_agent_time_step_spec = tensor_spec.from_spec(self.py_agent_time_step_spec)
    if not self.reduced_observation:
      self.skill_dynamics_observation_size = (
        self.py_env_time_step_spec.observation.shape[0] - self.num_skills
      )
    else:
      self.skill_dynamics_observation_size = self.reduced_observation

  def initialize_networks(self, projection_unit) -> None:
    """
    Initialize the actor and critic network attributes for to be used with dads policy and SAC algorithm

    :param hidden_layer_size: size of hidden layers in actor & critic
    :param projection_unit: function to generate continuous projection unit
    :return: None
    """

    self.actor_net = actor_distribution_network.ActorDistributionNetwork(
      self.tf_agent_time_step_spec.observation,
      self.tf_action_spec,
      fc_layer_params=(self.hidden_layer_size,) * 2,
      continuous_projection_net=projection_unit
    )

    self.critic_net = critic_network.CriticNetwork(
      (self.tf_agent_time_step_spec.observation, self.tf_action_spec),
      observation_fc_layer_params=None,
      action_fc_layer_params=None,
      joint_fc_layer_params=(self.hidden_layer_size,) * 2,
    )

  def get_dads_agent(self):
    """
    Get a dads agent instance from the DADSAgent class

    :return: agent object
    """
    if self.skill_dynamics_observation_relabel_type is not None and 'importance_sampling' in \
            self.skill_dynamics_relabel_type and self.is_clip_eps > 1.0:
      reweigh_batches = True
    else:
      reweigh_batches = False

    agent = dads_agent.DADSAgent(
      # DADS parameters
      self.save_dir,
      self.skill_dynamics_observation_size,
      observation_modify_fn=self.process_observation,
      restrict_input_size=self.observation_omit_size,
      latent_size=self.num_skills,
      latent_prior=self.skill_type,
      prior_samples=self.random_skills,
      fc_layer_params=(self.hidden_layer_size,) * 2,
      normalize_observations=self.normalize_data,
      network_type=self.graph_type,
      num_mixture_components=self.num_components,
      fix_variance=self.fix_variance,
      reweigh_batches=reweigh_batches,
      skill_dynamics_learning_rate=self.skill_dynamics_lr,
      # SAC parameters
      time_step_spec=self.tf_agent_time_step_spec,
      action_spec=self.tf_action_spec,
      actor_network=self.actor_net,
      critic_network=self.critic_net,
      target_update_tau=0.005,
      target_update_period=1,
      actor_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=self.agent_lr),
      critic_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=self.agent_lr),
      alpha_optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=self.agent_lr),
      td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
      gamma=self.agent_gamma,
      reward_scale_factor=1. / (self.agent_entropy + 1e-12),
      gradient_clipping=None,
      debug_summaries=self.debug,
      train_step_counter=self.global_step
    )
    return agent

  def initialize_policies(self, collect_policy_type):
    """
    Initialize policies to use withe DAD

    :param collect_policy_type: 'default' or 'ou_noise' (Ornstein–Uhlenbeck)
    :return: eval, collection & relabel policies
    """
    # evaluation policy
    eval_policy = py_tf_policy.PyTFPolicy(self.agent.policy)
    # collection policy
    if collect_policy_type == 'default':
      collect_policy = py_tf_policy.PyTFPolicy(self.agent.collect_policy)
    elif collect_policy_type == 'ou_noise':
      collect_policy = py_tf_policy.PyTFPolicy(
        ou_noise_policy.OUNoisePolicy(
          self.agent.collect_policy, ou_stddev=0.2, ou_damping=0.15))
    # relabelling policy deals with batches of data, unlike collect and eval
    relabel_policy = py_tf_policy.PyTFPolicy(self.agent.collect_policy)
    return eval_policy, collect_policy, relabel_policy

  def define_buffer_spec(self):
    """
    Define the specifications for the buffer based on the py_action spec

    :return: policy steps and trajectory spec
    """
    policy_step_spec = policy_step.PolicyStep(action=self.py_action_spec, state=(), info=())
    if self.skill_dynamics_relabel_type is not None and 'importance_sampling' in self.skill_dynamics_relabel_type and\
            self.is_clip_eps > 1.0:
      policy_step_spec = policy_step_spec._replace(
        info=policy_step.set_log_probability(
          policy_step_spec.info,
          array_spec.ArraySpec(
            shape=(), dtype=np.float32, name='action_log_prob'
          )
        )
      )
    trajectory_spec = from_transition(self.py_env_time_step_spec, policy_step_spec, self.py_env_time_step_spec)
    return policy_step_spec, trajectory_spec

  def initialize_buffer(self):
    """
    Initialize the buffer based on the spec for the dads agent

    :return: replay_buffer, on_policy_buffer
    """
    capacity = self.replay_buffer_capacity
    rbuffer = py_uniform_replay_buffer.PyUniformReplayBuffer(capacity=capacity, data_spec=self.trajectory_spec)
    if self.train_skill_dynamics_on_policy:
      on_buffer = py_uniform_replay_buffer.PyUniformReplayBuffer(
        capacity=self.initial_collect_steps + self.collect_steps + 10,
        data_spec=self.trajectory_spec
      )
      return rbuffer, on_buffer
    return rbuffer, None

  def initialize_checkpoints(self, global_step):
    """
    Initialize the checkpointers to save results to (in self.save_dir)

    :param global_step: current global var step
    :return: train, policy & replay_buffer checkpoints
    """
    train_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(self.save_dir, 'agent'),
      agent=self.agent,
      global_step=global_step)
    policy_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(self.save_dir, 'policy'),
      policy=self.agent.policy,
      global_step=global_step)
    rb_checkpointer = common.Checkpointer(
      ckpt_dir=os.path.join(self.save_dir, 'replay_buffer'),
      max_to_keep=1,
      replay_buffer=self.rbuffer)
    return train_checkpointer, policy_checkpointer, rb_checkpointer

  def initialize_session(self):
    """
    Initialize or restore a session

    :return: session object (needs to be closed at end)
    """
    sess = tf.compat.v1.Session()
    self.train_checkpointer.initialize_or_restore(sess)
    self.rb_checkpointer.initialize_or_restore(sess)
    self.agent.set_sessions(initialize_or_restore_skill_dynamics=True, session=sess)
    return sess

  def train_agent(self):
    """
    Train the dads agent for n_loops

    :return:
    """

    # Setup summary writer
    train_writer = tf.compat.v1.summary.FileWriter(os.path.join(self.log_dir, 'train'), self.sess.graph)
    common.initialize_uninitialized_variables(self.sess)
    self.sess.run(self.train_summary_writer.init())

    time_step = self.py_env.reset()
    iter_count = 0
    sample_count = 0
    self.episode_size_buffer.append(0)
    self.episode_return_buffer.append(0.)
    if self.restore_training:
      try:
        sample_count = np.load(os.path.join(self.log_dir, 'sample_count.npy')).tolist()
        iter_count = np.load(os.path.join(self.log_dir, 'iter_count.npy')).tolist()
        self.episode_size_buffer = np.load(os.path.join(self.log_dir, 'episode_size_buffer.npy')).tolist()
        self.episode_return_buffer = np.load(os.path.join(self.log_dir, 'episode_return_buffer.npy')).tolist()
      except:
        pass

    def _process_episode_data(ep_buffer, cur_data):
      """
      Process episode dat and only keep the last 100 data points of the buffer

      :param ep_buffer: current episode buffer
      :param cur_data: new data collected
      :return: buffer updated with new data
      """
      ep_buffer[-1] += cur_data[0]
      ep_buffer += cur_data[1:]

      # Only keep the last 100 episodes
      if len(ep_buffer) > 101:
        ep_buffer = ep_buffer[-101:]
      return ep_buffer

    def _filter_trajectories(trajectory):
      """
      Remove invalid transactions in the buffer that might not have been consecutive in the episode

      :param trajectory: trajectory to filter out
      :return: nested map structure
      """
      valid_indices = (trajectory.step_type[:, 0] != 2)
      return nest.map_structure(lambda x: x[valid_indices], trajectory)

    if iter_count == 0:
      with self.sess.as_default():
        time_step, collect_info = self.collect_experience(time_step,
                                                          buffer_list=[self.rbuffer] if not
                                                          self.train_skill_dynamics_on_policy
                                                          else [self.rbuffer, self.on_buffer],
                                                          num_steps=self.initial_collect_steps)
      self.episode_size_buffer = _process_episode_data(self.episode_size_buffer, collect_info['episode_sizes'])
      self.episode_return_buffer = _process_episode_data(self.episode_return_buffer, collect_info['episode_return'])
      sample_count += self.initial_collect_steps

    while iter_count < self.num_epochs:

      if self.save_model is not None and iter_count % self.save_freq == 0:
        with self.sess.as_default():
          self.train_checkpointer.save(global_step=iter_count)
          self.policy_checkpointer.save(global_step=iter_count)
          self.rb_checkpointer.save(global_step=iter_count)
          self.agent.save_variables(global_step=iter_count)
        # Save numpy binaries
        np.save(os.path.join(self.log_dir, 'sample_count'), sample_count)
        np.save(os.path.join(self.log_dir, 'episode_size_buffer'), self.episode_size_buffer)
        np.save(os.path.join(self.log_dir, 'episode_return_path'), self.episode_return_buffer)
        np.save(os.path.join(self.log_dir, 'iter_count'), iter_count)

      with self.sess.as_default():
        time_step, collect_info = self.collect_experience(time_step,
                                                          buffer_list=[self.rbuffer] if not
                                                          self.train_skill_dynamics_on_policy
                                                          else [self.rbuffer, self.on_buffer],
                                                          num_steps=self.collect_steps)
      sample_count += self.collect_steps
      self.episode_size_buffer = _process_episode_data(self.episode_size_buffer, collect_info['episode_sizes'])
      self.episode_return_buffer = _process_episode_data(self.episode_return_buffer, collect_info['episode_sizes'])

      skill_dynamics_buffer = self.rbuffer
      if self.train_skill_dynamics_on_policy:
        skill_dynamics_buffer = self.on_buffer

      for _ in range(1 if self.clear_buffer_every_iter else self.skill_dynamics_train_steps):
        if self.clear_buffer_every_iter:
          trajectory_sample = self.rbuffer.gather_all_transitions()
        else:
          trajectory_sample = skill_dynamics_buffer.get_next(
            sample_batch_size=self.skill_dynamics_batch_size, num_steps=2)
        trajectory_sample = _filter_trajectories(trajectory_sample)

        trajectory_sample, is_weights = self.relabel_skill(
          trajectory_sample,
          relabel_type=self.skill_dynamics_relabel_type,
          cur_policy=self.relabel_policy,
          cur_skill_dynamics=self.agent.skill_dynamics
        )
        input_obs = self.process_observation(trajectory_sample.observation[:, 0, :-self.num_skills])
        cur_skill = trajectory_sample.observation[:, 0, -self.num_skills:]
        target_obs = self.process_observation(trajectory_sample.observation[:, 1, :-self.num_skills])
        if self.clear_buffer_every_iter:
          self.agent.skill_dynamics.train(
            input_obs,
            cur_skill,
            target_obs,
            batch_size=self.skill_dynamics_batch_size,
            batch_weights=is_weights,
            num_steps=self.skill_dynamics_train_steps
          )
        else:
          self.agent.skill_dynamics.train(
            input_obs,
            cur_skill,
            target_obs,
            batch_size=-1,
            batch_weights=is_weights,
            num_steps=1
          )

      if self.train_skill_dynamics_on_policy:
        self.on_buffer.clear()

      running_dads_reward, running_logp, running_logp_altz = [], [], []

      for _ in range(1 if self.clear_buffer_every_iter else self.agent_train_steps):

        if self.clear_buffer_every_iter:
          trajectory_sample = self.rbuffer.gather_all_transitions()
        else:
          trajectory_sample = self.rbuffer.get_next(sample_batch_size=self.agent_batch_size, num_steps=2)

        trajectory_sample = _filter_trajectories(trajectory_sample)
        trajectory_sample, is_weights = self.relabel_skill(
          trajectory_sample,
          relabel_type=self.agent_relabel_type,
          cur_policy=self.relabel_policy,
          cur_skill_dynamics=self.agent.skill_dynamics
        )

        if self.skill_dynamics_relabel_type is not None and 'importance_sampling' in self.skill_dynamics_relabel_type:
          trajectory_sample = trajectory_sample._replace(policy_info=())

        if not self.clear_buffer_every_iter:
          dads_reward, info = self.agent.train_loop(
            trajectory_sample,
            recompute_reward=True,
            batch_size=-1,
            num_steps=1
          )
        else:
          dads_reward, info = self.agent.train_loop(
            trajectory_sample,
            recompute_reward=True,
            batch_size=self.agent_batch_size,
            num_steps=self.agent_train_steps
          )

        if dads_reward is not None:
          running_dads_reward.append(dads_reward)
          running_logp.append(info['logp'])
          running_logp_altz.append(info['logp_altz'])

      if len(self.episode_size_buffer) > 1:
        train_writer.add_summary(
          tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
              tag='episode_size',
              simple_value=np.mean(self.episode_size_buffer[:-1]))
          ]), sample_count)
      if len(self.episode_return_buffer) > 1:
        train_writer.add_summary(
          tf.compat.v1.Summary(value=[
            tf.compat.v1.Summary.Value(
              tag='episode_return',
              simple_value=np.mean(self.episode_return_buffer[:-1]))
          ]), sample_count)
      train_writer.add_summary(
        tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
            tag='dads/reward',
            simple_value=np.mean(
              np.concatenate(running_dads_reward)))
        ]), sample_count)

      train_writer.add_summary(
        tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
            tag='dads/logp',
            simple_value=np.mean(np.concatenate(running_logp)))
        ]), sample_count)
      train_writer.add_summary(
        tf.compat.v1.Summary(value=[
          tf.compat.v1.Summary.Value(
            tag='dads/logp_altz',
            simple_value=np.mean(np.concatenate(running_logp_altz)))
        ]), sample_count)

      if self.clear_buffer_every_iter:
        self.rbuffer.clear()
        time_step = self.py_env.reset()
        self.episode_size_buffer = [0]
        self.episode_return_buffer = [0.]

      if self.record_freq is not None and iter_count % self.record_freq == 0:
        cur_vid_dir = os.path.join(self.log_dir, 'videos', str(iter_count))
        tf.io.gfile.makedirs(cur_vid_dir)
        self.eval_loop(
          cur_vid_dir,
          self.eval_policy,
          dynamics=self.agent.skill_dynamics,
          vid_name=self.vid_name,
          plot_name='traj_plot'
        )

      iter_count += 1
      print(f'iter_count is: {iter_count}')

    return np.mean(np.concatenate(running_dads_reward)),\
           np.mean(np.concatenate(running_logp)),\
           np.mean(np.concatenate(running_logp_altz))

  def collect_experience(self, time_step, buffer_list, num_steps):
    """
    Collect episode size and reward over the buffer with a given number of steps

    :param time_step: current time_step
    :param buffer_list: list of buffers
    :param num_steps: number of steps to rollout for experience
    :return: updated time step, dict w/episode_sizes and episode_rewards
    """
    episode_sizes = []
    extrinsic_reward = []
    cur_return = 0.
    step_idx = 0
    for idx in range(num_steps):
      if time_step.is_last():
        episode_sizes.append(step_idx)
        extrinsic_reward.append(cur_return)
        cur_return = 0.

      action_step = self.collect_policy.action(self.hide_coords(time_step))

      if self.action_clipping < 1.0:
        action_step = action_step._replace(action=np.clip(action_step, -self.action_clipping, self.action_clipping))
      if self.skill_dynamics_relabel_type is not None and 'importance_sampling' in self.skill_dynamics_relabel_type and\
        self.is_clip_eps > 1.0:
          cur_action_log_prob = self.collect_policy.log_prob(
            nest_utils.batch_nested_array(self.hide_coords(time_step)),
            np.expand_dims(action_step.action, 0)
          )
          action_step = action_step._replace(
            info=policy_step.set_log_probability(action_step.info, cur_action_log_prob)
          )
      if np.isnan(action_step.action).any():
        print(f'action_step.action has a NaN problem: {action_step.action}')
      # print(f'action_step.action: {action_step.action}')
      next_time_step = self.py_env.step(action_step.action)
      cur_return += next_time_step.reward

      for buffer_ in buffer_list:
        buffer_.add_batch(
          from_transition(
            nest_utils.batch_nested_array(time_step),
            nest_utils.batch_nested_array(action_step),
            nest_utils.batch_nested_array(next_time_step)
          )
        )
      time_step = next_time_step
      step_idx += 1

    # Carry over calculation for next collection cycle
    episode_sizes.append(step_idx + 1)
    extrinsic_reward.append(cur_return)
    for idx in range(1, len(episode_sizes)):
      episode_sizes[-idx] -= episode_sizes[-idx - 1]

    return time_step, {
      'episode_sizes': episode_sizes,
      'episode_return': extrinsic_reward
    }

  def relabel_skill(self, trajectory_sample, relabel_type=None, cur_policy=None, cur_skill_dynamics=None):
    """
    Relabel skills based on their posterior

    :param trajectory_sample: sample of trajectories
    :param relabel_type: type of method used to relabel e.g. 'importance_sampling'
    :param cur_policy: current policy used to relabel_skills
    :param cur_skill_dynamics: current agent skill dynamics
    :return: a sample of the new trajectory
    """
    if relabel_type is None or ('importance_sampling' in relabel_type and self.is_clip_eps <= 1.0):
      return trajectory_sample, None

    next_trajectory = nest.map_structure(lambda x: x[:, 1:], trajectory_sample)
    trajectory = nest.map_structure(lambda x: x[:, :-1], trajectory_sample)
    action_steps = policy_step.PolicyStep(
      action=trajectory.action, state=(), info=trajectory.policy_info)
    time_steps = ts.TimeStep(
      trajectory.step_type,
      reward=nest.map_structure(np.zeros_like, trajectory.reward),  # unknown
      discount=np.zeros_like(trajectory.discount),  # unknown
      observation=trajectory.observation)
    next_time_steps = ts.TimeStep(
      step_type=trajectory.next_step_type,
      reward=trajectory.reward,
      discount=trajectory.discount,
      observation=next_trajectory.observation)
    time_steps, action_steps, next_time_steps = nest.map_structure(
      lambda t: np.squeeze(t, axis=1),
      (time_steps, action_steps, next_time_steps))

    # just return the importance sampling weights for the given batch
    if 'importance_sampling' in relabel_type:
      old_log_probs = policy_step.get_log_probability(action_steps.info)
      is_weights = []
      for idx in range(time_steps.observation.shape[0]):
        cur_time_step = nest.map_structure(lambda x: x[idx:idx + 1], time_steps)
        cur_time_step = cur_time_step._replace(
          observation=cur_time_step.observation[:, self.observation_omit_size:])
        old_log_prob = old_log_probs[idx]
        cur_log_prob = cur_policy.log_prob(cur_time_step,
                                           action_steps.action[idx:idx + 1])[0]
        is_weights.append(
          np.clip(
            np.exp(cur_log_prob - old_log_prob), 1. / self.is_clip_eps,
            self.is_clip_eps))

      is_weights = np.array(is_weights)
      if relabel_type == 'normalized_importance_sampling':
        is_weights = is_weights / is_weights.mean()

      return trajectory_sample, is_weights

    new_observation = np.zeros(time_steps.observation.shape)
    for idx in range(time_steps.observation.shape[0]):
      alt_time_steps = nest.map_structure(
        lambda t: np.stack([t[idx]] * self.num_samples_for_relabelling),
        time_steps)

      # sample possible skills for relabelling from the prior
      if self.skill_type == 'cont_uniform':
        # always ensure that the original skill is one of the possible option for relabelling skills
        alt_skills = np.concatenate([
          np.random.uniform(
            low=-1.0,
            high=1.0,
            size=(self.num_samples_for_relabelling - 1, self.num_skills)),
          alt_time_steps.observation[:1, -self.num_skills:]
        ])

      # choose the skill which gives the highest log-probability to the current action
      if relabel_type == 'policy':
        cur_action = np.stack([action_steps.action[idx, :]] *
                              self.num_samples_for_relabelling)
        alt_time_steps = alt_time_steps._replace(
          observation=np.concatenate([
            alt_time_steps.observation[:, self.observation_omit_size:-self.num_skills], alt_skills
          ],
            axis=1))
        action_log_probs = cur_policy.log_prob(alt_time_steps, cur_action)
        if self.debug_skill_relabelling:
          print('\n action_log_probs analysis----', idx,
                time_steps.observation[idx, -self.num_skills:])
          print('number of skills with higher log-probs:',
                np.sum(action_log_probs >= action_log_probs[-1]))
          print('Skills with log-probs higher than actual skill:')
          skill_dist = []
          for skill_idx in range(self.num_samples_for_relabelling):
            if action_log_probs[skill_idx] >= action_log_probs[-1]:
              print(alt_skills[skill_idx])
              skill_dist.append(
                np.linalg.norm(alt_skills[skill_idx] - alt_skills[-1]))
          print('average distance of skills with higher-log-prob:',
                np.mean(skill_dist))
        max_skill_idx = np.argmax(action_log_probs)

      # choose the skill which gets the highest log-probability under the dynamics posterior
      elif relabel_type == 'dynamics_posterior':
        cur_observations = alt_time_steps.observation[:, :-self.num_skills]
        next_observations = np.stack(
          [next_time_steps.observation[idx, :-self.num_skills]] *
          self.num_samples_for_relabelling)

        # max over posterior log probability is exactly the max over log-prob of transitin under skill-dynamics
        posterior_log_probs = cur_skill_dynamics.get_log_prob(
          self.process_observation(cur_observations), alt_skills,
          self.process_observation(next_observations))
        if self.debug_skill_relabelling:
          print('\n dynamics_log_probs analysis----', idx,
                time_steps.observation[idx, -self.num_skills:])
          print('number of skills with higher log-probs:',
                np.sum(posterior_log_probs >= posterior_log_probs[-1]))
          print('Skills with log-probs higher than actual skill:')
          skill_dist = []
          for skill_idx in range(self.num_samples_for_relabelling):
            if posterior_log_probs[skill_idx] >= posterior_log_probs[-1]:
              print(alt_skills[skill_idx])
              skill_dist.append(
                np.linalg.norm(alt_skills[skill_idx] - alt_skills[-1]))
          print('average distance of skills with higher-log-prob:',
                np.mean(skill_dist))

        max_skill_idx = np.argmax(posterior_log_probs)

      # make the new observation with the relabelled skill
      relabelled_skill = alt_skills[max_skill_idx]
      new_observation[idx] = np.concatenate(
        [time_steps.observation[idx, :-self.num_skills], relabelled_skill])

    traj_observation = np.copy(trajectory_sample.observation)
    traj_observation[:, 0] = new_observation
    new_trajectory_sample = trajectory_sample._replace(
      observation=traj_observation)

    return new_trajectory_sample, None

  def eval_loop(self, eval_dir, eval_policy, dynamics, vid_name, plot_name):
    """
    Evaluate the trajectories via rollout on a given environment with the eval_policy and make plots

    :param eval_dir: evaluation directory
    :param cur_vid_dir: current video directory
    :param eval_policy: evaluation policy
    :param dynamics: dynamics from skills
    :param vid_name: name of video file root
    :param plot_name: name of plot to produce
    :return: None
    """
    metadata = tf.io.gfile.GFile(os.path.join(eval_dir, 'metadata.txt'), 'a')
    if self.num_skills == 0:
      num_evals = self.num_evals
    elif self.deterministic_eval:
      num_evals = self.num_skills
    else:
      num_evals = self.num_evals

    if plot_name is not None:
      color_map = ['b', 'g', 'r', 'c', 'm', 'y']
      style_map = []
      for line_style in ['-', '--', '-.', ':']:
        style_map += [color + line_style for color in color_map]

      plt.xlim(-15, 15)
      plt.ylim(-15, 15)

    for idx in range(num_evals):
      if self.num_skills > 0:
        if self.deterministic_eval:
          preset_skill = np.zeros(self.num_skills, dtype=np.int64)
          preset_skill[idx] = 1
        elif self.skill_type == 'discrete_uniform':
          preset_skill = np.random.multinomial(1, [1. / self.num_skills] * self.num_skills)
        elif self.skill_type == 'gaussian':
          preset_skill = np.random.multivariate_normal(
            np.zeros(self.num_skills), np.eye(self.num_skills))
        elif self.skill_type == 'cont_uniform':
          preset_skill = np.random.uniform(
            low=-1.0, high=1.0, size=self.num_skills)
        elif self.skill_type == 'multivariate_bernoulli':
          preset_skill = np.random.binomial(1, 0.5, size=self.num_skills)
      else:
        preset_skill = None

      eval_env = get_environment(env_name=self.env_name, env_config=self.env_config)
      eval_env = wrap_env(
        skill_wrapper.SkillWrapper(
          eval_env,
          num_latent_skills=self.num_skills,
          skill_type=self.skill_type,
          preset_skill=preset_skill,
          min_steps_before_resample=self.min_steps_before_resample,
          resample_prob=self.resample_prob),
        max_episode_steps=self.max_env_steps
        )

      if vid_name is not None:
        full_vid_name = vid_name + '_' + str(idx)
        eval_env = video_wrapper.VideoWrapper(eval_env, base_path=eval_dir, base_name=full_vid_name)

      mean_reward = 0.
      per_skill_evaluations = 1
      predict_trajectory_steps = 0

      with self.sess.as_default():
        for eval_idx in range(per_skill_evaluations):
          eval_trajectory = self.run_on_env(
            eval_env,
            eval_policy,
            dynamics=dynamics,
            predict_trajectory_steps=predict_trajectory_steps,
            return_data=True,
            close_environment=True if eval_idx == per_skill_evaluations - 1 else False)

        trajectory_coordinates = np.array([
          eval_trajectory[step_idx][0][:2]
          for step_idx in range(len(eval_trajectory))
        ])

        if plot_name is not None:
          plt.plot(
            trajectory_coordinates[:, 0],
            trajectory_coordinates[:, 1],
            style_map[idx % len(style_map)],
            label=(str(idx) if eval_idx == 0 else None))
          if predict_trajectory_steps > 0:
            for step_idx in range(len(eval_trajectory)):
              if step_idx % 20 == 0:
                plt.plot(eval_trajectory[step_idx][-1][:, 0],
                         eval_trajectory[step_idx][-1][:, 1], 'k:')

          mean_reward += np.mean([
            eval_trajectory[step_idx][-1]
            for step_idx in range(len(eval_trajectory))
          ])
          metadata.write(
            str(idx) + ' ' + str(preset_skill) + ' ' +
            str(trajectory_coordinates[-1, :]) + '\n')

    if plot_name is not None:
      full_image_name = plot_name + '.png'

      # to save images while writing to CNS
      buf = io.BytesIO()
      # plt.title('Trajectories in Continuous Skill Space')
      plt.savefig(buf, dpi=600, bbox_inches='tight')
      buf.seek(0)
      image = tf.io.gfile.GFile(os.path.join(eval_dir, full_image_name), 'w')
      image.write(buf.read(-1))

      # clear before next plot
      plt.clf()

  def run_on_env(self, env, policy, dynamics=None, predict_trajectory_steps=0,
                 return_data=False, close_environment=True):
    """
    Single run on environment to get extrinsic reward or data

    :param env: environment to use
    :param policy: policy to follow in environment
    :param dynamics: dynamics from skill to follow
    :param predict_trajectory_steps: number of steps to predict the trajectory for
    :param return_data: boolean, to return data
    :param close_environment: boolean, set true to close environment at end
    :return: data or extrinsic reward
    """
    time_step = env.reset()
    data = []

    if not return_data:
      extrinsic_reward = []
    # with self.sess.as_default():
    while not time_step.is_last():
      action_step = policy.action(self.hide_coords(time_step))
      if self.action_clipping < 1.:
        action_step = action_step._replace(
          action=np.clip(action_step.action, -self.action_clipping,
                         self.action_clipping))

      env_action = action_step.action
      next_time_step = env.step(env_action)

      skill_size = self.num_skills
      if skill_size > 0:
        cur_observation = time_step.observation[:-skill_size]
        cur_skill = time_step.observation[-skill_size:]
        next_observation = next_time_step.observation[:-skill_size]
      else:
        cur_observation = time_step.observation
        next_observation = next_time_step.observation

      if dynamics is not None:
        if self.reduced_observation:
          cur_observation, next_observation = self.process_observation(
            cur_observation), self.process_observation(next_observation)
        logp = dynamics.get_log_prob(
          np.expand_dims(cur_observation, 0), np.expand_dims(cur_skill, 0),
          np.expand_dims(next_observation, 0))

        cur_predicted_state = np.expand_dims(cur_observation, 0)
        skill_expanded = np.expand_dims(cur_skill, 0)
        cur_predicted_trajectory = [cur_predicted_state[0]]
        for _ in range(predict_trajectory_steps):
          next_predicted_state = dynamics.predict_state(cur_predicted_state,
                                                        skill_expanded)
          cur_predicted_trajectory.append(next_predicted_state[0])
          cur_predicted_state = next_predicted_state
      else:
        logp = ()
        cur_predicted_trajectory = []
      time_step = next_time_step

      if return_data:
        data.append([
          cur_observation, action_step.action, logp, next_time_step.reward,
          np.array(cur_predicted_trajectory)
        ])
      else:
        extrinsic_reward.append([next_time_step.reward])

    if close_environment:
      env.close()

    if return_data:
      return data
    else:
      return extrinsic_reward

  def update_env_config(self, env_config):
    """
    Update the environment with a new config file

    :param env_config:
    :return:
    """
    # TODO: test this and look at how to allow dir change based on name
    # self.sess.close()  # close current session
    self.env_config = env_config
    # self.__init__()
    # self.py_env = self.wrap_env()
    # self.define_spec()
    # self.initialize_networks(self._normal_projection_net)
    # self.agent = self.get_dads_agent()
    # self.initialize_networks(self._normal_projection_net)
    # self.eval_policy, self.collect_policy, self.relabel_policy = self.initialize_policies(self.collect_policy_type)
    # self.policy_step_spec, self.trajectory_spec = self.define_buffer_spec()
    # self.rbuffer, self.on_buffer = self.initialize_buffer()
    # self.agent.build_agent_graph()
    # self.agent.build_skill_dynamics_graph()
    # self.agent.create_savers()
    # self.train_checkpointer, self.policy_checkpointer, self.rb_checkpointer = self.initialize_checkpoints(
    #   self.global_step)
    # self.sess = self.initialize_session()

  def eval_agent(self):
    """
    Evaluate the dads agent using the appropriate policy

    :return: performance of the agent on a given environment
    """
    # TODO: Add method to allow single eval -> Harder than it looks

    # time_step = self.py_env.reset()
    # iter_count = 0
    # sample_count = 0
    # self.episode_size_buffer.append(0)
    # self.episode_return_buffer.append(0.)
    #
    # def _process_episode_data(ep_buffer, cur_data):
    #   """
    #   Process episode dat and only keep the last 100 data points of the buffer
    #
    #   :param ep_buffer: current episode buffer
    #   :param cur_data: new data collected
    #   :return: buffer updated with new data
    #   """
    #   ep_buffer[-1] += cur_data[0]
    #   ep_buffer += cur_data[1:]
    #
    #   # Only keep the last 100 episodes
    #   if len(ep_buffer) > 101:
    #     ep_buffer = ep_buffer[-101:]
    #   return ep_buffer
    #
    # def _filter_trajectories(trajectory):
    #   """
    #   Remove invalid transactions in the buffer that might not have been consecutive in the episode
    #
    #   :param trajectory: trajectory to filter out
    #   :return: nested map structure
    #   """
    #   valid_indices = (trajectory.step_type[:, 0] != 2)
    #   return nest.map_structure(lambda x: x[valid_indices], trajectory)
    #
    # input_obs = trajectories.observation[:, 0, :-self._latent_size]
    # cur_skill = trajectories.observation[:, 0, :-self._latent_size:]
    # target_obs = trajectories.observation[:, 1, :-self._latent_size]

    return

  @staticmethod
  def _normal_projection_net(action_spec, init_means_output_factor=0.1):
    """
    Function to be passed into the actor net to generate the continuous projection net

    :param action_spec:
    :param init_means_output_factor:
    :return: None
    """
    return normal_projection_network.NormalProjectionNetwork(
      action_spec,
      mean_transform=None,
      state_dependent_std=True,
      init_means_output_factor=init_means_output_factor,
      std_transform=sac_agent.std_clip_transform,
      scale_distribution=True
    )

  def process_observation(self, observation):
    """
    Method for DADSAgent to process to give appropriate state for observation of network

    :param observation: initial raw collected observation from env
    :return: input observation for DADS
    """

    def _shape_based_observation_processing(observation, dim_idx):
      if len(observation.shape) == 1:
        return observation[dim_idx:dim_idx + 1]
      elif len(observation.shape) == 2:
        return observation[:, dim_idx:dim_idx + 1]
      elif len(observation.shape) == 3:
        return observation[:, :, dim_idx:dim_idx + 1]

    # for consistent use (no reduced observation)
    if self.reduced_observation == 0:
      return observation

    if self.env_name == 'HalfCheetah-v1':
      qpos_dim = 9
    elif self.env_name == 'Ant-v1':
      qpos_dim = 15
    elif self.env_name == 'Humanoid_v1':
      qpos_dim = 26
    elif 'DKitty' in self.env_name:
      qpos_dim = 36

    # x-axis
    if self.reduced_observation in [1, 5]:
      red_obs = [_shape_based_observation_processing(observation, 0)]
    # x-y plane
    elif self.reduced_observation in [2, 6]:
      if self.env_name == 'Ant-v1' or 'DKitty' in self.env_name or 'DClaw' in self.env_name:
        red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1)
        ]
      else:
        red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, qpos_dim)
        ]
      # x-y plane, x-y velocities
    elif self.reduced_observation in [4, 8]:
      if self.reduced_observation == 4 and 'DKittyPush' in self.env_name:
        # position of the agent + relative position of the box
        red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation, 3),
          _shape_based_observation_processing(observation, 4)
        ]
      elif self.env_name in ['Ant-v1']:
        red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation, qpos_dim),
          _shape_based_observation_processing(observation, qpos_dim + 1)
        ]
    # (x, y, orientation) only works for ant & point_mass
    elif self.reduced_observation == 3:
      if self.env_name in ['Ant-v1', 'point_mass']:
        red_obs = [
          _shape_based_observation_processing(observation, 0),
          _shape_based_observation_processing(observation, 1),
          _shape_based_observation_processing(observation, observation.shape[1] - 1)
        ]
      # x, y, z of the center of the block
      elif self.env in ['HandBlock']:
        red_obs = [
          _shape_based_observation_processing(observation,
                                              observation.shape[-1] - 7),
          _shape_based_observation_processing(observation,
                                              observation.shape[-1] - 6),
          _shape_based_observation_processing(observation,
                                              observation.shape[-1] - 5)
        ]

    if self.reduced_observation in [5, 6, 8]:
      red_obs += [
        _shape_based_observation_processing(observation,
                                            observation.shape[1] - idx)
        for idx in range(1, 5)
      ]

    if self.reduced_observation == 36 and 'DKitty' in self.env_name:
      red_obs = [
        _shape_based_observation_processing(observation, idx)
        for idx in range(qpos_dim)
      ]

    # x, y, z and the rotation quaternion
    if self.reduced_observation == 7 and self.env_name == 'HandBlock':
      red_obs = [
                  _shape_based_observation_processing(observation, observation.shape[-1] - idx)
                  for idx in range(1, 8)
                ][::-1]

    # the rotation quaternion
    if self.reduced_observation == 4 and self.env_name == 'HandBlock':
      red_obs = [
                  _shape_based_observation_processing(observation, observation.shape[-1] - idx)
                  for idx in range(1, 5)
                ][::-1]

    if isinstance(observation, np.ndarray):
      input_obs = np.concatenate(red_obs, axis=len(observation.shape) - 1)
    elif isinstance(observation, tf.Tensor):
      input_obs = tf.concat(red_obs, axis=len(observation.shape) - 1)
    return input_obs

  def hide_coords(self, time_step):
    """
    Hide the coordinates of an observation if omit size selected appropriately

    :param time_step:
    :return: time_step
    """
    if self.observation_omit_size > 0:
      sans_coords = time_step.observation[self.observation_omit_size:]
      return time_step._replace(observation=sans_coords)
    return time_step


class POET:
  """Paired Open-Ended Trailblazer"""
  def __init__(self,
               init_agent_config,
               log_dir,
               poet_config,
               mutator_config,
               reproducer_config):
    """
    POET constructor

    :param init_agent_config: initial configuration file for the agent
    :param log_dir: log_directory to use for POET responses (forms part of the algorithm training itself)
    :param poet_config: poet configuration hyper-parameters
    :param mutator_config: Mutator configuration dictionary
    :param reproducer_config: Reproducer configuration dictionary
    """

    self.ea_pairs = EnvPairs(init_agent_config, log_dir)

    # Setup POET hyper-parameters
    self.max_poet_iters = poet_config['max_poet_iters']
    self.mutation_interval = poet_config['mutation_interval']
    self.transfer_interval = poet_config['transfer_interval']
    self.train_episodes = poet_config['train_episodes']

    self.mutator = Mutator(
      max_admitted=mutator_config['max_admitted'],
      capacity=mutator_config['capacity'],
      min_performance=mutator_config['min_performance'],
      mc_low=mutator_config['mc_low'],
      mc_high=mutator_config['mc_high'],
      reproducer_config=reproducer_config
    )

  def run(self):
    """
    Run main POET loop, entry point for main POET algorithm after construction

    :return: None
    """
    for poet_step in range(self.max_poet_iters):
      if poet_step % self.mutation_interval == 0:
        self.ea_pairs.pairs, self.ea_pairs.archived_pairs = self.mutator.mutate_env(self.ea_pairs)
      print(f'mutated {poet_step} times')
      # Train each ea_pair in list
      for idx, pair in enumerate(self.ea_pairs.pairs):
        if idx > 0:
          pair = self.ea_pairs.train_agent(pair)
          self.ea_pairs.pairs[idx] = pair
      # Attempt mutation if able
      for idx, pair in enumerate(self.ea_pairs.pairs):
        if idx > 0 and poet_step % self.mutation_interval == 0:
          eval_pairs = self.ea_pairs.pairs
          eval_pairs.remove(pair)
          best_agent, best_score = self.ea_pairs.evaluate_transfer(pair.env_config)
          pair = pair._replace(agent_config=best_agent, agent_score=best_score)
          self.ea_pairs.pairs[idx] = pair


def main(_):

  stub_env_config = Env_config(
        name='stub_env',
        ground_roughness=0,
        pit_gap=[],
        stump_width=[],
        stump_height=[],
        stump_float=[],
        stair_height=[],
        stair_width=[],
        stair_steps=[]
      )

  # Setup tf values
  tf.compat.v1.enable_resource_variables()
  tf.compat.v1.disable_eager_execution()

  # Setup logging
  logging.set_verbosity(logging.INFO)

  # Setup initial directories
  root_dir, log_dir, save_dir = setup_top_dirs(FLAGS.logdir, FLAGS.environment)
  log_dir, model_dir, save_dir = setup_agent_dir(log_dir, 'default_env')

  init_env_config = Env_config(
    name='default_env',
    ground_roughness=0,
    pit_gap=[],
    stump_width=[],
    stump_height=[],
    stump_float=[],
    stair_height=[],
    stair_width=[],
    stair_steps=[]
  )

  # Setup initial dads configuration, must be done in main due to use of flags
  init_dads_config = {
    'env_name': FLAGS.environment,
    'env_config': init_env_config,
    'log_dir': model_dir,
    'num_skills': FLAGS.num_skills,
    'skill_type': FLAGS.skill_type,
    'random_skills': FLAGS.random_skills,
    'min_steps_before_resample': FLAGS.min_steps_before_resample,
    'resample_prob': FLAGS.resample_prob,
    'max_env_steps': FLAGS.random_skills,
    'observation_omit_size': 0,
    'reduced_observation': FLAGS.reduced_observation,
    'hidden_layer_size': FLAGS.hidden_layer_size,
    'save_dir': save_dir,
    'skill_dynamics_observation_relabel_type': FLAGS.skill_dynamics_relabel_type,
    'skill_dynamics_relabel_type': FLAGS.skill_dynamics_relabel_type,
    'is_clip_eps': FLAGS.is_clip_eps,
    'normalize_data': FLAGS.normalize_data,
    'graph_type': FLAGS.graph_type,
    'num_components': FLAGS.num_components,
    'fix_variance': FLAGS.fix_variance,
    'skill_dynamics_lr': FLAGS.skill_dynamics_lr,
    'agent_lr': FLAGS.agent_lr,
    'agent_gamma': FLAGS.agent_gamma,
    'agent_entropy': FLAGS.agent_entropy,
    'debug': FLAGS.debug,
    'collect_policy_type': FLAGS.collect_policy,
    'replay_buffer_capacity': FLAGS.replay_buffer_capacity,
    'train_skill_dynamics_on_policy': FLAGS.train_skill_dynamics_on_policy,
    'initial_collect_steps': FLAGS.initial_collect_steps,
    'collect_steps': FLAGS.collect_steps,
    'action_clipping': FLAGS.action_clipping,
    'num_epochs': FLAGS.num_epochs,
    'save_model': FLAGS.save_model,
    'save_freq': FLAGS.save_freq,
    'clear_buffer_every_iter': FLAGS.clear_buffer_every_iter,
    'skill_dynamics_train_steps': FLAGS.skill_dyn_train_steps,
    'skill_dynamics_batch_size': FLAGS.skill_dyn_batch_size,
    'num_samples_for_relabelling': FLAGS.num_samples_for_relabelling,
    'debug_skill_relabelling': FLAGS.debug_skill_relabelling,
    'agent_train_steps': FLAGS.agent_train_steps,
    'agent_relabel_type': FLAGS.agent_relabel_type,
    'agent_batch_size': FLAGS.agent_batch_size,
    'record_freq': FLAGS.record_freq,
    'vid_name': FLAGS.vid_name,
    'deterministic_eval': FLAGS.deterministic_eval,
    'num_evals': FLAGS.num_evals,
    'restore_training': True
  }

  poet_config = {
    'max_poet_iters': 20,
    'mutation_interval': 4,
    'transfer_interval': 8,
    'train_episodes': 10
  }

  mutator_config = {
    'max_admitted': 6,
    'capacity': 15,
    'min_performance': -4000,
    'mc_low': -4000,
    'mc_high': 4000,
  }

  reproducer_config = {
    'env_categories': ['pit'],
    'master_seed': 4,
    'max_children': 3
  }

  poet = POET(init_dads_config,
              log_dir,
              poet_config=poet_config,
              mutator_config=mutator_config,
              reproducer_config=reproducer_config)
  poet.run()
  print('Finished running POET!')

  # ea_pairs = EnvPairs(init_dads_config, log_dir)
  # ea_pairs.train_on_new_env(stub_env_config)
  # import pprint
  # pp = pprint.PrettyPrinter(indent=4)
  # pp.pprint(ea_pairs.pairs)
  # # TODO: manage access to agent_dir and env_config


if __name__ == '__main__':
  pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
  tf.compat.v1.app.run(main)
