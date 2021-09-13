from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import pickle as pkl
import os
import io
from absl import logging, flags
import functools

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

import dads_agent

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
flags.DEFINE_integer('deterministic_eval', 0,
                  'Evaluate all skills, only works for discrete skills')

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


class DADS:
  """Core class to implement a single DADS EA pair object"""
  def __init__(self,
               env_name,
               env_config,
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
               skill_dynamics_observation_size,
               skill_dynamics_observation_relabel_type,
               skill_dynamics_relabel_type,
               is_clip_eps,
               normalize_data,
               graph_type,
               num_components,
               fix_variance,
               reweigh_batches,
               skill_dynamics_lr,
               agent_lr,
               agent_gamma,
               reward_scale_factor,
               debug,
               collect_policy_type,
               replay_buffer_capacity,
               train_skill_dynamics_on_policy,
               initial_collect_steps,
               collect_steps,
               global_step
               ):
    self.env_name = env_name
    self.env_config = env_config
    self.env = get_environment(env_name, env_config)

    # Initialize env parameters
    self.num_skills = num_skills
    self.skill_type = skill_type
    self.random_skills = random_skills
    self.max_env_steps = max_env_steps
    self.py_env = self.wrap_env(min_steps_before_resample, resample_prob)

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
    self.skill_dynamics_observation_size = skill_dynamics_observation_size
    self.skill_dynamics_observation_relabel_type = skill_dynamics_observation_relabel_type
    self.skill_dynamics_relabel_type = skill_dynamics_relabel_type
    self.is_clip_eps = is_clip_eps
    self.normalize_data = normalize_data
    self.graph_type = graph_type
    self.num_components = num_components
    self.fix_variance = fix_variance
    self.reweigh_batches = reweigh_batches
    self.skill_dynamics_lr = skill_dynamics_lr
    self.agent_lr = agent_lr
    self.agent_gamma = agent_gamma
    self.reward_scale_factor = reward_scale_factor
    self.debug = debug
    self.agent = self.get_dads_agent()

    # Initialize policies
    self.eval_policy, self.collect_policy, self.relabel_policy = self.initialize_policies(collect_policy_type)

    # Initialize replay buffer
    self.policy_step_spec, self.trajectory_spec = self.define_buffer_spec()
    self.replay_buffer_capacity = replay_buffer_capacity
    self.train_skill_dynamics_on_policy = train_skill_dynamics_on_policy
    self.initial_collect_steps = initial_collect_steps
    self.collect_steps = collect_steps
    self.rbuffer, self.on_buffer = self.initialize_buffer()

    # Initialize agent methods
    self.agent.build_agent_graph()
    self.agent.build_skill_dynamics_graph()
    self.agent.create_savers()

    # Save current setup
    self.train_checkpointer, self.policy_checkpointer, self.rb_checkpointer = self.initialize_checkpoints(global_step)

  def wrap_env(self, min_steps_before_resample, resample_prob):
    """
    Wrap an environment with the skill wrapper

    :num_skills: number of skills to learn i.e. size of p(z) ~ Z
    :skill_type: types of skills to learn from p(z)
    :preset_skill: what preset skills are used (often None)
    :min_steps_before_resample: minimum number of steps to go before resampling skill distro
    :resample_prob: likelihood of resampling the distribution
    :return: None
    """
    py_env = wrap_env(
      skill_wrapper.SkillWrapper(
        self.env,
        self.num_skills,
        self.skill_type,
        min_steps_before_resample,
        resample_prob),
      self.max_env_steps
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
      (self.tf_agent_time_step_spec, self.tf_action_spec),
      observation_fc_layer_params=None,
      action_fc_layer_params=None,
      joint_fc_layer_params=(self.hidden_layer_size,) * 2,
    )

  def get_dads_agent(self):
    """
    Get a dads agent instance from the DADSAgent class

    :return: agent object
    """
    # Set a global step type
    global_step = tf.compat.v1.train.get_or_create_global_step()  # graph in which to create global step tensor
    if self.skill_dynamics_observation_relabel_type is not None and 'importance_sampling' in \
            self.skill_dynamics_relabel_type and self.is_clip_eps > 1.0:
      reweigh_batches_flag = True
    else:
      reweigh_batches_flag = False

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
      reweigh_batches=self.reweigh_batches,
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
      reward_scale_factor=self.reward_scale_factor,
      gradient_clipping=None,
      debug_summaries=self.debug,
      train_step_counter=global_step
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


def main(_):

  # Setup tf values
  tf.compat.v1.enable_resource_variables()
  tf.compat.v1.disable_eager_execution()

  # Setup logging
  logging.set_verbosity(logging.INFO)

  root_dir, log_dir, save_dir = setup_top_dirs(FLAGS.logdir, FLAGS.environment)

  # Get initial gym environment
  dads_algo = DADS(FLAGS.environment)
  print(dads_algo.env)


if __name__ == '__main__':
  try:
    tf.compat.v1.app.run(main)
  except SystemExit:
    pass
