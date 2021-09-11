from collections import namedtuple
import json
import os

from envs.bipedal_walker_custom import Env_config

Env_performance = namedtuple('Env_performance', [
  'name',
  'parent',
  'tot_return',
  'active_env'
])


def read_json(json_file_name):
  with open(json_file_name) as json_file:
    dictionary = json.load(json_file)
  return dictionary


def write_config_json(config_data, env_dir):
  json_file_name = os.path.join(env_dir, 'config.json')
  with open(json_file_name, 'w') as out_file:
    json.dump(config_data._asdict(), out_file)


def write_perf_json(perf_data, env_dir):
  json_file_name = os.path.join(env_dir, 'perf.json')
  with open(json_file_name, 'w') as out_file:
    json.dump(perf_data._asdict(), out_file)


def write_tree_json(tree_data, log_dir):
  json_file_name = os.path.join(log_dir, 'tree.json')
  if os.path.exists(json_file_name):
    os.remove(json_file_name)
  with open(json_file_name, 'w') as out_file:
    json.dump(tree_data, out_file)


def read_config_json(env_dir):
  json_file_name = os.path.join(env_dir, 'config.json')
  with open(json_file_name) as json_file:
    env = json.load(json_file)
  env = Env_config(**env)
  return env


def read_perf_json(env_dir):
  json_file_name = os.path.join(env_dir, 'perf.json')
  with open(json_file_name) as json_file:
    perf = json.load(json_file)
  perf = Env_performance(**perf)
  return perf
