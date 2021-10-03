from collections import namedtuple
import os
from shutil import copy2, move, copy, copytree, rmtree
from pathlib import Path

EAPair = namedtuple('EAPair', [
      'env_name',  # Name of the environment (usually based on the environment)
      'env_config',  # configuration of the envionment
      'agent_config',  # configuration of the agent (contains env_config)
      'agent_score',  # score of the agent
      'parent',  # parent EAPair
      'pata_ec'  # Performance of All Trained Agents Environmental Criteria
    ]
  )


def transfer_model(new_agent_config, new_score, pair):
  # copy new model across
  id_path = 1
  new_model_path = new_agent_config['save_dir']
  cur_model_path = pair.agent_config['save_dir']
  cur_model_name = Path(cur_model_path).parts[-1]
  cur_model_root_path = Path(cur_model_path).parts[:-1]
  cur_model_root_path = os.path.join(*cur_model_root_path)
  new_name = cur_model_name + '-' + str(id_path)
  model_path = os.path.join(cur_model_root_path, new_name)
  while os.path.exists(model_path):
    id_path += 1
    new_name = cur_model_name + '-' + str(id_path)
    model_path = os.path.join(cur_model_root_path, new_name)
  os.rename(cur_model_path, model_path)
  if os.path.exists(cur_model_path):
    rmtree(cur_model_path)
    copytree(new_model_path, cur_model_path)
  pair = pair._replace(agent_score=new_score)
  return pair
