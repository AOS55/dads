from collections import namedtuple
import os
from shutil import copy2, move, copy
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
  os.rename(cur_model_name, new_name)
  copy(new_model_path, cur_model_path)
  pair = pair.agent_config._replace(score=new_score)
  return pair
