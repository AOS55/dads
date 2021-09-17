from collections import namedtuple


EAPair = namedtuple('EAPair', [
      'env_name',  # Name of the environment (usually based on the environment)
      'env_config',  # configuration of the envionment
      'agent_config',  # configuration of the agent (contains env_config)
      'agent_score',  # score of the agent
      'parent',  # parent EAPair
      'pata_ec'  # Performance of All Trained Agents Environmental Criteria
    ]
  )
