import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def episode_stat_plot(data):
  """
  Given statistics from a single run generate a graph against the steps

  :param data:
  :return: None
  """
  plt.plot(data)
