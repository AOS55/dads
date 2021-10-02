import numpy as np
from unsupervised_skill_learning.eval_analysis import trajectory_diff, calculate_trajectory_error_stats,\
  plot_trajectory_planner_error, density_estimation, find_best_z


# def test_trajectory_diff():
#   actual_trajectory = np.load('test_trajectory.npy')
#   predicted_trajectory = np.load('test_predicted_trajectory.npy')
#   predicted_error = trajectory_diff(actual_trajectory, predicted_trajectory)
#   mean_error, var_error = calculate_trajectory_error_stats(predicted_error)
#   plot_trajectory_planner_error(mean_error, var_error)


def test_reward():
  one_hot_stats = np.load('one_hot_trajectories.npy', allow_pickle=True)
  z_max = find_best_z(one_hot_stats)
  print(z_max)


# def test_diversity():
#   trajectories = np.load('one_hot_trajectories.npy', allow_pickle=True)
#   density_estimation(trajectories)
