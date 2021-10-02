import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
import statsmodels.tsa.statespace.mlemodel as mle
from pydlm import dlm, trend, seasonality, dynamic, autoReg, longSeason
import pandas as pd


def trajectory_diff(actual_trajectory, predicted_trajectory) -> np.array:
  """
  Error between the actual and predicted trajectory

  :param actual_trajectory:
  :param predicted_trajectory:
  :return: error
  """
  prediction_error = []
  for idx, prediction in enumerate(predicted_trajectory):
    if len(prediction) < len(actual_trajectory) - idx:
      error = prediction - actual_trajectory[idx:idx+len(prediction)]
      prediction_error.append(error)
    else:
      error = prediction[:len(actual_trajectory)-idx] - actual_trajectory[idx:]
      prediction_error.append(error)
  return prediction_error


def calculate_trajectory_error_stats(prediction_error) -> np.array:
  """
  Calculate the average error for the prediction at each point further out from the initial starting state

  :param prediction_error:
  :return: mean trajectory error
  """
  pred_error_mean = np.array([])
  pred_error_var = np.array([])
  for step in range(len(prediction_error[0])):
    step_pred_error = [prediction_error[idx][step] for idx in range(len(prediction_error)-step)]
    step_mean = []
    step_var = []
    for obs in range(len(step_pred_error[0])):
      # Calculate mean & var with obs @ each step
      obs_pred_error = np.asarray([step_pred_error[idx][obs] for idx in range(len(step_pred_error))],
                                  dtype=np.float32)
      obs_error_mean = obs_pred_error.mean()
      obs_error_var = obs_pred_error.var()
      step_mean.append(obs_error_mean)
      step_var.append(obs_error_var)
    step_mean = np.array(step_mean, dtype=np.float32)
    step_var = np.array(step_var, dtype=np.float32)
    if step == 0:
      pred_error_mean = np.append(pred_error_mean, step_mean, axis=0)
      pred_error_var = np.append(pred_error_var, step_var, axis=0)
    else:
      pred_error_mean = np.vstack((pred_error_mean, step_mean))
      pred_error_var = np.vstack((pred_error_var, step_var))
  pred_error_mean = np.transpose(pred_error_mean)
  pred_error_var = np.transpose(pred_error_var)
  return pred_error_var, pred_error_mean


def plot_trajectory_planner_error(pred_mean_data, pred_var_data):
  """
  Plot the mean with var areas bounded top and bottom of the main trend

  :param pred_mean_data:
  :param pred_var_data:
  :return: None
  """
  plt.figure()
  idx = 0
  observation_names = ['Hull Angle', 'Hull Angular Velocity', 'Velocity-x', 'Velocity-y', 'Hip 1 Joint Angle',
                       'Hip 1 Joint Speed', 'Knee 1 Joint Angle', 'Knee 1 Joint Speed', 'Leg 1 Ground Contact Flag',
                       'Hip 2 Joint Angle', 'Hip 2 Joint Speed', 'Knee 2 Joint Angle', 'Knee 2 Joint Speed',
                       'Leg 2 Ground Contact Flag', 'LIDAR 1', 'LIDAR 2', 'LIDAR 3', 'LIDAR 4', 'LIDAR 5',
                       'LIDAR 6', 'LIDAR 7', 'LIDAR 8', 'LIDAR 9', 'LIDAR 10']
  for mean_obs, var_obs in zip(pred_mean_data, pred_var_data):
    plt.plot(mean_obs)
    plt.title(observation_names[idx])
    plt.plot(mean_obs+abs(var_obs))
    plt.plot(mean_obs-abs(var_obs))
    plt.fill_between(np.linspace(0, len(mean_obs)-1, len(mean_obs)),
                     mean_obs+abs(var_obs), mean_obs-abs(var_obs),
                     alpha=0.2)
    plt.show()
    idx += 1
  return


def density_estimation(one_hot_array):
  observation_names = ['Hull Angle', 'Hull Angular Velocity', 'Velocity-x', 'Velcoity-y', 'Hip 1 Joint Angle',
                       'Hip 1 Joint Speed', 'Knee 1 Joint Angle', 'Knee 1 Joint Speed', 'Leg 1 Ground Contact Flag',
                       'Hip 2 Joint Angle', 'Hip 2 Joint Speed', 'Knee 2 Joint Angle', 'Knee 2 Joint Speed',
                       'Leg 2 Ground Contact Flag', 'LIDAR 1', 'LIDAR 2', 'LIDAR 3', 'LIDAR 4', 'LIDAR 5',
                       'LIDAR 6', 'LIDAR 7', 'LIDAR 8', 'LIDAR 9', 'LIDAR 10']
  for z in one_hot_array:
    for sample in z:
      trajectories = np.empty((len(sample), len(sample[0][0])))
      for idx in range(len(sample)):
        for obs_id in range(len(sample[idx][0])):
          trajectories[idx][obs_id] = sample[idx][0][obs_id]
      trajectory_df = pd.DataFrame(data=trajectories, columns=observation_names)
      model = mle.MLEModel(trajectories, k_states=20, initialization='stationary', k_posdef=1)
      # This isn't working perhaps try ARMA, I don't know if that is a good idea though?
      # model = KalmanFilter(k_endog=trajectories, k_states=20, initialization='stationary')
      # model['design'] = np.ones(len(sample[0][0])).reshape(len(sample[0][0]), 1)
      # model['obs_cov'] = 100*np.eye(len(sample[0][0]))
      # model['selection'] = np.ones(1)
      # model['state_cov'] = [10000]
      # res = model.filter(sample[idx][0])
      # states_kfsm = res.filtered_state
