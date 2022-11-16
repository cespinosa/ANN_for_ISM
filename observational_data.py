import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_obs_data(path=None):
  if path is None:
    path = os.path.expanduser('~/GoogleDrive/cespinosa/data/obs/') +\
      'Marino_CHAOS_data_clean_v2.csv'
    print(path)
  data = pd.read_csv(path)
  return data

def obs_data_O3N2_plot(data, ax=None, xlim=[-2.7, 0.5], ylim=[-1.5, 1.2],
                       params_dict=None):
  if params_dict is None:
    params_dict = dict(c='k', s=20, marker='*', alpha=0.5)
  if ax is None:
    ax_flag = False
    fig, ax = plt.subplots()
  ax.scatter(data['lN2'], data['lO3'], **params_dict)
  ax.set_xlim(xlim[0], xlim[1])
  ax.set_ylim(ylim[0], ylim[1])

if __name__ == '__main__':
  data = read_obs_data()
  obs_data_O3N2_plot(data)
  plt.show()