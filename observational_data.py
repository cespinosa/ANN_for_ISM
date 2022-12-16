import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CALIFA_data import get_data, clean_data_incl

def read_Marino_data(path: str=None) -> pd.DataFrame:
  if path is None:
    path = os.path.expanduser('~/GoogleDrive/cespinosa/data/obs/') +\
      'Marino_CHAOS_data_clean_v2.csv'
    print(path)
  data = pd.read_csv(path)
  return data

def read_CALIFA_data():
  data = get_data()
  data = clean_data_incl(data, 70)

  return data

def get_obs_data():
  Marino_data = read_Marino_data()
  CHAOS_data = Marino_data.loc[565:].copy()
  Marino_data = Marino_data.loc[:564]
  CALIFA_data = read_CALIFA_data()
  OIII4363_snr = CALIFA_data.loc[:,'fluxOIII4363']/CALIFA_data.loc[:,'e_fluxOIII4363']
  mask_OIII4363 = OIII4363_snr > 3
  CALIFA_data = CALIFA_data[mask_OIII4363]
  # Rename columns
  CHAOS_data = CHAOS_data.loc[:,['OIIIHb', 'NIIHb', 'SIIHb']]
  CHAOS_data['NIIHb'] = CHAOS_data['NIIHb']/2.86
  CHAOS_data.rename(columns={'NIIHb': 'NIIHa'},
                    inplace=True)
  Marino_data = Marino_data.loc[:,['lN2', 'lO3', 'OIIHb',
  'OH', 'O3', 'N2', 'O32']]
  Marino_data.rename(columns={'lN2': 'log_NIIHa', 'lO3':'log_OIIIHb',
                              'O3':'log_OIII4363/5007',
                              'N2':'log_NII5755/6584', 'O32':'log_(OII+OIIIHb)'},
                    inplace=True)
  Marino_data['log_OIIHb'] = np.log10(Marino_data['OIIHb'])
  CALIFA_data['log_OIIIHb'] = np.log10(CALIFA_data['O3Hb_dd'])
  CALIFA_data['log_NIIHa'] = np.log10(CALIFA_data['N2Ha_dd'])
  CALIFA_data['log_OIIHb'] = np.log10(CALIFA_data['fluxOII3727'])\
    - np.log10(CALIFA_data['fluxHb4861'])
  CALIFA_data['log_OIII4363/5007'] = np.log10(CALIFA_data['fluxOIII4363']) \
    - np.log10(CALIFA_data['fluxOIII5006'])
  CALIFA_data['log_(OII+OIIIHb)'] = np.log10(CALIFA_data['fluxOII3727']/CALIFA_data['fluxHb4861']
    + CALIFA_data['fluxOIII5006']/CALIFA_data['fluxHb4861'])
  CALIFA_data = CALIFA_data.loc[:,['log_OIIHb', 'log_OIIIHb', 'log_NIIHa',
                                   'log_OIII4363/5007', 'log_(OII+OIIIHb)']]

  return Marino_data, CHAOS_data, CALIFA_data

def get_oxygen_CALIFA():
  CALIFA_data = read_CALIFA_data()
  OIII4363_snr = CALIFA_data.loc[:,'fluxOIII4363']/CALIFA_data.loc[:,'e_fluxOIII4363']
  mask_OIII4363 = OIII4363_snr > 3
  CALIFA_data = CALIFA_data[mask_OIII4363]


def obs_data_O3N2_plot(data, ax=None, xlim=[-2.7, 0.5], ylim=[-1.5, 1.2],
                       params_dict=None):
  if params_dict is None:
    params_dict = dict(c='k', s=20, marker='*', alpha=0.5)
  if ax is None:
    ax_flag = False
    fig, ax = plt.subplots()
  ax.scatter(data['log_NIIHa'], data['log_OIIIHb'], **params_dict)
  ax.set_xlim(xlim[0], xlim[1])
  ax.set_ylim(ylim[0], ylim[1])

if __name__ == '__main__':
  data = get_obs_data()
  obs_data_O3N2_plot(data)
  plt.show()