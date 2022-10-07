import os
import numpy as np
import pandas as pd
from itertools import product
from ANN_manager import read_ANN

def arctg_w(x, x0, tol_v=0.1):
  return (np.pi/2+np.arctan((x-x0)/tol_v))/np.pi

def line_f(x, a, b):
  return a * x + b

def line_2_physics_params(params: list, oxy_lims: list):
  a1, a3, x0_2, b1, c, a2, x0 = params.values()
  b2 = c - a2 * x0
  b3 = (a1-a3)*x0_2 + b1
  phys_params = {'x1_NO': oxy_lims[0],
                 'x2_NO': oxy_lims[1],
                 'y1_NO': c,
                 'y2_NO': a2*(oxy_lims[1])+b2,
                 'xb_NO': x0,
                 'yb_NO': c,
                 'x1_lU': oxy_lims[0],
                 'x2_lU': oxy_lims[1],
                 'y1_lU': a1*oxy_lims[0]+b1,
                 'y2_lU': a3*oxy_lims[1]+b3,
                 'xb_lU': x0_2,
                 'yb_lU': a1*x0_2+b1
                }
  return phys_params

def physics_2_line_params(params: list):
  y1_lU, y2_lU, xb_lU, yb_lU, y2_NO, xb_NO,\
  yb_NO, x1_lU, x2_lU, _, _, x2_NO = params.values()
  a1 = (yb_lU-y1_lU)/(xb_lU-x1_lU)
  a3 = (y2_lU-yb_lU)/(x2_lU-xb_lU)
  b1 = yb_lU-a1*xb_lU
  a2 = (y2_NO-yb_NO)/(x2_NO-xb_NO)
  pts_params = {'a1': a1,
              'a3': a3,
              'x0_2': xb_lU,
              'b1': b1,
              'c': yb_NO,
              'a2': a2,
              'x0': xb_NO
              }
  return pts_params

def get_lU(x, params:list, tol_v=0.1):
  a1, a2, x0, b1 = params.loc[['a1', 'a2', 'x0', 'b1']]
  b2 = (a1 - a2) * x0 + b1
  logU_tab = (line_f(x, a1, b1)*(1-arctg_w(x, x0, tol_v)) + 
              line_f(x, a2, b2)*arctg_w(x, x0, tol_v))
  return logU_tab

def get_NO(x, params:list, tol_v=0.1):
  c, a1, x0 = params.loc[['c', 'a1', 'x0']]
  b1 = c - a1 * x0
  NO_tab = (c*(1-arctg_w(x, x0, tol_v))+line_f(x, a1, b1)*
            arctg_w(x, x0, tol_v))
  return NO_tab

def get_params(params: dict, params_type: str, oxy_lims: list):
  if params_type == 'points':
    pts_params = params
    phys_params = line_2_physics_params(params, oxy_lims)
  else:
    phys_params = params
    phys_params['x1_lU'] = oxy_lims[0]+12
    phys_params['x2_lU'] = oxy_lims[1]+12
    phys_params['x1_NO'] = oxy_lims[0]+12
    phys_params['y1_NO'] = phys_params['y2_NO']
    phys_params['x2_NO'] = oxy_lims[1]+12
    pts_params = physics_2_line_params(phys_params)
  params_pd = pd.Series({**pts_params, **phys_params}) 
  return params_pd


def get_predict_ANN(ANN, tabs):
    ANN.set_test(tabs)
    ANN.predict()
    return ANN.pred

def generate_from_ANN(ANN, OH, logU, NO, Hbfrac, age=1, fr='low'):
  fr_value = {'low': 0.03, 'high':3.0}
  output_names = {0:'O3', 1:'N2', 2:'O2', 3:'rO3', 4:'rN2', 5:'Np_Op'}
  fr_tab = np.ones_like(OH) * fr_value[fr]
  age_tab = np.ones_like(OH) * age
  ANN_data = np.array([OH, NO, logU, fr_tab, age_tab, Hbfrac]).T
  ANN_pred = get_predict_ANN(ANN, ANN_data)
  output = pd.Series({f'{name}_{fr}_{age}':ANN_pred[:,i]
                for (i, name) in output_names.items()})
  return output

def create_model(ANN, params, params_type, oxy_lims,
                 Hbfrac=1, grid_size=100):
  model = get_params(params, params_type, oxy_lims)
  OH_tab = np.linspace(oxy_lims[0]-12, oxy_lims[1]-12, grid_size)
  Hbfrac_tab = np.ones_like(OH_tab) * Hbfrac
  logU_tab = get_lU(OH_tab+12, model) 
  NO_tab = get_NO(OH_tab+12, model)
  model =pd.concat([model, pd.Series({'OH_tab':OH_tab})])
  model =pd.concat([model, pd.Series({'Hbfrac_tab':Hbfrac_tab})])
  model =pd.concat([model, pd.Series({'logU_tab':logU_tab})])
  model =pd.concat([model, pd.Series({'NO_tab':NO_tab})])
  for fr, age in product(['low', 'high'], np.arange(6)):
    model = pd.concat([model,
                      generate_from_ANN(ANN, model['OH_tab'],
                                        model['logU_tab'], model['NO_tab'],
                                        model['Hbfrac_tab'],
                                        age=age, fr=fr)])
  return model
 
# ANN = read_ANN('ANN_BOND_ALL',
#                 os.path.expanduser('~/GoogleDrive/cespinosa/data/ANNs/'))
# model = create_model(ANN, {'a1': -1.37, 'a3': -1.37, 'x0_2': 8,
#                       'b1': 8.42, 'c': -1.6, 'a2': 1, 'x0': 8},
#                       'points', [6.6, 9.4]) 
