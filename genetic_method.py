
import os
import numpy as np
import pandas as pd
from itertools import product
from typing import Callable
from ANN_manager import read_ANN
from data import create_model
from observational_data import read_Marino_data
from scipy.spatial import KDTree

def mutate(parent, n_childs):
  childs = parent + np.random.uniform(-1,1,n_childs)
  return childs

def make_childs(ANN, data, vars, n_childs=10):
  child_params = {param: mutate(data.loc[param], n_childs) for param in vars}
  child_models_list = []
  for i in np.arange(n_childs):
    params = {key:value[i] for key, value in child_params.items()}
    child_model = create_model(ANN, params, 'points', [6.6, 9.4])
    child_models_list.append(child_model)
  child_models_pd = pd.concat(child_models_list, ignore_index=True, axis=1).T
  return child_models_pd

def randomize_params(params):
  params_new = {key:value+np.random.uniform(-0.5, 0.5)
                for key, value in params.items()}
  return params_new

def create_parents(ANN, nmodels=10, distr='uniform', OHlims=[6.7, 9.4]):
  x1_OH_lst = np.ones(nmodels) * OHlims[0]
  x2_OH_lst = np.ones(nmodels) * OHlims[1]
  if distr == 'uniform':
    y1_NO_lst = np.random.uniform(low=-2, high=0, size=nmodels) 
    y2_NO_lst = np.random.uniform(low=-2, high=0, size=nmodels)
    xb_NO_lst = np.random.uniform(low=7.5, high=8.5, size=nmodels)
    yb_NO_lst = y2_NO_lst
    y1_lU_lst = np.random.uniform(low=-4, high=-1, size=nmodels) 
    y2_lU_lst = np.random.uniform(low=-4, high=-1, size=nmodels)
    xb_lU_lst = np.random.uniform(low=7.5, high=8.5, size=nmodels)
    yb_lU_lst = np.random.uniform(low=-4, high=-1, size=nmodels)
  parents_models = []
  for (x1_OH, x2_OH, xb_lU, xb_NO, y1_NO, y2_NO, yb_NO, y1_lU, y2_lU,
       yb_lU) in zip(x1_OH_lst, x2_OH_lst, xb_lU_lst, xb_NO_lst, y1_NO_lst,
                     y2_NO_lst, yb_NO_lst, y1_lU_lst, y2_lU_lst, yb_lU_lst):
    params_points = {'x1_NO':x1_OH, 'x2_NO':x2_OH, 'xb_NO':xb_NO,
                     'x1_lU':x1_OH, 'x2_lU':x2_OH, 'xb_lU':xb_lU,
                     'y1_NO':y1_NO, 'y2_NO':y2_NO, 'yb_NO':yb_NO,
                     'y1_lU':y1_lU, 'y2_lU':y2_lU, 'yb_lU':yb_lU}
    parents_models.append(create_model(ANN, params_points, 'points', OHlims))
  return pd.concat(parents_models, axis=1).transpose() 

def get_cneighbors(data_model, data_obs, nparents):
  tree = KDTree(data_obs)
  # inds = ctree.query_ball_point(data_model, radius)
  dist, inds = tree.query(data_model, nparents)
  inds = inds[0]
  return inds, dist

def next_gen(data, data_obs, nparents):
  indices, distances = get_cneighbors(data, data_obs, nparents)
  
  

if __name__ == '__main__':
  ANN = read_ANN('ANN_BOND_ALL',
                  os.path.expanduser('~/GoogleDrive/cespinosa/data/ANNs/'))
  model = create_model(ANN, {'a1': -1.37, 'a3': -1.37, 'x0_2': 8,
                        'b1': 8.42, 'c': -1.6, 'a2': 1, 'x0': 8},
                        'func', [6.6, 9.4]) 

  vars_gen = [["x","y"],["1","2","b"],["NO","lU"]]
  vars_names = [fst+snd+'_'+thr for fst, snd, thr in product(*vars_gen)]

  child_models = make_childs(ANN, model, vars_names, 2)