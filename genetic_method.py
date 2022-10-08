
import os
import numpy as np
import pandas as pd
from itertools import product
from typing import Callable
from ANN_manager import read_ANN
from data import create_model

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


ANN = read_ANN('ANN_BOND_ALL',
                os.path.expanduser('~/GoogleDrive/cespinosa/data/ANNs/'))
model = create_model(ANN, {'a1': -1.37, 'a3': -1.37, 'x0_2': 8,
                      'b1': 8.42, 'c': -1.6, 'a2': 1, 'x0': 8},
                      'func', [6.6, 9.4]) 

vars_gen = [["x","y"],["1","2","b"],["NO","lU"]]
vars_names = [fst+snd+'_'+thr for fst, snd, thr in product(*vars_gen)]

child_models = make_childs(ANN, model, vars_names, 2)
