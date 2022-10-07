import numpy as np
from ai4neb import manage_RM

def load_ANN(RMName, X_train, y_train):
    RM = manage_RM(RM_filename=RMName)
    if not RM.model_read:
        RM = manage_RM(RM_type='SK_ANN',
                       X_train=X_train, y_train=y_train,
                       verbose=True, scaling=True)
        RM.init_RM(max_iter=20000, tol=1e-8, solver='lbfgs',
                   activation='tanh',
                   hidden_layer_sizes=(50,50,50))
        RM.train_RM()
        RM.save_RM(RMName, save_train=True)
    print('Trained with {} models, tested with {} models'.format(RM.N_train,
                                                                 RM.N_test))
    return RM

def read_ANN(ANN, DataPathANNs):
  print('Reading', ANN)
  ANNname = ANN
  ANNs = load_ANN(DataPathANNs+ANNname,
                  X_train=None, y_train=None)
  print('Reading ANNs: Done')
  return ANNs