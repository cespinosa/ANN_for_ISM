"""
Testing different sizes on training sample
"""
# !/usr/bin/env python3
import numpy as np
import pandas as pd
from ai4neb import manage_RM
from get_data import get_bond_data


def ann_training(data: pd.DataFrame, filename: str,
                 split_ratio: float) -> None:
    """
    ANN training with a data set
    """
    x_keys = ['O', 'NO', 'logU', 'fr', 'age', 'HbFrac']
    y_keys = ['O3', 'N2', 'O2', 'rO3', 'rN2', 'Np_Op']
    train_x_set = np.array(data[x_keys])
    train_y_set = np.array(data[y_keys])
    RM_filename = filename
    RM_ANN = manage_RM(RM_filename=RM_filename)
    if not RM_ANN.model_read:
        RM_ANN = manage_RM(RM_type='SK_ANN',
                           X_train=train_x_set, y_train=train_y_set,
                           verbose=True, scaling=True, split_ratio=split_ratio)
        RM_ANN.init_RM(max_iter=20000, tol=1e-8, solver='lbfgs',
                       activation='tanh',
                       hidden_layer_sizes=(50, 50, 50))
        RM_ANN.train_RM()
        RM_ANN.save_RM(RM_filename, save_train=True)
    print(f'Trained with {RM_ANN.N_train} models,' +
          f'tested with {RM_ANN.N_test}')


def main():
    """
    Main function
    """
    bond_data = get_bond_data('/home/espinosa/data/data_BOND.csv',
                              force=False)
    print('Total number of Models:', len(bond_data))
    output_path = '/home/espinosa/data/ANNs/'
    split_values_list = [i/10 for i in np.arange(1, 10)]
    for split_value in split_values_list:
        split_label = str(np.round(1-split_value, 1))
        RM_filename = 'ANN_BOND_ALL'+'_'+split_label
        ann_training(bond_data, filename=output_path+RM_filename,
                     split_ratio=split_value)


if __name__ == '__main__':
    main()
