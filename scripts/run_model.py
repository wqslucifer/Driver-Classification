# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:24:10 2016

@author: qiushi
"""

import os
import cv2
import sys
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

import theano
import theano.tensor as T

from sklearn.cross_validation import KFold

''' cache image: save by driver id '''

PIXELS = 24
img_cols = 36
img_rows = 24
imageSize = PIXELS * PIXELS
num_features = imageSize
main_path1 = 'F:\\kaggle\\Driver Classification'
main_path2 = 'F:\\kaggle\\State Farm Distracted Driver Detection\\'
driver_id_path = os.path.join(main_path1, 'source_data', 'driver_imgs_list.csv')
current_date = 20160511


def load_data(driver_path, driver_id_list, save_date, img_cols, img_rows):
    train_set = np.array()
    labels = np.array()
    df_driver = pd.read_csv(driver_path)
    for driver_id in driver_id_list:  # load driver files by list
        X_filename = 'cache/X_%s_%s.npy' % (driver_id, save_date)
        y_filename = 'cache/y_%s_%s.npy' % (driver_id, save_date)
        X_filename = os.path.join(main_path1, X_filename)
        y_filename = os.path.join(main_path1, y_filename)

        print("load driver %s" % driver_id)
        sub_dr = df_driver.loc[df_driver.subject == driver_id]  # get sub_dr list
        X_shape = (len(sub_dr), 3, img_rows, img_cols)
        y_shape = (len(sub_dr), 2)
        X = np.memmap(X_filename, dtype=np.float32, mode='r', shape=X_shape)
        y = np.memmap(y_filename, dtype=np.int8, mode='r', shape=y_shape)
        X = np.array(X)
        y = np.array(y)
        # concat
        train_set = np.concatenate((train_set, X), axis=0)
        labels = np.concatenate((labels, y), axis=0)
    return train_set, labels


def k_fold_cv(X, y, test_size=0.2, random_state=42):
    n_folds = int(1 / float(test_size))
    kf = KFold(len(X), n_folds=n_folds, random_state=random_state)
    for train_idx, test_idx in kf:
        pass
# np.concatenate((a,b), axis=0)
