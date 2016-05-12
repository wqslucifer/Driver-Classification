# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 16:24:10 2016

@author: qiushi
"""

import os
import cv2
import sys
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

''' cache image: save by driver id '''

img_cols = 36
img_rows = 24
PIXELS = 24
imageSize = PIXELS * PIXELS
num_features = imageSize
main_path1 = 'F:\\kaggle\\Driver Classification'
main_path2 = 'F:\\kaggle\\State Farm Distracted Driver Detection\\'
driver_id_path = os.path.join(main_path1, 'source_data', 'driver_imgs_list.csv')


# p002_$number$_$DATE$.npy


def get_current_date():
    return time.strftime('%Y%m%d')


def show_image(im, name='image'):
    cv2.imshow(name, im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def create_cache(driver_path, img_cols, img_rows, color_type):
    df_driver = pd.read_csv(driver_path)
    # driver_id_list = pd.unique(df_driver['subject'])
    # sub_dr = df_driver.loc[df_driver.subject == driver_id]
    X_shape = (len(df_driver), 3, img_rows, img_cols)
    y_shape = (len(df_driver), 2)

    X_filename = 'cache/X_%s.npy' % (get_current_date())  # save image by driver id
    y_filename = 'cache/y_%s.npy' % (get_current_date())  # save image class
    X_filename = os.path.join(main_path1, X_filename)
    y_filename = os.path.join(main_path1, y_filename)

    if os.path.exists(X_filename):
        print('cache file %s exists' % X_filename)
        sys.exit(0)
    if os.path.exists(y_filename):
        print('cache file %s exists' % y_filename)
        sys.exit(0)
    X_fp = np.memmap(X_filename, dtype=np.float32, mode='w+', shape=X_shape)
    y_fp = np.memmap(y_filename, dtype=np.int8, mode='w+', shape=y_shape)

    for i, row in tqdm(df_driver.iterrows(), total=len(df_driver)):
        filename = os.path.join(main_path2, 'train', row['classname'], row['img'])
        try:
            if color_type == 1:
                img = cv2.imread(filename, 0)
            elif color_type == 3:
                img = cv2.imread(filename)

            rotate = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotate, 1)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
            img = cv2.resize(img, (img_cols, img_rows), cv2.INTER_LINEAR)
            # show_image(img) # resized.shape == ((3, img_rows, img_cols)
            img = img.transpose(2, 0, 1)  # to form (3, img_rows, img_cols) works on color_type 3
            img = img.astype(np.float32)

            assert img.shape == (3, img_rows, img_cols)
            assert img.dtype == np.float32

            X_fp[i] = img
            y_fp[i] = (np.int8(row.classname[1]), np.int8(row.subject[1:]))

            X_fp.flush()
            y_fp.flush()
        except():
            print('%s has failed' % i)


create_cache(driver_id_path, img_cols, img_rows, color_type=3)
