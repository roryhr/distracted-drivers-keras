from __future__ import print_function
from __future__ import division

import os
import glob
import pickle
# import random
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
# from skimage.io import imread, imsave
# from scipy.misc import imresize

SUBSET = False
DOWNSAMPLE = 20
NUM_CLASSES = 10

WIDTH, HEIGHT = 640 // DOWNSAMPLE, 480 // DOWNSAMPLE


def load_train(base):
    driver_imgs_list = pd.read_csv('dataset/driver_imgs_list.csv')
    driver_imgs_grouped = driver_imgs_list.groupby('classname')

    X_train = []
    y_train = []
    driver_ids = []

    print('Reading train images...')
    for j in range(NUM_CLASSES):
        print('Loading folder c{}...'.format(j))
        paths = glob.glob(os.path.join(base, 'c{}/*.jpg'.format(j)))
        driver_ids_group = driver_imgs_grouped.get_group('c{}'.format(j))

        if SUBSET:
            paths = paths[:100]
            driver_ids_group = driver_ids_group.iloc[:100]

        driver_ids += driver_ids_group['subject'].tolist()

        for i, path in enumerate(paths):
            X_train.append(path)
            y_train.append(j)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    y_train = OneHotEncoder(n_values=NUM_CLASSES) \
        .fit_transform(y_train.reshape(-1, 1)) \
        .toarray()

    return X_train, y_train, driver_ids


def load_test(base):
    X_test = []
    X_test_id = []
    paths = glob.glob(os.path.join(base, '*.jpg'))

    if SUBSET:
        paths = paths[:100]

    print('Reading test images...')
    for path in paths:
        img_id = os.path.basename(path)
        X_test.append(path)
        X_test_id.append(img_id)

    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)

    return X_test, X_test_id

X_train, y_train, driver_ids = load_train('dataset/imgs/train/')
X_test, X_test_ids = load_test('dataset/imgs/test/')

if SUBSET:
    dest = 'data_files_subset.pkl'
else:
    dest = 'data_files.pkl'

with open(dest, 'wb') as f:
    pickle.dump((X_train, y_train, X_test, X_test_ids, driver_ids), f)
