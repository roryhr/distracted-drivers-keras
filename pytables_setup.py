from __future__ import print_function

import os
import glob
import pandas as pd
from scipy import misc

from sklearn.preprocessing import OneHotEncoder
import tables

# %% Configuration
hdf5_file_name = "some_photos.hdf5"
SUBSET = True
NUM_CLASSES = 10
HEIGHT, WIDTH = 480, 640
filter_ = tables.Filters(complevel=9)
h5file = tables.open_file(hdf5_file_name,
                          mode="w",
                          title="Statefarm Photos")

# enc = OneHotEncoder(sparse=False)
# enc.fit(np.arange(9).reshape(9, 1))


def load_train(base):
    class Images(tables.IsDescription):
        image = tables.UInt8Col(shape=(HEIGHT, WIDTH, 3))  # Unsigned 8 byte integers
        file_name = tables.StringCol(40)
        driver_id = tables.StringCol(4)
        class_id = tables.IntCol()

    table = h5file.create_table(where='/', name='train', description=Images,
                                title='Train photo and filename',
                                filters=filter_,
                                expectedrows=20000)
    image_row = table.row

    driver_imgs_list = pd.read_csv('dataset/driver_imgs_list.csv')
    driver_imgs_grouped = driver_imgs_list.groupby('classname')

    print('Reading train images...')
    for j in range(NUM_CLASSES):
        print('Loading folder c{}...'.format(j))
        paths = glob.glob(os.path.join(base, 'c{}/*.jpg'.format(j)))
        driver_ids_group = driver_imgs_grouped.get_group('c{}'.format(j))

        if SUBSET:
            paths = paths[:100]
            driver_ids_group = driver_ids_group.iloc[:100]

        driver_ids = driver_ids_group['subject'].tolist()

        for path, driver_id in zip(paths, driver_ids):
            image_row['image'] = misc.imread(path)
            image_row['file_name'] = path
            image_row['driver_id'] = driver_id
            image_row.append()

        table.flush()


def load_test(base):
    class TestImages(tables.IsDescription):
        image = tables.UInt8Col(shape=(HEIGHT, WIDTH, 3))  # Unsigned 8 byte integers
        file_name = tables.StringCol(40)
        image_id = tables.StringCol(18)

    table = h5file.create_table(where='/', name='test', description=TestImages,
                                title='Test images and filename',
                                filters=filter_,
                                expectedrows=80000)
    image_row = table.row


    paths = glob.glob(os.path.join(base, '*.jpg'))

    if SUBSET:
        paths = paths[:100]

    print('Reading test images...')
    for path in paths:
        img_id = os.path.basename(path)

        image_row['image'] = misc.imread(path)
        image_row['file_name'] = path
        image_row['image_id'] = img_id
        image_row.append()

    table.flush()


load_train('dataset/imgs/train/')
load_test('dataset/imgs/test/')

h5file.close()
