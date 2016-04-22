from __future__ import print_function
from __future__ import division

import os
import time
import pickle
import numpy as np

from keras.models import Model
from keras.layers import (Input,
                          Activation,
                          merge,
                          Dense,
                          Flatten)
from keras.layers.convolutional import (Convolution2D,
                                        MaxPooling2D,
                                        AveragePooling2D)
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.metrics import log_loss
from sklearn.cross_validation import LabelShuffleSplit

from utilities import write_submission, calc_geom, calc_geom_arr, mkdirp


TESTING = True

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/data_20.pkl' if not TESTING else 'dataset/data_20_subset.pkl')

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summaries/')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')

mkdirp(CHECKPOINT_PATH)
mkdirp(SUMMARY_PATH)
mkdirp(MODEL_PATH)

NB_EPOCHS = 5 if not TESTING else 1
MAX_FOLDS = 1
DOWNSAMPLE = 20

WIDTH, HEIGHT, NB_CHANNELS = 640//DOWNSAMPLE, 480//DOWNSAMPLE, 3
BATCH_SIZE = 60

with open(DATASET_PATH, 'rb') as f:
    X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids = pickle.load(f)

_, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)

predictions_total = []  # accumulated predictions from each fold
scores_total = []  # accumulated scores from each fold
num_folds = 0




# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                             init="he_normal", border_mode="same")(activation)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._shape[2].value / residual._shape[2].value
    stride_height = input._shape[3].value / residual._shape[3].value
    equal_channels = residual._shape[1].value == input._shape[1].value

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._shape[1].value, nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal", border_mode="valid")(input)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetations, is_first_layer=False):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# http://arxiv.org/pdf/1512.03385v1.pdf
# 50 Layer resnet
def resnet():
    input = Input(shape=(3, 224, 224))

    conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

    # Build residual blocks..
    block_fn = _bottleneck
    block1 = _residual_block(block_fn, nb_filters=64, repetations=3, is_first_layer=True)(pool1)
    block2 = _residual_block(block_fn, nb_filters=128, repetations=4)(block1)
    block3 = _residual_block(block_fn, nb_filters=256, repetations=6)(block2)
    block4 = _residual_block(block_fn, nb_filters=512, repetations=3)(block3)

    # Classifier block
    pool2 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode="same")(block4)
    flatten1 = Flatten()(pool2)
    dense = Dense(output_dim=1000, init="he_normal", activation="softmax")(flatten1)

    model = Model(input=input, output=dense)
    return model








for train_index, valid_index in LabelShuffleSplit(driver_indices, n_iter=MAX_FOLDS,
                                                  test_size=0.2, random_state=67):
    print('Fold {}/{}'.format(num_folds + 1, MAX_FOLDS))

    # skip fold if a checkpoint exists for the next one
    # next_checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}.h5'.format(num_folds + 1))
    # if os.path.exists(next_checkpoint_path):
    #     print('Checkpoint exists for next fold, skipping current fold.')
    #     continue

    X_train, y_train = X_train_raw[train_index, ...], y_train_raw[train_index, ...]
    X_valid, y_valid = X_train_raw[valid_index, ...], y_train_raw[valid_index, ...]

    # model = vgg_bn()
    # model = vgg2_plain()
    model = resnet()
    model.compile(loss="categorical_crossentropy", optimizer="sgd")

    model_path = os.path.join(MODEL_PATH, 'model_{}.json'.format(num_folds))
    with open(model_path, 'w') as f:
        f.write(model.to_json())

    # restore existing checkpoint, if it exists
    checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}.h5'.format(num_folds))
    if os.path.exists(checkpoint_path):
        print('Restoring fold from checkpoint.')
        model.load_weights(checkpoint_path)

    summary_path = os.path.join(SUMMARY_PATH, 'model_{}'.format(num_folds))
    mkdirp(summary_path)

    callbacks = [EarlyStopping(monitor='val_loss', patience=1, verbose=0, mode='auto'),
                 ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                 TensorBoard(log_dir=summary_path, histogram_freq=0)]
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, nb_epoch=NB_EPOCHS,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, y_valid),
              callbacks=callbacks)

    predictions_valid = model.predict(X_valid, batch_size=100, verbose=1)
    score_valid = log_loss(y_valid, predictions_valid)
    scores_total.append(score_valid)

    print('Score: {}'.format(score_valid))

    predictions_test = model.predict(X_test, batch_size=100, verbose=1)
    predictions_total.append(predictions_test)

    num_folds += 1

score_geom = calc_geom(scores_total, MAX_FOLDS)
predictions_geom = calc_geom_arr(predictions_total, MAX_FOLDS)

submission_path = os.path.join(SUMMARY_PATH, 'submission_{}_{:.2}.csv'.format(int(time.time()), score_geom))
write_submission(predictions_geom, X_test_ids, submission_path)
