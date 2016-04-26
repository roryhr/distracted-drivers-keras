import numpy as np
import random
from scipy import misc

#import cv2
#OPEN_CV = False
# RESIZE_SIZE = (256, 480)    # Shorter side length of resized image
RESIZE_SIZE = (40, 50)    # Shorter side length of resized image
CROP_SIZE = (32, 24)      # Size of final image passed into convolution network

    
def grouper(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

# def load_image(imfile):
#     if OPEN_CV:
#         # TODO: Why can't cv2 read this image when PIL can?
#         img = cv2.imread(imfile, 1)
#         img = img[:, :, ::-1]
#     else:
#         img = misc.imread(imfile)
#
#     return img


def list_to_tensor(imgs_list):
    # Convert list into a tensor
    images_tensor = np.stack(imgs_list)

    # Reshape tensor into Theano format
    images_tensor = images_tensor.reshape(-1, 3, CROP_SIZE[0], CROP_SIZE[1])
    return images_tensor.astype('float32')


def resnet_image_processing(file_path):
    """Apply image processing as in He et al:
    Scale augmentation, per-pixel mean subtraction, horizontal flips,
    and color augmentation

    He et al. Deep Residual Learning for Image Recognition. arXiv, 2015.
    http://arxiv.org/abs/1512.03385

    K. Simonyan and A. Zisserman. Very deep convolutional networks
    for large-scale image recognition. In ICLR, 2015.

    A. Krizhevsky, I. Sutskever, and G. Hinton. Imagenet classification
    with deep convolutional neural networks. In NIPS, 2012.
    """

    # img = load_image(file_path)
    img = misc.imread(file_path)
#    img = cv2.imread(file_path)
    (height, width, channels) = img.shape

    # Resize the image with so that shorter side is between 256-480
    scale = float(random.sample(range(RESIZE_SIZE[0], 
                                      RESIZE_SIZE[1]+1), 1)[0]) / min(height, width)
    img = misc.imresize(img, size=scale)

    (height, width, channels) = img.shape

    # Take a randomly located crop of size CROP_SIZE
    row = random.sample(range(height - CROP_SIZE[0]), 1)[0]
    col = random.sample(range(width - CROP_SIZE[1]), 1)[0]

    crop_img = img[row:row + CROP_SIZE[0], col:col + CROP_SIZE[1], :]
    return crop_img


def train_generator(im_files, y_train, batch_size=50):
    """Read in the images and yield a training tuple."""
    while True:
        ind_selection = random.sample(range(len(im_files)), batch_size)
        im_selection, y_train_selection = im_files[ind_selection], y_train[ind_selection]

        images_tensor = [resnet_image_processing(im_file) for im_file in im_selection]

        # Convert list into a tensor
        images_tensor = list_to_tensor(images_tensor)

        # Subtract mean on a per-batch basis
        images_tensor -= images_tensor.mean()

        yield (images_tensor, y_train_selection)


def validation_generator(im_files, y_valid, batch_size=50):
    # TODO: zip im_files, y_train
    for index_selection in grouper(list(range(len(im_files))), batch_size):
        print(index_selection)
        im_selection, y_valid_selection = im_files[index_selection], y_valid[index_selection]

        images_tensor = [resnet_image_processing(im_file) for im_file in im_selection]

        # Convert list into a tensor
        images_tensor = list_to_tensor(images_tensor)

        # Subtract mean on a per-batch basis
        images_tensor -= images_tensor.mean()

        yield (images_tensor, y_valid_selection)


def test_generator(im_files, batch_size=50):
    for im_selection in grouper(im_files, batch_size):
        images_tensor = [resnet_image_processing(im_file) for im_file in im_selection]
        images_tensor = list_to_tensor(images_tensor)

        # Subtract mean on a per-batch basis
        images_tensor -= images_tensor.mean()

        yield images_tensor


if __name__ == '__main__':
    # import os
    # from scipy import misc
    # import cv2
    import pickle

    # imfile = os.path.join('dataset', 'c0.jpg')
    # img = misc.imread(imfile)
    #
    # img2 = cv2.imread(imfile, cv2.IMREAD_COLOR)

    train_index = valid_index = list(range(109))
    with open('data_files.pkl', 'rb') as f:
        X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids = pickle.load(f)

    X_train, y_train = X_train_raw[train_index], y_train_raw[train_index, ...]
    X_valid, y_valid = X_train_raw[valid_index], y_train_raw[valid_index, ...]

    tr_generator = test_generator(im_files=X_train, batch_size=50)
    # val_generator = validation_generator(im_files=X_valid, y_valid=y_train, batch_size=50)

    for _ in range(3):
        x2 = next(tr_generator)