import numpy as np
import random
from scipy import misc

try:
    import cv2
    OPEN_CV = True
except ImportError:
    OPEN_CV = False


def load_image(imfile):
    if OPEN_CV:
        img = cv2.imread(filename=imfile)
        img = img[:, :, ::-1]
    else:
        img = misc.imread(imfile)

    return img


CROP_SIZE = (224, 224)   # Size of final image passed into convolution network


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

    img = load_image(file_path)
    (height, width, channels) = img.shape

    # Resize the image with so that shorter side is between 256-480
    scale = float(random.sample(range(256, 480), 1)[0]) / min(height, width)
    img = misc.imresize(img, size=scale)

    (height, width, channels) = img.shape

    # Take a CROP_SIZE = 224x224 randomly selected crop
    row = random.sample(range(height - CROP_SIZE[0]), 1)[0]
    col = random.sample(range(width - CROP_SIZE[1]), 1)[0]

    crop_img = img[row:row+CROP_SIZE[0], col:col+CROP_SIZE[1], :]
    return crop_img


def graph_train_generator(im_files, augment_data=True):
    """Read in the images and yield a training dict."""
    while True:
        images_tensor = [resnet_image_processing(im_file) for im_file in im_files]

        # Convert list into a tensor
        images_tensor = np.stack(images_tensor)

        # Reshape tensor into Theano format
        images_tensor = images_tensor.reshape(-1, 3, CROP_SIZE[0], CROP_SIZE[1])
        images_tensor = images_tensor.astype('float32')

        # Subtract mean on a per-batch basis
        images_tensor -= images_tensor.mean()

        yield images_tensor


if __name__ == '__main__':
    import os
    from scipy import misc
    import cv2

    imfile = os.path.join('dataset', 'c0.jpg')
    img = misc.imread(imfile)

    img2 = cv2.imread(imfile, cv2.IMREAD_COLOR)

