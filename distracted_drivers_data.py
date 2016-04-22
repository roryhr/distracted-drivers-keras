import numpy as np
import random

try:
    import cv2
    OPEN_CV = True
except ImportError:
    from scipy import misc
    OPEN_CV = False


def load_image(imfile):
    if OPEN_CV:
        img = cv2.imread(filename=imfile)
        img = img[::-1]
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

    img = misc.imread(file_path)
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
        images_tensor = images_tensor.reshape(batch_size, 3, CROP_SIZE[0], CROP_SIZE[1])
        images_tensor = images_tensor.astype('float32')
        images_tensor -= self.im_mean

        yield {'input': images_tensor, 'output': self.get_target_labels(im_files)}

def test_image_generator(im_files, batch_size=30):
    """Read in the test images and yield a test dict."""
    for im_selection in grouper(im_files, batch_size):
        # Load images into a list
        images_tensor = [self.resnet_image_processing(im_file) for im_file in im_selection]
        photo_ids = [int(path_id.stem) for path_id in im_selection]

        # Convert list into a tensor
        images_tensor = np.stack(images_tensor)

        # Reshape tensor into Theano format
        images_tensor = images_tensor.reshape(batch_size, 3, CROP_SIZE[0], CROP_SIZE[1])
        images_tensor = images_tensor.astype('float32')
        images_tensor -= self.im_mean

        yield {'input': images_tensor}, photo_ids