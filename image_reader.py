import math
import numpy as np
import tensorflow as tf


crop_size = (480, 640)
image_size = (480, 640)
err = 4.9406564584124654e-324
img_mean = [116.190, 97.203, 92.318]


def image_rotate(img, lab):
    """
    Randomly rotates the image and label.

    Inputs:
        img: img to rotate.
        lab: lab to rotate
    Outputs:
        img: img after rotate.
        lab: lab after rotate
    """
    random_angle = tf.random_uniform([], minval=-math.pi/6, maxval=math.pi/6, dtype=tf.float32)
    lab = tf.cast(lab, dtype=tf.float32)
    img -= 255.
    lab -= 255.
    img = tf.contrib.image.rotate(img, random_angle, "BILINEAR")
    lab = tf.contrib.image.rotate(lab, random_angle, "NEAREST")
    img += 255.
    lab += 255.
    lab = tf.cast(lab, dtype=tf.uint8)
    return img, lab


def image_mirror(img, lab):
    """
    Randomly mirrors the image and label.

    Inputs:
        img: img to mirror.
        lab: lab to mirror.
    Outputs:
        img: img after mirror.
        lab: lab after mirror.
    """
    distort_left_right_random = tf.random_uniform([], 0, 1.0, dtype=tf.float32)
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    lab = tf.reverse(lab, mirror)
    return img, lab


def distort_color(image, color_ordering):
    """
    Randomly change color of the image.

    Inputs:
        img: img to change color.
    Outputs:
        img: img after change color.
    """
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
    elif color_ordering == 3:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 4:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 5:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 6:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
    elif color_ordering == 7:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 8:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
    elif color_ordering == 9:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    elif color_ordering == 10:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 11:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    elif color_ordering == 12:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
    elif color_ordering == 13:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 14:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
    elif color_ordering == 15:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    elif color_ordering == 16:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 17:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    elif color_ordering == 18:
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 19:
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 20:
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 21:
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    elif color_ordering == 22:
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_contrast(image, lower=0.5,upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    return image


def read_data_list(data_list):
    """
    Function to read data list
    """
    f = open(data_list, 'r')
    imgs = []
    labs = []
    for line in f:
        try:
            img = line.strip("\n").split(' ')[0]
            lab = line.strip("\n").split(' ')[1]
        except ValueError:
            img = lab = line.strip("\n")
        imgs.append(img)
        labs.append(lab)
    return imgs, labs


def read_images_from_disk(input_queue, shuffle, random_color, random_mirror, random_rotate, img_mean):
    """
    Read image, and label with optional pre-processing.
    """
    # Read files
    img_contents = tf.read_file(input_queue[0])
    lab_contents = tf.read_file(input_queue[1])
    # Decode files
    img = tf.image.decode_png(img_contents, channels=3)
    lab = tf.image.decode_png(lab_contents, channels=1)
    # RGB -> BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Expand in dimension 0 for resize
    img = tf.expand_dims(img, dim=0)
    lab = tf.expand_dims(lab, dim=0)
    # Train
    if shuffle:
        # Randomly rescale the image, normal and valid mask
        scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32)
        h_new = tf.to_int32(tf.multiply(tf.to_float(image_size[0]), scale))
        w_new = tf.to_int32(tf.multiply(tf.to_float(image_size[1]), scale))
        new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
        img = tf.image.resize_bilinear(img, new_shape)
        lab = tf.image.resize_nearest_neighbor(lab, new_shape)
    # Remove dimension 0
    img = tf.squeeze(img, axis=0)
    lab = tf.squeeze(lab, axis=0)
    # Train
    if shuffle:
        # Random pad and crop
        crop_h = crop_size[0]
        crop_w = crop_size[1]
        lab = tf.cast(lab, dtype=tf.float32)
        img = img - 255.
        lab = lab - 255.
        combined = tf.concat(axis=2, values=[img, lab]) 
        img_shape = tf.shape(img)
        combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, img_shape[0]), tf.maximum(crop_w, img_shape[1]))
        last_img_dim = tf.shape(img)[-1]
        last_lab_dim = last_img_dim + tf.shape(lab)[-1]
        combined_crop = tf.random_crop(combined_pad, [crop_h, crop_w, last_lab_dim])
        img = combined_crop[:, :, : last_img_dim]
        lab = combined_crop[:, :, last_img_dim :]
        img = img + 255.
        lab = lab + 255.
        lab = tf.cast(lab, dtype=tf.uint8)
    if random_mirror:
        img, lab = image_mirror(img, lab)
    if random_rotate:
        img, lab = image_rotate(img, lab)
    if random_color:
        img = distort_color(img, np.random.randint(24))
    # Extract mean.
    img -= img_mean
    # Set static shape so that tensorflow knows shape at compile time.
    img.set_shape((480, 640, 3))
    lab.set_shape((480, 640, 1))
    return img, lab


class Reader(object):
    """
    Generic Data Reader which reads images, normals and masks
    from the disk, and enqueues them into a TensorFlow queue.
    """

    def __init__(self, coord, data_list, shuffle=False, random_color=False, random_mirror=False, random_rotate=False, img_mean=img_mean):
        """
        Initialize a Reader.
        """
        self.coord = coord
        self.data_list = data_list
        self.img_list, self.lab_list = read_data_list(data_list)
        self.imgs = tf.convert_to_tensor(self.img_list, dtype=tf.string)
        self.labs = tf.convert_to_tensor(self.lab_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.imgs, self.labs], shuffle=shuffle)
        self.img, self.lab = read_images_from_disk(self.queue, shuffle, random_color, random_mirror, random_rotate, img_mean)

    def dequeue(self, num_elements):
        img_bat, lab_bat = tf.train.batch([self.img, self.lab], num_elements)
        return img_bat, lab_bat
