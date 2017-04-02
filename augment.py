import cv2
import numpy as np


def add_random_shadow(image, add_l = 0.6):
    """
    Applies random shadow to some part of the image
    :param image: np.array with RGB image data
    :param add_l: float controlling amount of shadow
    :return: np.array with RGB image data and applied shadow
    """
    image_width = image.shape[1]
    image_height = image.shape[0]

    top_x = image_width * np.random.uniform()
    top_y = 0
    bot_y = image_height
    bot_x = image_width * np.random.uniform()
    image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    shadow_mask = np.zeros_like(image[:, :, 0])
    grid = np.mgrid[0:image_height, 0:image_width]
    X_m = grid[0]
    Y_m = grid[1]
    shadow_mask[(X_m - top_y) * (bot_x - top_x) - (bot_y - top_y) * (Y_m - top_x) >= 0] = 1

    cond = shadow_mask == np.random.randint(2)
    image_hls[:, :, 1][cond] = image_hls[:, :, 1][cond] * add_l

    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def augment_lightness(image, add_l=0.6):
    """
    Applies random brightness to the image
    :param image: np.array with RGB image data
    :param add_l: float controlling max amount of lightness removed
    :return: np.array with darkened RGB image data
    """
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls_image[:, :, 1] = hls_image[:, :, 1] * np.random.rand() * add_l
    return cv2.cvtColor(hls_image, cv2.COLOR_HLS2RGB)


def random_shift(image, max_shift=20):
    """
    Applies random shift
    :param image: np.array with RGB image data
    :param max_shift: int with max shift in pixels
    :return: np.array with shifted RGB data, actual shift in pixels
    """
    shift = np.random.randint(max_shift * 2) - max_shift
    pad_row = image[:, 0, :] if shift > 0 else image[:, -1, :]
    new_image = np.empty_like(image)
    # print("Shift is: {}".format(shift))
    for i in range(0, image.shape[1]):
        if (shift >= 0 and i < shift) or (shift < 0 and image.shape[1] + shift < i):
            # print("Row {} is padded".format(i))
            new_image[:, i, :] = pad_row
        elif i - shift < image.shape[1]:
            # print("Row {} is taken from row {}".format(i, i - shift))
            new_image[:, i, :] = image[:, i - shift, :]
    return new_image, shift
