import os
from math import sqrt, ceil
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.models import load_model
from augment import *


def visualize(model, layer_names, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 100))

    for layer_name in layer_names:
        # Show layer
        layer_out = Model(input=model.layers[0].input, output=model.get_layer(layer_name).output)
        out = layer_out.predict(np.array([image]))

        print(out.shape)
        for i in range(out.shape[3]):
            result = np.empty((out.shape[1], out.shape[2], 3))
            result[:, :, 0] = out[0, :, :, i]
            result[:, :, 1] = out[0, :, :, i]
            result[:, :, 2] = out[0, :, :, i]
            result += 0.5
            cv2.imwrite("visualize/layer_{}_{}.png".format(layer_name, i), result * 255)


def show_augmentation(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 100))
    translated_image, shift = random_shift(image)
    cv2.imwrite("visualize/brightness.png", augment_lightness(image))
    cv2.imwrite("visualize/shadow.png", add_random_shadow(image))
    cv2.imwrite("visualize/translate.png", translated_image)
    cv2.imwrite("visualize/flip.png", cv2.flip(image, 1))


def random_image(folder):
    files = os.listdir(folder + "/IMG/")
    index = randint(0, len(files))
    return cv2.imread(folder + "/IMG/" + files[index])


model = load_model('model.h5')
# visualize(model, ['visual_1', 'color_space'], random_image('samples/from_side'))
show_augmentation(random_image('samples/from_side'))