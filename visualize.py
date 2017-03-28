import os
from math import sqrt, ceil
from random import randint

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.models import load_model
from keras.utils.visualize_util import plot


def visualize(model, layer_name, image, all_filters=False):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (200, 100))
    # Show scheme
    plot(model, to_file='visualize/scheme.png')

    # Show layer
    layer_out = Model(input=model.layers[0].input, output=model.get_layer(layer_name).output)
    out = layer_out.predict(np.array([image]))

    if all_filters:
        plt.imshow(out[0, :, :, 0:3])
        plt.axis('off')
        plt.savefig("visualize/filter.png", pad_inches=0)

    figure = plt.figure()
    x = int(ceil(sqrt(out.shape[3])))
    for i in range(out.shape[3]):
        sp = figure.add_subplot(x, x, i + 1)
        sp.axis('off')
        sp.imshow(out[0, :, :, i], cmap='gray')

    plt.axis('off')
    plt.savefig('visualize/plot.png', pad_inches=0)


def random_image(folder):
    files = os.listdir(folder + "/IMG/")
    index = randint(0, len(files))
    return cv2.imread(folder + "/IMG/" + 'center_2017_03_28_22_08_07_995.jpg')


model = load_model('model.h5')
visualize(model, 'color_space', random_image('samples/from_side'))