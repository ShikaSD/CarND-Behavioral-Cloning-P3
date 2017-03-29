import csv

import cv2
import keras
from keras.layers import *
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
import math
import random


def add_random_shadow(image):
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

    add_l = 1.5
    cond = shadow_mask == np.random.randint(2)
    image_hls[:, :, 1][cond] = image_hls[:, :, 1][cond] * add_l

    return cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)


def augment_brightness(image, add_l=1.5):
    hls_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls_image[:, :, 1] = hls_image[:, :, 1] * np.random.rand() * (add_l - 1) + 1
    return cv2.cvtColor(hls_image, cv2.COLOR_HLS2RGB)


def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            measurements = []
            for sample in batch_samples:
                center_path = sample[0]
                left_path = sample[1]
                right_path = sample[2]
                if not center_path.startswith('/'):
                    center_path = "samples/udacity/" + center_path
                    left_path = "samples/udacity/" + left_path
                    right_path = "samples/udacity/" + right_path
                steering = int(float(sample[3]) / 0.04) * 0.04

                def add_image(path, angle_correction):
                    image = cv2.imread(path.replace(" ", ""))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (200, 100))
                    flipped = cv2.flip(image, 1)
                    images.append(image)
                    measurements.append(steering + angle_correction)

                    if np.random.randint(2) > 0:
                        images.append(add_random_shadow(image))
                        measurements.append(steering + angle_correction)
                    else:
                        if np.random.randint(2) > 0:
                            augment_brightness(flipped)
                        images.append(flipped)
                        measurements.append(-(steering + angle_correction))

                angle_correction = 0.08

                add_image(center_path, 0)
                add_image(left_path, angle_correction)
                add_image(right_path, -angle_correction)

            yield shuffle(np.array(images), np.array(measurements))


class CustomCallback(keras.callbacks.Callback):
    min_loss = float("inf")
    model = None

    def __init__(self, model):
        self.model = model
        super().__init__()

    def on_batch_end(self, batch, logs={}):
        current_loss = float(logs.get("loss"))
        if self.min_loss > current_loss:
            self.min_loss = current_loss
            self.model.save('model.h5')


def cnn(load=False):
    if load:
        return load_model('model.h5')

    # NVIDIA model
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(100, 200, 3)))
    model.add(Cropping2D(cropping=((26, 8), (0, 0))))
    model.add(Conv2D(3, 1, 1))
    model.add(LeakyReLU(name='color_space'))
    model.add(Conv2D(24, 5, 5, subsample=(2, 2)))
    model.add(LeakyReLU(name='visual_1'))
    model.add(Conv2D(36, 5, 5, subsample=(2, 2)))
    model.add(LeakyReLU())
    model.add(Conv2D(64, 3, 3, subsample=(2, 1)))
    model.add(LeakyReLU())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    model.add(Conv2D(64, 3, 3))
    model.add(LeakyReLU())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(Dense(50))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(Dense(10))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.summary()

    return model

data = []
near_zero = []
folders = ['samples/udacity',
           'samples/bridge',
           'samples/turning',
           'samples/turning_2',
           'samples/second_track',
          # 'samples/first_track',
           'samples/from_side']

for folder in folders:
    with open(folder + '/driving_log.csv') as file:
        reader = csv.reader(file)
        for line in reader:
            # Filter out angles < 0.7
            if abs(float(line[3])) < 0.01:
                near_zero.append(line)
            else:
                data.append(line)

data += random.sample(near_zero, int(len(near_zero) * 0.1))

value, counts = np.unique([(float(i[3]) / 0.04) * 0.04 for i in data], return_counts=True)
plt.plot(value, counts)
plt.savefig('visualize/data.png')

model = cnn()
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

train, valid = train_test_split(shuffle(data), test_size=0.3)
train_gen = generator(train)
valid_gen = generator(valid)
callback = CustomCallback(model)
history = model.fit_generator(train_gen,
                              samples_per_epoch=len(train) * 6,
                              validation_data=valid_gen,
                              nb_val_samples=len(valid) * 6,
                              nb_epoch=5,
                              verbose=1,
                              callbacks=[callback])

model.save('model.h5')