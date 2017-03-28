import csv

import cv2
import keras
import tensorflow as tf
from keras.layers import *
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = []
folders = ['samples/udacity', 'samples/bridge', 'samples/turning', 'samples/turning_2', 'samples/second_track']
angle_correction = 0.06

for folder in folders:
    with open(folder + '/driving_log.csv') as file:
        reader = csv.reader(file)
        for line in reader:
            # Filter out angles < 0.7
            if abs(float(line[3])) > 0.013:
                data.append(line)


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
                    left_path = "samples/udacity/" + sample[1]
                    right_path = "samples/udacity/" + sample[2]
                steering = float(sample[3])
                speed = float(sample[6])

                def add_image(path, angle_correction):
                    image = cv2.imread(path.replace(" ", ""))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (200, 100))
                    flipped = cv2.flip(image, 1)
                    images.append(image)
                    images.append(flipped)
                    measurements.append(steering + angle_correction)
                    measurements.append(-(steering + angle_correction))

                add_image(center_path, 0)
                add_image(left_path, angle_correction)
                add_image(right_path, -angle_correction)

            yield shuffle(np.array(images), np.array(measurements))


class CustomCallback(keras.callbacks.Callback):
    min_loss = float("inf")

    def on_batch_end(self, batch, logs={}):
        current_loss = float(logs.get("loss"))
        if self.min_loss > current_loss:
            self.min_loss = current_loss
            model.save('model.h5')

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
# model = load_model('model.h5')
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

train, valid = train_test_split(shuffle(data), test_size=0.1)
train_gen = generator(train)
valid_gen = generator(valid)
callback = CustomCallback()
history = model.fit_generator(train_gen,
                              samples_per_epoch=len(train) * 6,
                              validation_data=valid_gen,
                              nb_val_samples=len(valid) * 6,
                              nb_epoch=5,
                              verbose=1,
                              callbacks=[callback])

model.save('model.h5')
