import csv
import random

import keras
from keras.layers import *
from keras.models import Sequential
from keras.models import load_model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from augment import *

augmented_images_per_sample = 6


def generator(samples, batch_size=128):
    """
    Generates data based on batch_size amount of samples
    :param samples: rows from CSV files read before
    :param batch_size: amount of samples rows taken
    :return: batch_size
    """
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

                # Round the angle to 0.04 (1/25)
                steering = 0.04 * round(float(sample[3])/0.04)

                def add_image(path, angle_correction):
                    image = cv2.imread(path.replace(" ", ""))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (200, 100))
                    images.append(image)
                    measurements.append(steering + angle_correction)

                    rand = np.random.randint(3)
                    if rand == 0:
                        images.append(cv2.flip(image, 1))
                        measurements.append(-(steering + angle_correction))
                    elif rand == 1:
                        images.append(add_random_shadow(image))
                        measurements.append(steering + angle_correction)
                    else:
                        images.append(augment_lightness(image))
                        measurements.append(steering + angle_correction)

                angle_correction = 0.08
                add_image(center_path, 0)
                add_image(left_path, angle_correction)
                add_image(right_path, -angle_correction)

            yield shuffle(np.array(images), np.array(measurements))


class CustomCallback(keras.callbacks.Callback):
    """
    Defines callback to save model with best loss on every batch in case
    learning will be interrupted
    """
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
    """
    Defines CNN
    :param load: flag indicating whether to load the model or create a new one
    :return: CNN model
    """
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
           # 'samples/second_track',
           'samples/first_track',
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
# Remove 90% of samples near zero, as there exists a large bias towards them
data += random.sample(near_zero, int(len(near_zero) * 0.1))

# Visualise final data with rounded angles
value, counts = np.unique([0.04 * round(float(i[3]) / 0.04) for i in data], return_counts=True)
plt.bar(value, counts, width=0.03)
plt.savefig('visualize/data.png')

model = cnn()
optimizer = Adam(lr=0.001)
model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

train, valid = train_test_split(shuffle(data), test_size=0.3)
train_gen = generator(train)
valid_gen = generator(valid)
callback = CustomCallback(model)
history = model.fit_generator(train_gen,
                              samples_per_epoch=len(train) * augmented_images_per_sample,
                              validation_data=valid_gen,
                              nb_val_samples=len(valid) * augmented_images_per_sample,
                              nb_epoch=5,
                              verbose=1,
                              callbacks=[callback])

model.save('model.h5')
