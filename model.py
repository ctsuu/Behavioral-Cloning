import tensorflow as tf
from keras.layers import Dense, Flatten, Lambda, Activation, MaxPooling2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import utility

tf.python.control_flow_ops = tf

number_of_epochs = 20
num_train_images = 19200
num_val_images = 4200
learning_rate = 1e-4
keep_prob = 0.5
batch_size = 64


# The model is based on NVIDIA's "End to End Learning for Self-Driving Cars" paper
# Source:  https://arxiv.org/pdf/1604.07316.pdf
model = Sequential()
# First Normalize layer, credit to comma ai model
model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
# Color space conversion layer, credit to Vivek's model
model.add(Convolution2D(3, 1, 1, border_mode='same', name='color_conv'))
# Classic five convolutional, Nvidia model and additional maxpooling layers
model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))

model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu', subsample=(2, 2)))

model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='valid', activation='relu', subsample=(1, 1)))

model.add(Flatten())

# Next, five fully connected layers
model.add(Dense(1164, activation='relu'))

model.add(Dropout(keep_prob))
model.add(Dense(100, activation='relu'))

model.add(Dense(50, activation='relu'))

model.add(Dense(10, activation='relu'))

model.add(Dense(1))

model.summary()

#model.compile(optimizer=Adam(learning_rate), loss="mse",metrics=['accuracy'] )
model.compile(optimizer=Adam(learning_rate), loss="mse" )

# create two generators for training and validation
train_data_gen = utility.generate_train_batch()
validation_data_gen = utility.generate_val_batch()

history = model.fit_generator(train_data_gen,
                              samples_per_epoch=num_train_images,
                              nb_epoch=number_of_epochs,
                              validation_data=validation_data_gen,
                              nb_val_samples=num_val_images,
                              verbose=1)

# finally save our model and weights
utility.save_model(model)
