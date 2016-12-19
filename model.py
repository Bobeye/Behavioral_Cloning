from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Merge, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.layers.noise import GaussianNoise
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adadelta, Adam
import pandas as pd
import numpy as np
import cv2
import json
import os
import h5py
import dataset

# load dataset
(X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset.dataset_for_train()

X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test  = X_test.astype('float32')
X_train -= 0.5
X_val -= 0.5
X_test  -= 0.5

input_shape = X_train.shape[1:]
print(input_shape, 'input shape')

batch_size = 8
nb_classes = 1
nb_epoch = 6

# Initiating the model
model = Sequential()
model.add(BatchNormalization(input_shape=input_shape))
model.add(Convolution2D(8,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(16,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(32,3,3, border_mode='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Activation('elu'))
model.add(Dense(48))
model.add(Dropout(0.5))
model.add(Activation('elu'))
model.add(Dense(nb_classes))

print (model.summary())

model.compile(loss='mean_squared_error',
              optimizer='Adam')


print('Using real-time data augmentation.')

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=2.0,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.,
            zoom_range=0.1,
            channel_shift_range=0.,
            fill_mode='nearest',
            cval=0.,
            horizontal_flip=False,
            vertical_flip=False)
datagen.fit(X_train)

try:
    model.fit_generator(datagen.flow(X_train, y_train,
                        batch_size=batch_size),
                        verbose=1,
                        nb_epoch=nb_epoch,
                        samples_per_epoch=X_train.shape[0],
                        validation_data=(X_val, y_val))
except KeyboardInterrupt:
    pass

# evaluate test data
print (model.evaluate(X_test, y_test, verbose=0))


# Save model as json file
json_string = model.to_json()

with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

    # save weights
    model.save_weights('./model.h5')
    print("Saved")