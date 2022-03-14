from numpy import mean, mod, pad
from numpy import std
import pickle
from matplotlib import pyplot as plt
from numpy.matrixlib import matrix
from sklearn.model_selection import KFold
from keras.models import Model, load_model
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization, Dropout,GaussianNoise
from keras.layers import Flatten
import tensorflow as tf
import cv2
from keras.initializers import glorot_normal, RandomNormal, Zeros
import time
from keras import regularizers, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback, EarlyStopping, LambdaCallback

def model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(56, 56, 1)))
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.2))

    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))

    model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
    model.add(MaxPooling2D(2,2))
    model.add(Dropout(0.25))
    model.add(GaussianNoise(0.05))

    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(GaussianNoise(0.05))
    model.add(Dense(260, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', matrix=['accuracy'])
    return model

