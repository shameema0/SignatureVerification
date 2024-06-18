# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:54:28 2023

@author: DELL
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical

target_size = (128, 128)
X=np.loadtxt(r'C:/Users/DELL/signverify/csvfiles/bengalids_pixel_values.csv', delimiter=',')
Y=np.loadtxt(r'C:/Users/DELL/signverify/csvfiles/bengalids_labels.csv')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#X_train=np.loadtxt('C:/Users/DELL/signverify/csvfiles/FinFullds.csv', delimiter=',')
#Y_train=np.loadtxt('C:/Users/DELL/signverify/csvfiles/FinFulldslab.csv')
#X_test =np.loadtxt('C:/Users/DELL/signverify/csvfiles/FinFulldstest.csv', delimiter=',')
#Y_test=np.loadtxt('C:/Users/DELL/signverify/csvfiles/FinFulldstestlab.csv')
X_train = X_train.reshape(X_train.shape[0], target_size[0], target_size[1], 1)
X_test = X_test.reshape(X_test.shape[0], target_size[0], target_size[1], 1)

Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)

#print(X_train[:10])
#print(Y_train[:10])
#print(X_test.shape)
#print(Y_test.shape)

Fulldsmodel = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

Fulldsmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Fulldsmodel.fit(X_train, Y_train, epochs=30, batch_size=64, validation_data=(X_test, Y_test))

loss, accuracy = Fulldsmodel.evaluate(X_test, Y_test)
print("Test accuracy:", accuracy)

Fulldsmodel.save('C:/Users/DELL/signverify/models/bengalimodel')