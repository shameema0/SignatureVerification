# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:12:40 2023

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

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float32') / 255.0
    return image

X=np.loadtxt(r'C:/Users/DELL/signverify/csvfiles/ownds80.csv', delimiter=',')
Y=np.loadtxt(r'C:/Users/DELL/signverify/csvfiles/owndslab80.csv')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
#X_train=np.loadtxt('/content/drive/MyDrive/cedrealpix.csv', delimiter=',')
#Y_train=np.loadtxt('/content/drive/MyDrive/cedreallabb.csv')
#X_test =np.loadtxt('/content/drive/MyDrive/cedrealpix.csv', delimiter=',')
#Y_test=np.loadtxt('/content/drive/MyDrive/cedreallabb.csv')
X_train = X_train.reshape(X_train.shape[0], target_size[0], target_size[1], 1)
X_test = X_test.reshape(X_test.shape[0], target_size[0], target_size[1], 1)

Y_train = to_categorical(Y_train, num_classes=2)
Y_test = to_categorical(Y_test, num_classes=2)

#print(X_train[:10])
#print(Y_train[:10])
#print(X_test.shape)
#print(Y_test.shape)

Ownds80model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

Ownds80model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

Ownds80model.fit(X_train, Y_train, epochs=30, batch_size=64, validation_data=(X_test, Y_test))

loss, accuracy = Ownds80model.evaluate(X_test, Y_test)
print("Test accuracy:", accuracy)

Ownds80model.save('C:/Users/DELL/signverify/models/RevOwnds80')

