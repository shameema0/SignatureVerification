# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 00:35:43 2023

@author: DELL
"""

import csv
import os
import numpy as np
import cv2

input_folder = 'C:/Users/DELL/signverify/BHSig260-Hindi_PNG'
target_size = (128, 128)

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float32') / 255.0
    return image

pixel_values_list = []
labels_list = []
label_forged=0
label_genuine=1

for filename in os.listdir(input_folder):
        image_path = os.path.join(input_folder, filename)

        preprocessed_image = preprocess_image(image_path, target_size)

        pixel_values = preprocessed_image.flatten().tolist()

        pixel_values_list.append(pixel_values)

        if "F" in filename:
            labels_list.append([label_forged])
        else:
            labels_list.append([label_genuine])

output_file = 'C:/Users/DELL/signverify/csvfiles/hindids_pixel_values.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(pixel_values_list)

labels_output_file = 'C:/Users/DELL/signverify/csvfiles/hindids_labels.csv'
with open(labels_output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(labels_list)