# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 23:28:06 2023

@author: DELL
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 22:53:17 2023

@author: DELL

import matplotlib.pyplot as plt
X=stacked_images.reshape(-1,128,128)
idx=3
plt.imshow(X[idx,:],cmap='gray')
"""

import csv
import os
import numpy as np
import cv2

input_folder = 'C:/Users/DELL/signverify/BHSig260-Bengali_PNG'
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
        stacked_images = np.array(pixel_values_list)

        if "F" in filename:
            labels_list.append([label_forged])
        else:
            labels_list.append([label_genuine])

output_file = 'C:/Users/DELL/signverify/csvfiles/bengalids1_pixel_values.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(pixel_values_list)

labels_output_file = 'C:/Users/DELL/signverify/csvfiles/bengalids1_labels.csv'
with open(labels_output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(labels_list)
    
    
import matplotlib.pyplot as plt
X=stacked_images.reshape(-1,128,128)
idx=3
plt.imshow(X[idx,:],cmap='gray')    