# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 22:37:24 2023

@author: DELL
"""

import csv
import os
import numpy as np
import cv2
from PIL import Image

input_folder = 'C:/Users/DELL/signverify/BHSig260-Hindi'
output_folder = 'C:/Users/DELL/signverify/BHSig260-Hindi_PNG'  # Output folder for PNG images

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

for folder_num in range(1, 161):
    folder_path = os.path.join(input_folder, str(folder_num))

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)

        # Check if the file is a TIFF image
        if image_path.lower().endswith(".tif") or image_path.lower().endswith(".tiff"):
            # Load the TIFF image
            tiff_image = Image.open(image_path)

            # Create a PNG filename by replacing the extension
            png_filename = os.path.splitext(filename)[0] + ".png"
            png_image_path = os.path.join(output_folder, png_filename)

            # Save the image as PNG
            tiff_image.save(png_image_path)

        # If the file is not a TIFF image, you can add code here to handle other formats

# Now, your TIFF images in the input folder will be converted to PNG format
