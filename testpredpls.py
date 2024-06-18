# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 21:10:22 2023

@author: DELL
"""

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os

app = Flask(__name__)

# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the folder where uploaded test signature images will be stored
TEST_UPLOAD_FOLDER = 'test_uploads'
app.config['TEST_UPLOAD_FOLDER'] = TEST_UPLOAD_FOLDER

@app.route('/upload.html')
def upload_page():
    return render_template('upload.html')

@app.route('/testuplaod.html')
def test_upload_page():
    return render_template('testuplaod.html')

@app.route('/')
def home():
    return render_template('upload.html')

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float32') / 255.0
    return image

import csv

def append_to_csv(data, csv_file):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data.flatten().tolist())

def append_label_to_csv(label, csv_file):
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([label])


@app.route('/verify', methods=['POST'])
def verify():
    uploaded_files = request.files.getlist('signature')  # Use getlist to handle multiple files

    for file in uploaded_files:
        if file.filename != '':
            # Save the uploaded image to the uploads folder
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            preprocessed_image = preprocess_image(file_path, target_size)  # Implement preprocess_image function as before

            # Append the preprocessed image to an existing CSV file
            append_to_csv(preprocessed_image, 'C:/Users/DELL/signverify/csvfiles/FinFullds.csv')  # Replace with your CSV file path

            # Append label '1' to another CSV file
            append_label_to_csv(1, 'C:/Users/DELL/signverify/csvfiles/FinFulldslab.csv') 
            #cnn()

    return render_template('uploadsuccess.html')



# Define a route to handle the test signature image upload
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
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model


target_size = (128, 128)

def preprocess_image(image_path, target_size):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype('float32') / 255.0
    return image



@app.route('/verify_test', methods=['POST'])
def verify_test():
    # Load the pre-trained model
    Fulldsmodel = load_model('C:/Users/DELL/signverify/models/CDBHFulldsmodel')

    if 'test_signature' in request.files:
        file = request.files['test_signature']
        if file.filename != '':
            # Save the uploaded test signature image to a folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['TEST_UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Preprocess the uploaded image
            new_signature = preprocess_image(file_path, target_size)
            new_signature = new_signature.reshape(1, target_size[0], target_size[1], 1)

            # Use your pre-trained model for prediction
            prediction = Fulldsmodel.predict(new_signature)
            predicted_label = np.argmax(prediction)

            if predicted_label == 0:
                result = "Forged"
            else:
                result = "Genuine"
                
            predicted_probabilities = prediction[0] * 100
            percent_genuine = predicted_probabilities[1]
            percent_forged = predicted_probabilities[0]

            # Serve the uploaded image through Flask
            return render_template('testuploadsuccess1.html', test_image_filename=filename, result=result, percent_genuine=percent_genuine, percent_forged=percent_forged)

    # Handle the case where no file was uploaded
    return render_template('testuploadsuccess.html', result="No image uploaded.")

# Add a route to serve uploaded test images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['TEST_UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
