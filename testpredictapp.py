# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:39:16 2023

@author: DELL
"""

from flask import Flask, render_template, request, redirect, url_for
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

@app.route('/verify', methods=['POST'])
def verify():
    uploaded_files = request.files.getlist('signature')  # Use getlist to handle multiple files

    for file in uploaded_files:
        if file.filename != '':
            # Save the uploaded image to the uploads folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

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
    Ownds80model = load_model('C:/Users/DELL/signverify/models/RevOwnds80')

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
            prediction = Ownds80model.predict(new_signature)
            predicted_label = np.argmax(prediction)

            if predicted_label == 0:
                result = "Forged"
            else:
                result = "Genuine"
                
            predicted_probabilities = prediction[0] * 100
            percent_genuine = predicted_probabilities[1]
            percent_forged = predicted_probabilities[0]
            absolute_path = os.path.abspath(file_path)

            # Check if the image file exists
            if os.path.exists(file_path):
                # Pass the data to the template
                return render_template('testuploadsuccess1.html', test_image_path=absolute_path, result=result, percent_genuine=percent_genuine, percent_forged=percent_forged,file_path=absolute_path)
            else:
                # Handle the case where the image file does not exist
                return render_template('testuploadsuccess1.html', result="Image file not found.")

    # Handle the case where no file was uploaded
    return render_template('testuploadsuccess.html', result="No image uploaded.")
if __name__ == '__main__':
    app.run(debug=True)
