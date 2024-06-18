# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 21:15:55 2023

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

def cnn():
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

    
    #X=np.loadtxt(r'C:/Users/DELL/signverify/csvfiles/ownds80.csv', delimiter=',')
    #Y=np.loadtxt(r'C:/Users/DELL/signverify/csvfiles/owndslab80.csv')

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train=np.loadtxt('C:/Users/DELL/signverify/csvfiles/FinFullds.csv', delimiter=',')
    Y_train=np.loadtxt('C:/Users/DELL/signverify/csvfiles/FinFulldslab.csv')
    X_test =np.loadtxt('C:/Users/DELL/signverify/csvfiles/FinFulldstest.csv', delimiter=',')
    Y_test=np.loadtxt('C:/Users/DELL/signverify/csvfiles/FinFulldstestlab.csv')
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

    Fulldsmodel.save('C:/Users/DELL/signverify/models/Fulldsmodel')



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
    Fulldsmodel = load_model('C:/Users/DELL/signverify/models/Fulldsmodel')

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
