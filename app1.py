# -*- coding: utf-8 -*-

# Import necessary modules
from flask import Flask, render_template, request, redirect, url_for
import os

# Initialize Flask
app = Flask(__name__)

# Define the folder where uploaded test signature images will be stored
TEST_UPLOAD_FOLDER = 'test_uploads'
app.config['TEST_UPLOAD_FOLDER'] = TEST_UPLOAD_FOLDER

@app.route('/upload.html')
def upload_page():
    return render_template('upload.html')

@app.route('/testuplaod.html')
def test_upload_page():
    return render_template('testuplaod.html')

# Define a route to display the upload form
@app.route('/')
def home():
    return render_template('testuplaod.html')  # Create a new HTML file for the test upload form

# Define a route to handle the test signature image upload
@app.route('/verify_test', methods=['POST'])
def verify_test():
    if 'test_signature' in request.files:
        file = request.files['test_signature']
        if file.filename != '':
            # Save the uploaded test signature image to the test_uploads folder
            file.save(os.path.join(app.config['TEST_UPLOAD_FOLDER'], file.filename))

    return render_template('testuploadsuccess.html')

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
