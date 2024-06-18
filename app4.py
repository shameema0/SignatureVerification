# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:57:05 2023

@author: DELL
"""

from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('upload1.html', success_message="")

@app.route('/upload1.html')
def upload_page():
    return render_template('upload1.html', success_message="")

@app.route('/testupload1.html')
def test_upload_page():
    return render_template('testupload1.html', success_message="")

@app.route('/verify', methods=['POST'])
def verify():
    uploaded_files = request.files.getlist('signature')  # Use getlist to handle multiple files

    for file in uploaded_files:
        if file.filename != '':
            # Save the uploaded image to the uploads folder
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    success_message = "Images uploaded successfully."

    return render_template('upload1.html', success_message=success_message)

@app.route('/verify_test', methods=['POST'])
def verify_test():
    uploaded_test_signature = request.files.get('test_signature')

    if uploaded_test_signature:
        if uploaded_test_signature.filename != '':
            # Save the uploaded test signature image to the test_uploads folder
            uploaded_test_signature.save(os.path.join(app.config['UPLOAD_FOLDER'], uploaded_test_signature.filename))

            success_message = "Test signature uploaded successfully."
        else:
            success_message = "No file selected for test signature upload."
    else:
        success_message = "Test signature upload failed."

    return render_template('testupload1.html', success_message=success_message)

if __name__ == '__main__':
    app.run(debug=True)
