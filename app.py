# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Define the folder where uploaded images will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

if __name__ == '__main__':
    app.run(debug=True)
