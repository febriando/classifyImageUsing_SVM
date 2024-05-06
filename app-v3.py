# coding=utf-8
import sys
import os, shutil
import glob
import re

# Flask utils
from flask import Flask,flash, request, render_template,send_from_directory
from werkzeug.utils import secure_filename

# modeling image hsi
from modeling import rgb_to_hsi

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import graycomatrix, graycoprops
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import imutils.paths as path
import cv2
import numpy as np
# from texturing import histogram_indexing
import matplotlib.pyplot as plt

# Menentukan path ke folder dataset
PATH = 'penyakit2_aug/train'
imagePathsDataset = sorted(list(path.list_images(PATH)))

# Define a flask app
app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(24)

app.config['HSI_FOLDER'] = 'hsi_images'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ASSETS'] = 'assets'


@app.route('/uploads/<filename>')
def upload_img(filename):
    
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/hsi_images/<filename>')
def hsi_img(filename):
    
    return send_from_directory(app.config['HSI_FOLDER'], filename)

@app.route('/assets_file/<filename>')
def assets_file(filename):
    
    return send_from_directory(app.config['ASSETS'], filename)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

#labeling dan prepocessing data
Data = []
labels = []
print( os.listdir(PATH)) #ngambil semua isi dari folder train list kelas dari dataset.
for classname in os.listdir(PATH):
    for imagename in os.listdir(os.path.join(PATH, classname)):
        labels.append(classname) #iterasi nama kelas.

        img = cv2.imread(os.path.join(PATH, classname, imagename))
        imS = cv2.resize(img, (256, 256))
        konverIMG = cv2.cvtColor(imS, cv2.COLOR_BGR2HLS)
        arrt = konverIMG[:, :, 2]
        flattened_arrt = arrt.flatten()

        Data.append(flattened_arrt)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Membagi data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(Data, labels, test_size=0.2, random_state=42)

    if request.method == 'POST':
        
        # Get the file from post request
        f = request.files['file']
               
        # Train an SVM classifier
        svm_classifier = SVC(kernel='linear')
        svm_classifier.fit(X_train, y_train)
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        
        f.save(file_path)
        file_name=os.path.basename(file_path)
        
        imageSamplePaths = ''
        fileSample = file_path + imageSamplePaths
        dataNew = []
        img_new = cv2.imread(fileSample)
        imS_n = cv2.resize(img_new, (256, 256))
        konverIMG_n = cv2.cvtColor(imS_n, cv2.COLOR_BGR2HLS)
        arr = konverIMG_n[:, :, 2]
        flattened_arr = arr.flatten()
        dataNew.append (flattened_arr)
        
        label = svm_classifier.predict(dataNew)
            
        return render_template('predict.html', file_name=file_name, label=label[0])

if __name__ == '__main__':
        app.run(debug=True)
