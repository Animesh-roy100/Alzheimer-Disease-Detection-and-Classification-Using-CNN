from __future__ import division, print_function
import sys
import os
import glob
import re

# Keras
from keras.models import load_model
from keras_preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app

app = Flask(__name__)

model  = load_model('weight.h5')

print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    verdict = ""
    p=np.argmax(model.predict(x))
    if p==0:
        verdict = "Mild Dementia"
    elif p==1:
        verdict = "Moderate Dementia"
    elif p==2:
        verdict = "Non Dementia"
    else:
        verdict = "Very Mild Dementia"
        
    return verdict

# img_path = "D:/app/uploads/26.jpg"

# img = image.load_img(img_path, target_size=(224, 224))

# x = image.img_to_array(img)
# x = np.expand_dims(x,axis=0)

# p=np.argmax(model.predict(x))
# if p==0:
#     print("Predicted Image: Mild Dementia")
# elif p==1:
#     print("Predicted Image: Moderate Dementia")
# elif p==2:
#     print("Predicted Image: Non Dementia")
# else:
#     print("Predicted Image: Very Mild Dementia")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)

        print(basepath)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        print(preds)
        return preds
    return None


if __name__ == '__main__':
   app.run(debug=True)