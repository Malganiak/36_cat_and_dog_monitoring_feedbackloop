from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
from io import BytesIO
from PIL import Image
import base64

import keras

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "model.keras")

model = load_model(MODEL_PATH)
model.make_predict_function()

def model_predict(img, model):
    img = img.resize((128, 128))  # L'erreur est ici : Redimensionner l'image à la taille attendue
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/result', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        buffered_img = BytesIO(f.read())
        img = Image.open(buffered_img)

        base64_img = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

        preds = model_predict(img, model)
        result = "Chien" if preds[0][0] < 0.5 else "Chat"
        
        return render_template('result.html', result=result, image_base64_front=base64_img)
    
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
