from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import os
from io import BytesIO
from PIL import Image
import base64
import keras
import mlflow
import mlflow.keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

# Initialisation de l'application Flask
app = Flask(__name__)

# Chemin vers le modèle initial
MODEL_PATH = os.path.join("models", "model.keras")

# Charger le modèle initial
model = load_model(MODEL_PATH)
model.make_predict_function()

# Fonction de prédiction du modèle
def model_predict(img, model):
    # Prétraitement de l'image pour la rendre compatible avec le modèle
    img = img.resize((128, 128))
    x = keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    # Prédiction avec le modèle chargé
    preds = model.predict(x)
    return preds

# Créer une structure de données pour stocker les données étiquetées
labeled_data = []

# Fonction pour ajouter des données étiquetées à la structure de données
def add_labeled_data(image_data, label):
    labeled_data.append((image_data, label))

# Fonction pour prétraiter les données étiquetées et les convertir en X_train et y_train
def preprocess_labeled_data(labeled_data):
    X_train = []
    y_train = []

    label_encoder = LabelEncoder()

    for data in labeled_data:
        image_data, label = data
        # Décodage de l'image depuis base64
        img_data = base64.b64decode(image_data)
        img = Image.open(BytesIO(img_data))
        img = img.resize((128, 128))
        x = keras.preprocessing.image.img_to_array(img)
        x = x / 255.0
        X_train.append(x)
        y_train.append(label)

    # Convertir les étiquettes en valeurs numériques
    y_train = label_encoder.fit_transform(y_train)

    return np.array(X_train), np.array(y_train)

# Fonction pour réentraîner le modèle avec les nouvelles données étiquetées
def train_model(X_train, y_train):
    input_shape = (128, 128, 3)
    num_classes = 2

    # Définir l'architecture du modèle
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    # Compiler le modèle
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Entraîner le modèle
    model.fit(X_train, y_train, batch_size=32, epochs=15)

    # Sauvegarder le modèle dans le répertoire spécifié
    model.save('models/model.keras')

    # Enregistrer les données d'entraînement dans MLflow
    with mlflow.start_run():
        mlflow.log_param("num_samples", len(X_train))
        mlflow.log_param("input_shape", X_train[0].shape)
        mlflow.keras.log_model(model, "initial_model")

# Page d'accueil de l'application
@app.route('/', methods=['GET'])
def home():
    # Afficher la page d'accueil de l'application
    return render_template('index.html')

# Page de résultat après avoir téléchargé une image
@app.route('/result', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        # Convertir l'image en base64 pour l'afficher dans le navigateur
        buffered_img = BytesIO(f.read())
        img = Image.open(buffered_img)
        base64_img = base64.b64encode(buffered_img.getvalue()).decode("utf-8")

        # Faire une prédiction sur l'image
        preds = model_predict(img, model)
        result = "Chien" if preds[0][0] < 0.5 else "Chat"
        
        # Afficher la page de résultat avec le résultat de la prédiction et l'image téléchargée
        return render_template('result.html', result=result, image_base64_front=base64_img)
    
    # Rediriger vers la page d'accueil si la méthode n'est pas POST
    return redirect('/')

# Page pour donner un feedback sur la prédiction
@app.route('/feedback', methods=['POST'])
def feedback():
    image_data = request.form['image_data']
    label = request.form['label']

    # Ajouter les données étiquetées à la structure de données
    add_labeled_data(image_data, label)

    # Réentraîner le modèle avec les nouvelles données étiquetées
    X_train, y_train = preprocess_labeled_data(labeled_data)
    train_model(X_train, y_train)

    # Rediriger vers la page d'accueil après avoir donné le feedback
    return redirect('/')

# Configuration de MLflow avec le port spécifié
mlflow.set_tracking_uri("http://localhost:5001")  # URL:Port pour MLflow
mlflow.set_experiment("CatDogClassification")  # Nom de l'expérience MLflow

# Point d'entrée de l'application Flask
if __name__ == '__main__':
    # Lancer l'application en mode debug
    app.run(debug=True)