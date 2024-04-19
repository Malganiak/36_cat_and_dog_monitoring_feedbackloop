# 36_cat_and_dog_monitoring_feedbackloop

Ce projet est une application Flask qui met en œuvre un système de classification d'images entre chats et chiens. Voici un aperçu des fonctionnalités principales de l'application :

Importation des bibliothèques nécessaires, y compris Flask pour le développement web, numpy pour le traitement des tableaux, et des modules de Keras et scikit-learn pour le modèle de classification.

Initialisation de l'application Flask.

Chargement d'un modèle de classification d'images pré-entraîné depuis un fichier.

Définition de fonctions pour la prédiction d'images, l'ajout de données étiquetées, le prétraitement des données étiquetées, et le réentraînement du modèle avec les nouvelles données.

Définition des routes web de l'application :
'/' pour la page d'accueil.
'/result' pour la page de résultat après avoir téléchargé une image. Cette route prend en charge l'envoi d'une image, la prédiction de cette image, puis affiche le résultat et l'image téléchargée.
'/feedback' pour permettre aux utilisateurs de fournir des commentaires sur la prédiction. Ces données sont utilisées pour réentraîner le modèle avec les nouvelles étiquettes fournies.

Configuration de MLflow pour le suivi des expériences de formation de modèle.

Lancement de l'application Flask.


Essentiellement, l'application permet aux utilisateurs de télécharger une image, puis elle prédit si l'image contient un chat ou un chien. Si la prédiction est incorrecte, l'utilisateur peut fournir un retour d'information pour améliorer le modèle.
