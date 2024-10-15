import streamlit as st
from PIL import Image
import tensorflow as tf

def display():
    st.title("Démonstration du modèle")
    st.write("Uploader une image pour obtenir une prédiction")

    # Chargement du modèle (remplacer par le chemin correct)
    model = tf.keras.models.load_model('models/CNNFirstComplet.keras')

    uploaded_file = st.file_uploader("Choisir une image...", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image chargée', use_column_width=True)

        # Prétraitement et prédiction
        img_array = preprocess_image(image)  # Fonction de preprocessing à définir
        prediction = model.predict(img_array)
        st.write(f"Prédiction du modèle : {prediction}")
