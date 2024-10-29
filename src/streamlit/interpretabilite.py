import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import plotly.graph_objects as go
import time
import gc
from utils.classes import class_names

# Lire le fichier CSS et l'inclure dans le markdown
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Charger le fichier CSS
load_css("utils/styles.css")


# Charger le modèle pré-entraîné
@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(model_name)

model = load_model('models/MobileNetV1_finetuned.keras')
last_conv_layer_name = 'conv_pw_13_relu'  # Nom de la dernière couche convolutionnelle de votre modèle
img_size = (224, 224)

# Fonction pour générer la heatmap Grad-CAM avec intensification
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, intensity=4):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])  # Avoid using .numpy() to maintain compatibility with TF operations
        # class_channel = predictions[0][pred_index]
        # Convertir en valeur scalaire, avec une vérification pour gérer les tableaux
        pred_index = pred_index.numpy()  # Convertir le tenseur en un tableau NumPy

        # Si `pred_index` est un tableau, prendre la première valeur s'il a plusieurs éléments
        if isinstance(pred_index, np.ndarray):
            if pred_index.size == 1:  # Si c'est un tableau de taille 1
                pred_index = pred_index.item()  # Extraire la valeur scalaire
            else:
                pred_index = pred_index[0]  # Utiliser le premier élément si plusieurs indices sont présents

        # Utiliser `pred_index` pour accéder à la bonne classe prédite
        class_channel = predictions[0][pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = heatmap * intensity
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Fonction pour superposer la heatmap Grad-CAM sur l'image originale
def overlay_gradcam(img, heatmap, alpha=0.3):
    # img is already an Image object, no need to open it again
    img = img.resize(img_size)
    img_array = np.array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("inferno")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = image.array_to_img(superimposed_img)
    
    return superimposed_img

# Fonction pour générer l'explication LIME
def explain_with_lime(img_array, model):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=1, hide_color=0, num_samples=15)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,
                                                hide_rest=True)
    lime_img = mark_boundaries(temp, mask)
    return lime_img

# Interface de l'application
st.title("Classification d'Image avec Visualisation Interprétable")

# Utilisation de trois colonnes pour les actions, les résultats et l'interprétabilité
col_actions, col_results, col_interpretability = st.columns([1, 2, 2])

with col_actions:
    # Ajout d'un sélecteur pour choisir le modèle
    model_choice = st.selectbox("Choisissez un modèle à charger :", ["CNN Maison", "Mobilenet1FineTuned","Mobilenet1Augmented"], index=1)

    # Charger dynamiquement le modèle basé sur la sélection
    if model_choice == "CNN Maison":
        model = load_model('models/CNNFirstComplet.keras')
        last_conv_layer_name = 'conv2d_7'  # Nom de la dernière couche convolutionnelle du modèle 1
    elif model_choice == "Mobilenet1FineTuned":
        model = load_model('models/MobileNetV1_finetuned.keras')
        last_conv_layer_name = 'conv_pw_13_relu'  # Nom de la dernière couche convolutionnelle du modèle 2
    elif model_choice == "Mobilenet1Augmented":
        model = load_model('models/MobileNetV1_augmented.keras')
        last_conv_layer_name = 'conv_pw_13_relu'  # Nom de la dernière couche convolutionnelle du modèle 2

    # Ajout d'un sélecteur pour choisir la méthode d'interprétabilité
    option = st.radio(
        "Choisissez la méthode d'interprétabilité que vous souhaitez afficher :",
        ('Grad-CAM', 'LIME'), index=0)

    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"], key="uploader1")

with col_results:
    if uploaded_file is not None:
        st.subheader("Image")
        img = Image.open(uploaded_file)
        img = img.resize(img_size)
        
        # Regrouper les ajustements dans un expander
        with st.expander("Ajustements de l'image"):
            # Ajout d'un slider pour ajuster la luminosité
            brightness = st.slider("Ajuster la luminosité", 0.5, 2.0, 1.0, key='brightness_slider_unique')

            # Appliquer la luminosité sélectionnée par l'utilisateur
            from PIL import ImageEnhance
            img = ImageEnhance.Brightness(img).enhance(brightness)

            # Ajout d'un slider pour ajuster le flou gaussien
            blur_intensity = st.slider("Ajuster l'intensité du flou", 0, 5, 0, key='blur_slider_unique')

            # Fonction pour appliquer le flou gaussien
            from PIL import ImageFilter
            if blur_intensity > 0:
                img = img.filter(ImageFilter.GaussianBlur(blur_intensity))

            # Ajout d'un slider pour faire pivoter l'image
            rotation_angle = st.slider("Faire pivoter l'image", 0, 360, 0, key='rotation_slider')

            # Appliquer la rotation de l'image
            img = img.rotate(rotation_angle, expand=True)
            # Redimensionner l'image après la rotation
            img = img.resize(img_size)
        
        
        # Appliquer la luminosité sélectionnée par l'utilisateur
        from PIL import ImageEnhance
        img = ImageEnhance.Brightness(img).enhance(brightness)
        st.image(img, caption='Image originale', use_column_width=True)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        if st.button("Suivant - Lancer la Prédiction", key='next_prediction'):
            start_time = time.time()
            predictions = model.predict(img_array)
            end_time = time.time()

            predicted_class_index = tf.argmax(predictions[0])
            predicted_class_name = class_names[predicted_class_index.numpy()]

            st.subheader("Résultats de la Classification")
            st.write(f"**Classe prédite :** {predicted_class_name}")
            st.write(f"**Temps de prédiction :** {end_time - start_time:.2f} secondes")

            # Affichage du graphique des probabilités des classes les plus probables
            st.subheader("Top 5 des probabilités de classification")
            top_5_indices = np.argsort(predictions[0])[-5:][::-1]
            top_5_probs = [predictions[0][i] for i in top_5_indices]
            top_5_class_names = [class_names[i] for i in top_5_indices]

            fig = go.Figure()
            fig.add_trace(go.Bar(x=top_5_probs, y=top_5_class_names, orientation='h', marker=dict(color='skyblue')))
            fig.update_layout(xaxis_title='Probabilité')
            fig.update_layout(title='Top 5 des classes prédites', yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            

with col_interpretability:
    if uploaded_file is not None and option is not None and 'predicted_class_name' in locals():
        st.subheader("Interprétabilité")
        if option == 'Grad-CAM':
            # Générer la heatmap Grad-CAM
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, intensity=3)
            gradcam_image = overlay_gradcam(img, heatmap, alpha=0.7)
            st.image(gradcam_image, caption=f'Image Grad-CAM pour {predicted_class_name}', use_column_width=True)
        elif option == 'LIME':
            # Générer l'explication LIME
            lime_img = explain_with_lime(img_array, model)
            st.image(lime_img, caption=f'Explication LIME pour {predicted_class_name}', use_column_width=True)
        
        # Libérer la mémoire après toutes les opérations
        if 'img' in locals():
            del img
        if 'img_array' in locals():
            del img_array
        if 'predictions' in locals():
            del predictions
        if 'gradcam_image' in locals():
            del gradcam_image
        if 'lime_img' in locals():
            del lime_img
        gc.collect()



