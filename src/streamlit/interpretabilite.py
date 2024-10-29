import streamlit as st

# Configuration de la page - DOIT ÊTRE EN PREMIER
st.set_page_config(layout="wide", page_title="Classification d'Images IA")

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

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("utils/styles.css")

@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(model_name)

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


# Container principal avec marge
with st.container():
    # Titre principal avec style personnalisé
    st.markdown("""
        <div class='main-title'>
            <h1>Classification d'Image avec Visualisation Interprétable</h1>
        </div>
    """, unsafe_allow_html=True)

    # Création des colonnes avec des ratios ajustés
    col_actions, col_results, col_interpretability = st.columns([1, 2, 2])

    with col_actions:
        st.markdown("""
            <div class='section-title'>
                <h2>Configuration</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Sélection du modèle avec style amélioré
        model_choice = st.selectbox(
            "📊 Sélection du modèle",
            ["CNN Maison", "Mobilenet1FineTuned", "Mobilenet1Augmented"],
            index=1,
            help="Choisissez le modèle de classification à utiliser"
        )

        # Chargement du modèle selon le choix
        if model_choice == "CNN Maison":
            model = load_model('models/CNNFirstComplet.keras')
            last_conv_layer_name = 'conv2d_7'
        elif model_choice == "Mobilenet1FineTuned":
            model = load_model('models/MobileNetV1_finetuned.keras')
            last_conv_layer_name = 'conv_pw_13_relu'
        else:
            model = load_model('models/MobileNetV1_augmented.keras')
            last_conv_layer_name = 'conv_pw_13_relu'

        # Sélection de la méthode d'interprétabilité
        st.markdown("#### 🔍 Méthode d'interprétabilité")
        option = st.radio(
            "",
            ('Grad-CAM', 'LIME'),
            index=0,
            help="Choisissez la méthode de visualisation"
        )

        # Upload de l'image avec style amélioré
        st.markdown("#### 📤 Charger une image")
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            help="Formats supportés: JPG, JPEG, PNG"
        )

    with col_results:
        if uploaded_file is not None:
            st.markdown("""
                <div class='section-title'>
                    <h2>Analyse de l'image</h2>
                </div>
            """, unsafe_allow_html=True)
            
            img = Image.open(uploaded_file)
            img = img.resize(img_size := (224, 224))

            # Expander pour les ajustements avec style amélioré
            with st.expander("⚙️ Paramètres d'image"):
                col1, col2 = st.columns(2)
                with col1:
                    brightness = st.slider("🔆 Luminosité", 0.5, 2.0, 1.0)
                    rotation_angle = st.slider("🔄 Rotation", 0, 360, 0)
                with col2:
                    blur_intensity = st.slider("🌫️ Flou", 0, 5, 0)

                # Appliquer les transformations
                from PIL import ImageEnhance , ImageFilter
                img = ImageEnhance.Brightness(img).enhance(brightness)
                if blur_intensity > 0:
                    img = img.filter(ImageFilter.GaussianBlur(blur_intensity))
                img = img.rotate(rotation_angle, expand=True)
                img = img.resize(img_size)

            # Affichage de l'image avec style
            st.image(img, caption='Image à analyser', use_column_width=True)
            
            # Bouton de prédiction stylisé
            if st.button("🚀 Lancer l'analyse", key='launch_prediction'):
                with st.spinner('Analyse en cours...'):
                    start_time = time.time()
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0) / 255.0
                    predictions = model.predict(img_array)
                    end_time = time.time()

                    # Affichage des résultats
                    predicted_class_index = tf.argmax(predictions[0])
                    predicted_class_name = class_names[predicted_class_index.numpy()]
                    
                    # Métriques dans des colonnes
                    metric1, metric2 = st.columns(2)
                    with metric1:
                        st.metric("Classe prédite", predicted_class_name)
                    with metric2:
                        st.metric("Temps d'analyse", f"{(end_time - start_time):.2f}s")

                    # Graphique des probabilités
                    st.markdown("#### 📊 Top 5 des prédictions")
                    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
                    fig = go.Figure(go.Bar(
                        x=[predictions[0][i] * 100 for i in top_5_indices],
                        y=[class_names[i] for i in top_5_indices],
                        orientation='h',
                        marker=dict(
                            color='rgba(74, 124, 89, 0.8)',
                            line=dict(color='rgba(74, 124, 89, 1.0)', width=2)
                        )
                    ))
                    fig.update_layout(
                        title=dict(
                            text="Probabilités de classification",
                            x=0.5,
                            xanchor='center'
                        ),
                        xaxis_title="Probabilité (%)",
                        yaxis=dict(autorange="reversed"),
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Colonne d'interprétabilité
    with col_interpretability:
        if uploaded_file is not None and 'predicted_class_name' in locals():
            st.markdown("""
                <div class='section-title'>
                    <h2>Visualisation</h2>
                </div>
            """, unsafe_allow_html=True)
            
            if option == 'Grad-CAM':
                heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, intensity=3)
                gradcam_image = overlay_gradcam(img, heatmap, alpha=0.7)
                st.image(gradcam_image, caption=f'Carte de chaleur Grad-CAM : {predicted_class_name}', use_column_width=True)
            else:
                with st.spinner('Génération de l\'explication LIME...'):
                    lime_img = explain_with_lime(img_array, model)
                    st.image(lime_img, caption=f'Zones d\'importance LIME : {predicted_class_name}', use_column_width=True)

# Nettoyage de la mémoire
gc.collect()