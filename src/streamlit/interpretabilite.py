import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import plotly.graph_objects as go
import time
import gc
import pandas as pd
from utils.classes import class_names
from utils.config_loader import load_model_configs

# Configuration de la page - DOIT ÊTRE EN PREMIER
st.set_page_config(layout="wide", page_title="Classification d'Images IA")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("utils/styles.css")

@st.cache_resource
def load_model(model_name):
    return tf.keras.models.load_model(model_name)


MODEL_CONFIGS = load_model_configs("utils/model_configs.json")

def preprocess_image(img, model_choice, for_lime=False):
    """Prétraite l'image selon le modèle choisi."""
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if MODEL_CONFIGS[model_choice]["preprocessing"] == "simple":
        return img_array / 255.0
    else:  # Pour MobileNet
        if for_lime:
            # Pour LIME, on normalise simplement entre 0 et 1
            return img_array / 255.0
        else:
            # Pour la prédiction normale, on utilise le prétraitement MobileNet
            return tf.keras.applications.mobilenet.preprocess_input(img_array)

def create_lime_predictor(model, model_choice):
    """Crée une fonction de prédiction adaptée pour LIME."""
    def predictor(images):
        # Convertir le batch d'images en float et le normaliser
        processed_images = []
        for img in images:
            # Assurer que l'image est en float et normalisée entre 0 et 1
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            
            # Appliquer le prétraitement spécifique au modèle
            if MODEL_CONFIGS[model_choice]["preprocessing"] == "mobilenet":
                # Convertir de [0,1] au format attendu par MobileNet
                img = img * 255.0
                img = tf.keras.applications.mobilenet.preprocess_input(img)
            
            processed_images.append(img)
        
        batch = np.stack(processed_images)
        return model.predict(batch)
    
    return predictor

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, intensity=4):
    """Génère la heatmap Grad-CAM."""
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        pred_index = pred_index.numpy()
        if isinstance(pred_index, np.ndarray):
            if pred_index.size == 1:
                pred_index = pred_index.item()
            else:
                pred_index = pred_index[0]
                
        class_channel = predictions[0][pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = heatmap * intensity
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.3):
    """Superpose la heatmap Grad-CAM sur l'image originale."""
    img = img.resize((224, 224))
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

def explain_with_lime(img_array, model, model_choice):
    """Génère l'explication LIME."""
    # Créer un prédicteur adapté pour LIME
    predictor = create_lime_predictor(model, model_choice)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array[0].astype('double'),  # Assurer que l'image est en double précision
        predictor,
        top_labels=1,
        hide_color=0,
        num_samples=15
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=True
    )
    
    # S'assurer que l'image est normalisée entre 0 et 1
    if temp.max() > 1.0:
        temp = temp / 255.0
        
    return mark_boundaries(temp, mask)

def predict_with_model(img, model, model_choice):
    """Effectue la prédiction avec le modèle spécifié."""
    start_time = time.time()
    processed_img = preprocess_image(img, model_choice)
    predictions = model.predict(processed_img)
    end_time = time.time()
    
    return predictions, end_time - start_time

def display_results(predictions, analysis_time, class_names):
    """Affiche les résultats de la prédiction."""
    predicted_class_index = tf.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index.numpy()]
    
    metric1, metric2 = st.columns(2)
    with metric1:
        st.metric("Classe prédite", predicted_class_name)
    with metric2:
        st.metric("Temps d'analyse", f"{analysis_time:.2f}s")
    
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
        title=dict(text="Probabilités de classification", x=0.5, xanchor='center'),
        xaxis_title="Probabilité (%)",
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    st.plotly_chart(fig, use_container_width=True)
    return predicted_class_name

def generate_interpretability(img, model, model_choice, option, predicted_class_name):
    """Génère la visualisation d'interprétabilité selon la méthode choisie."""
    processed_img = preprocess_image(img, model_choice)
    
    if option == 'Grad-CAM':
        heatmap = make_gradcam_heatmap(
            processed_img,
            model,
            MODEL_CONFIGS[model_choice]["last_conv_layer"],
            intensity=3
        )
        gradcam_image = overlay_gradcam(img, heatmap, alpha=0.7)
        st.image(
            gradcam_image,
            caption=f'Carte de chaleur Grad-CAM : {predicted_class_name}',
            use_column_width=True
        )
    else:
        with st.spinner('Génération de l\'explication LIME...'):
            processed_img = preprocess_image(img, model_choice, for_lime=True)
            lime_img = explain_with_lime(processed_img, model, model_choice)
            st.image(
                lime_img,
                caption=f'Zones d\'importance LIME : {predicted_class_name}',
                use_column_width=True,
                clamp=True  # Ajouter clamp=True pour éviter l'erreur
            )

def compare_models(img, selected_models, class_names, interpretation_method='Grad-CAM'):
    """
    Compare les prédictions de plusieurs modèles sur une même image avec choix de la méthode d'interprétabilité.
    
    Args:
        img: Image PIL à analyser
        selected_models: Liste des modèles sélectionnés
        class_names: Liste des noms de classes
        interpretation_method: 'Grad-CAM' ou 'LIME'
    """
    cols = st.columns(len(selected_models))
    results = {}
    
    # Préparer l'image une seule fois
    img_display = img.resize((224, 224))
    
    for idx, model_name in enumerate(selected_models):
        with cols[idx]:
            st.markdown(f"##### {model_name}")
            
            # Charger le modèle
            model = load_model(MODEL_CONFIGS[model_name]["path"])
            
            # Faire la prédiction
            predictions, analysis_time = predict_with_model(img_display, model, model_name)
            pred_index = tf.argmax(predictions[0])
            confidence = predictions[0][pred_index] * 100
            
            # Afficher les métriques
            st.metric("Classe", class_names[pred_index])
            st.metric("Confiance", f"{confidence:.1f}%")
            st.metric("Temps", f"{analysis_time:.3f}s")
            
            # Générer la visualisation d'interprétabilité
            if interpretation_method == 'Grad-CAM':
                heatmap = make_gradcam_heatmap(
                    preprocess_image(img_display, model_name),
                    model,
                    MODEL_CONFIGS[model_name]["last_conv_layer"],
                    intensity=3
                )
                viz_image = overlay_gradcam(img_display, heatmap, alpha=0.7)
                st.image(viz_image, caption="Grad-CAM", use_column_width=True)
            else:  # LIME
                with st.spinner(f'Génération LIME pour {model_name}...'):
                    processed_img = preprocess_image(img_display, model_name, for_lime=True)
                    lime_img = explain_with_lime(processed_img, model, model_name)
                    st.image(
                        lime_img,
                        caption="LIME",
                        use_column_width=True,
                        clamp=True
                    )
            
            # Stocker les résultats
            results[model_name] = {
                "class": class_names[pred_index],
                "confidence": confidence,
                "time": analysis_time
            }
    
    return results

def main():
    # Container principal avec marge
    with st.container():
        # Titre principal
        st.markdown("""
            <div class='main-title'>
                <h1>Classification d'Image avec Comparaison de Modèles</h1>
            </div>
        """, unsafe_allow_html=True)

        # Paramètres dans la sidebar
        with st.sidebar:
            st.markdown("### 📊 Configuration")
            
            # Multi-sélection des modèles
            selected_models = st.multiselect(
                "Sélectionner les modèles à comparer",
                list(MODEL_CONFIGS.keys()),
                default=["CNN Maison", "Mobilenet1FineTuned"],
                help="Sélectionnez au moins un modèle"
            )
            
            # Sélection de la méthode d'interprétabilité
            interpretation_method = st.radio(
                "🔍 Méthode d'interprétabilité",
                ('Grad-CAM', 'LIME'),
                help="Choisissez la méthode de visualisation"
            )
            
            if interpretation_method == 'LIME':
                st.warning("⚠️ L'analyse LIME peut prendre plus de temps")
            
            # Vérification du nombre de modèles sélectionnés
            if len(selected_models) == 0:
                st.warning("⚠️ Veuillez sélectionner au moins un modèle")
            elif len(selected_models) > 3:
                st.warning("⚠️ Veuillez sélectionner maximum 3 modèles")
            
            # Upload de l'image
            st.markdown("### 📤 Image à analyser")
            uploaded_file = st.file_uploader(
                "",
                type=["jpg", "jpeg", "png"],
                help="Formats supportés: JPG, JPEG, PNG"
            )
            
            # Paramètres d'image
            if uploaded_file:
                st.markdown("### ⚙️ Paramètres d'image")
                brightness = st.slider("🔆 Luminosité", 0.5, 2.0, 1.0)
                rotation_angle = st.slider("🔄 Rotation", 0, 360, 0)
                blur_intensity = st.slider("🌫️ Flou", 0, 5, 0)

        # Zone principale
        if uploaded_file and 0 < len(selected_models) <= 3:
            # Charger et prétraiter l'image
            img = Image.open(uploaded_file)
            
            # Appliquer les transformations
            img = ImageEnhance.Brightness(img).enhance(brightness)
            if blur_intensity > 0:
                img = img.filter(ImageFilter.GaussianBlur(blur_intensity))
            if rotation_angle != 0:
                img = img.rotate(rotation_angle, expand=True)
            
            # Afficher l'image originale
            st.markdown("### 🖼️ Image analysée")
            st.image(img, caption='Image source', use_column_width=True)
            
            # Bouton d'analyse avec indication de la méthode
            if st.button(f"🚀 Lancer l'analyse comparative ({interpretation_method})", type="primary"):
                st.markdown("### 📊 Résultats de la comparaison")
                
                with st.spinner(f'Analyse comparative avec {interpretation_method} en cours...'):
                    # Lancer la comparaison des modèles avec la méthode sélectionnée
                    results = compare_models(img, selected_models, class_names, interpretation_method)
                
                # Afficher un tableau récapitulatif
                st.markdown("### 📋 Récapitulatif")
                df_results = pd.DataFrame.from_dict(results, orient='index')
                st.dataframe(
                    df_results.style.format({
                        'confidence': '{:.1f}%',
                        'time': '{:.3f}s'
                    }),
                    use_container_width=True
                )

if __name__ == "__main__":
    main()
