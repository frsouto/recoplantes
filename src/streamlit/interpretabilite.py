# interpretabilite.py

import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import plotly.graph_objects as go
import time
import pandas as pd
from utils.classes import class_names
from utils.config_loader import load_model_configs
from components.image_processing import preprocess_image, make_gradcam_heatmap, overlay_gradcam, explain_with_lime
from components.model_utils import load_model

# Configuration de la page - DOIT ÊTRE EN PREMIER
st.set_page_config(layout="wide", page_title="Classification d'Images IA")

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("utils/styles.css")

@st.cache_resource
def cached_load_model(model_name):
    return load_model(model_name)

MODEL_CONFIGS = load_model_configs("utils/model_configs.json")

def predict_with_model(img, model, model_choice):
    """Effectue la prédiction avec le modèle spécifié."""
    start_time = time.time()
    processed_img = preprocess_image(img, model_choice, model_configs=MODEL_CONFIGS)
    predictions = model.predict(processed_img)
    end_time = time.time()
    
    return predictions, end_time - start_time

def display_results(predictions, analysis_time, class_names):
    """Affiche les résultats de la prédiction."""
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    
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
    processed_img = preprocess_image(img, model_choice, model_configs=MODEL_CONFIGS)
    
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
            processed_img = preprocess_image(img, model_choice, for_lime=True, model_configs=MODEL_CONFIGS)
            lime_img = explain_with_lime(processed_img, model, model_choice, model_configs=MODEL_CONFIGS)
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
            model = cached_load_model(MODEL_CONFIGS[model_name]["path"])
            
            # Faire la prédiction
            predictions, analysis_time = predict_with_model(img_display, model, model_name)
            pred_index = np.argmax(predictions[0])
            confidence = predictions[0][pred_index] * 100
            
            # Afficher les métriques
            st.metric("Classe", class_names[pred_index])
            st.metric("Confiance", f"{confidence:.1f}%")
            st.metric("Temps", f"{analysis_time:.3f}s")
            
            # Générer la visualisation d'interprétabilité
            if interpretation_method == 'Grad-CAM':
                heatmap = make_gradcam_heatmap(
                    preprocess_image(img_display, model_name, model_configs=MODEL_CONFIGS),
                    model,
                    MODEL_CONFIGS[model_name]["last_conv_layer"],
                    intensity=3
                )
                viz_image = overlay_gradcam(img_display, heatmap, alpha=0.7)
                st.image(viz_image, caption="Grad-CAM", use_column_width=True)
            else:  # LIME
                with st.spinner(f'Génération LIME pour {model_name}...'):
                    processed_img = preprocess_image(img_display, model_name, for_lime=True, model_configs=MODEL_CONFIGS)
                    lime_img = explain_with_lime(processed_img, model, model_name, model_configs=MODEL_CONFIGS)
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
