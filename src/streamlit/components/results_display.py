# results_display.py

import numpy as np
import streamlit as st
import plotly.graph_objects as go

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

def display_interpretability(option, img, model, model_choice, predicted_class_name, preprocess_image, make_gradcam_heatmap, overlay_gradcam, explain_with_lime, model_configs):
    """Génère et affiche la visualisation d'interprétabilité selon la méthode choisie."""
    processed_img = preprocess_image(img, model_choice, model_configs=model_configs)
    
    if option == 'Grad-CAM':
        heatmap = make_gradcam_heatmap(
            processed_img,
            model,
            model_configs[model_choice]["last_conv_layer"],
            intensity=3
        )
        gradcam_image = overlay_gradcam(img, heatmap, alpha=0.7)
        st.image(
            gradcam_image,
            caption=f'Carte de chaleur Grad-CAM : {predicted_class_name}',
            width=350
        )
    else:
        with st.spinner('Génération de l\'explication LIME...'):
            processed_img = preprocess_image(img, model_choice, for_lime=True, model_configs=model_configs)
            lime_img = explain_with_lime(processed_img, model, model_choice, model_configs=model_configs)
            st.image(
                lime_img,
                caption=f'Zones d\'importance LIME : {predicted_class_name}',
                width=350,
                clamp=True  # Ajouter clamp=True pour éviter l'erreur
            )
