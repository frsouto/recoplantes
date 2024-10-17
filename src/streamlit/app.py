
import streamlit as st
from sections import presentation_sujet, presentation_donnees, analyse_donnees, preprocessing, modelisation, meilleur_modele, conclusion, demonstration
import altair as alt

# Configuration de la page : Titre, Favicon et Layout
st.set_page_config(
    page_title="Diagnostic des Plantes",
    page_icon="🌿",
    layout="centered",
)

# CSS pour un en-tête fixe, réactif, et un meilleur style général
st.markdown("""
    <style>
    /* En-tête stylisé avec position fixe */
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #3a5f3f;
        color: white;
        text-align: center;
        padding: 0 100px;
        font-size: clamp(1.2rem, 1.2rem + 1.5vw, 2.5rem);
        z-index: 100;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        height: 160px;
    }
    .header h1 {
        font-size: clamp(1.5rem, 1.5rem + 0.5vw, 2rem);
        margin: 0;
        line-height: 1.2;
    }
    .main .block-container {
        padding-top: 170px;
        max-width: 1000px;
        margin: auto;
    }
    /* Style pour le pied de page */
    footer {
        background-color: #333333;
        color: #f9f9f9;
        text-align: center;
        padding: 20px 0;
        margin-top: 30px;
        border-top: 1px solid #ccc;
    }
    footer p {
        margin: 0;
    }
    footer a {
        color: #ffcc00;
        text-decoration: none;
        transition: color 0.3s ease;
    }
    footer a:hover {
        color: #ff6600;
        text-decoration: underline;
    }
    /* Réduire la taille du texte sur les petits écrans */
    @media (max-width: 768px) {
        .header h1 {
            font-size: 1.5rem;
        }
        .header h3 {
            font-size: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Contenu de l'en-tête
header = """
<div class="header">
    <h1>Maladies des Plantes - Diagnostic rapide grâce au Deep Learning</h1>
</div>
"""

# Injecter l'en-tête dans l'application
st.markdown(header, unsafe_allow_html=True)

# Sidebar pour la navigation avec des emojis et un style simplifié
st.sidebar.markdown("## 🌐 Navigation")
option = st.sidebar.radio(
    "Choisissez une section :",
    [
        "🌱 Présentation du sujet", 
        "📊 Données", 
        "🔎 Analyse", 
        "⚙️ Préprocessing", 
        "🤖 Modèles", 
        "🏆 Meilleur modèle", 
        "📝 Conclusion", 
        "🚀 Démonstration"
    ]
)



# Contenu principal selon la sélection dans la barre latérale
if option == "🌱 Présentation du sujet":
    presentation_sujet.display()
elif option == "📊 Données":
    presentation_donnees.display()
elif option == "🔎 Analyse":
    analyse_donnees.display()
elif option == "⚙️ Préprocessing":
    preprocessing.display()
elif option == "🤖 Modèles":
    modelisation.display()
elif option == "🏆 Meilleur modèle":
    meilleur_modele.display()
elif option == "📝 Conclusion":
    conclusion.display()
elif option == "🚀 Démonstration":
    uploaded_image = st.file_uploader("Téléverser une image de feuille", type=["jpg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Image téléversée", use_column_width=True)
        with st.spinner("Modèle en cours d'exécution..."):
            demonstration.display(uploaded_image)

# Footer pour un style plus soigné
footer = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #333333;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #f9f9f9;
        box-shadow: 0 -1px 10px rgba(0,0,0,0.1);
    }
    </style>
    <div class="footer">
        <p>Projet réalisé par <a href="https://www.linkedin.com/in/tonprofil" target="_blank">Ton Nom</a> | 
        <a href="https://github.com/tonprofil" target="_blank">GitHub</a> | 
        <a href="https://linkedin.com/tonprofil" target="_blank">LinkedIn</a></p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
