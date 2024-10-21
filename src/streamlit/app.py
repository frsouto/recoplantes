import streamlit as st
from sections import (
    presentation, 
    data_overview, 
    exploratory_analysis, 
    preprocessing, 
    modeling, 
    best_model, 
    conclusion, 
    demo
)
import altair as alt



# Configuration de la page : Titre, Favicon et Layout
st.set_page_config(
    page_title="Reconnaissance de plantes",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state='expanded'
)

# CSS pour le style (amélioration des couleurs et du design général)
st.markdown("""
    <style>
    /* En-tête stylisé avec position fixe */
    .header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        background-color: #4A7C59;
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
        padding-top: 130px;
        max-width: 1200px;
        padding-left: 2%;
        padding-right: 2%;
        color: #333333;
    }
    /* Amélioration de la lisibilité et contrastes dans le footer */
    footer {
        background-color: #222;
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
    /* Personnalisation des boutons */
    .stButton button {
        background-color: #4A7C59;
        color: white;
        border-radius: 12px;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #3c6246;
        transition: background-color 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# Contenu de l'en-tête
st.header("Reconnaissance de plantes - Détection des maladies")

# Sidebar pour la navigation avec des emojis et un style simplifié
st.sidebar.title("## 🌐Navigation")

# Dictionnaire pour mapper les pages aux fonctions correspondantes
pages = {
    "🏰 Accueil": lambda: (st.title("Reconnaissance de plantes - Détection des maladies"),
                          st.subheader("DataScientest - Promotion Février 2024"),
                          st.write("Par Felipe Souto et Nicolas Papegaey")),
    "📄 Présentation": presentation.display,
    "📊 Données": data_overview.display,
    "🔍 Analyse exploratoire": exploratory_analysis.display,
    "⚙️ Preprocessing": preprocessing.display,
    "🤖 Modélisation": modeling.display,
    "🏆 Meilleur modèle": best_model.display,
    "📜 Conclusion": conclusion.display,
    "🚀 Démonstration": demo.display
}

# Navigation par le menu latéral
page = st.sidebar.radio("Aller à", list(pages.keys()))

# Affichage de la page sélectionnée
if page in pages:
    pages[page]()

# Footer
footer = """
    <style>
    .footer {
        position: relative;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: #222;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: #f9f9f9;
        box-shadow: 0 -1px 10px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    <div class="footer">
        <p>Projet réalisé par Felipe Souto et Nicolas Papegaey | DataScientest - Promotion Février 2024</p>
    </div>
"""
st.markdown(footer, unsafe_allow_html=True)
