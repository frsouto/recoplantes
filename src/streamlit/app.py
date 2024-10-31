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
    layout="wide",
    initial_sidebar_state='expanded'
)

# Ajout du logo et du titre dans l'en-tête
st.markdown(
    """
    <div class="header">
        <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADIAAAAyCAYAAAAeP4ixAAAACXBIWXMAAAsTAAALEwEAmpwYAAABNmlDQ1BQaG90b3Nob3AgSUNDIHByb2ZpbGUAAHjarY6xSsNQFEDPi6LiUCsEcXB4kygotupgxqQtRRCs1SHJ1qShSmkSXl7VfoSjWwcXd7/AyVFwUPwC/0Bx6uAQIYODCJ7p3MPlcsGo2HWnYZRhEGvVbjrS9Xw5+8QMUwDQCbPUbrUOAOIkjvjB5ysC4HnTrjsN/sZ8mCoNTIDtbpSFICpA/0KnGsQYMIN+qkHcAaY6addAPAClXu4vQCnI/Q0oKdfzQXwAZs/1fDDmADPIfQUwdXSpAWpJOlJnvVMtq5ZlSbubBJE8HmU6GmRyPw4TlSaqo6MukP8HwGK+2G46cq1qWXvr/DOu58vc3o8QgFh6LFpBOFTn3yqMnd/n4sZ4GQ5vYXpStN0ruNmAheuirVahvAX34y/Axk/96FpPYgAAACBjSFJNAAB6JQAAgIMAAPn/AACA6AAAUggAARVYAAA6lwAAF2/XWh+QAAAB/ElEQVR42uya7W3CMBCG31QM4A1aNggQlhKkhIGGTG5embID5JINH8BvOnTF-jSSTfPO2qfXQRxhD2XuMhLz5x8TGfZRerkxtp-WR5MtMZRMxjNiEbRFBEBXjQkKW58henO0M8j0j9jXGg44e gybdljVo5xlJ2V9CKB60Q1h/2-ViI1m6nb0XL wnbFlyDbcPReOxVZAEwLA gAH7qRmkZADgYYW10i2RpcNDgVnfXTDRHA2mOHuuFW1y1mPOTs6EW73-kd10kbtk6u cI0B6qKVy Q0DAXky24XH5g1Yj2X5nzRAu7AiAL4oAr5AaYNVQ1c uTJDCY-uD4Nln3HgUo/4f8ZQ2E-C7iHY0Xpq6OFFXoQCQdUkALCQASCMjAzALw7AGg2y8Y/cUjQA4fg4y3URWD4CoaYVYAEAiMr_6G3-BK8-Qj4fCQM-DgEUgY0f8AJuIwFh8Qjgs0eAZw0hH5eVQaJgbUBbfvfRWhXuq/mwei6b-vSLEZiCB hQe7yxR9x/MO4hZRp zRhH7Xb/YC2HbMEC7MFP- pCMGRYx4D6_PMlQc9ID/DXU/4GgYHIqALQBETDCrS14/p/+wFVgABK5mtvHX7CXPIY4Bcz03Nc3zZvhjiflY99Ksw7OX9z5JyOz2UyvI/q4g0Zw/670C8OOXGK3hHnBM53CYFmEQuxAR1h2 MYxKEgIUIVyNEK8MVQEZ3Gl9WBEI8mEEV2xYgr/AXVldQAuqBlxtAAAAAElFTkSuQmCC" height="50" width="50">Reconnaissance de plantes - Détection des maladies
    </div>
    """, unsafe_allow_html=True)

# Sidebar pour la navigation avec des emojis et un style simplifié
st.sidebar.title("Navigation")
st.sidebar.write("Choisissez une page ci-dessous:")

# Dictionnaire pour mapper les pages aux fonctions correspondantes
pages = {
    "🏰 Accueil": lambda: (st.title("Reconnaissance de plantes - Détection des maladies"),
                          st.subheader("DataScientest - Promotion Février 2024")),
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
selected_page = st.sidebar.radio("", list(pages.keys()), index=0)

# Affichage de la page sélectionnée
if selected_page in pages:
    pages[selected_page]()

# Personnalisation des boutons
st.markdown("""
<style>
.stButton button {
    background-color: #4A7C59;
    color: white;
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    transition: background-color 0.3s ease;
}
.stButton button:hover {
    background-color: #3c6246;
}
</style>
""", unsafe_allow_html=True)

# Footer
footer = """
<div class="footer">
    <p>Projet réalisé par Felipe Souto et Nicolas Papegaey | DataScientest - Promotion Février 2024</p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)

# CSS pour le style (amélioration des couleurs et du design général)
st.markdown("""
<style>
.block-container {
    max-width: 1200px;
    padding: 2rem;
    background-color: #f9f9f9;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)