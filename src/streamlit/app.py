import streamlit as st
from sections import presentation_sujet, presentation_donnees, analyse_donnees, preprocessing, modelisation, meilleur_modele, conclusion, demonstration

# Titre principal de l'application
st.title("Présentation de Machine Learning pour la classification d'images")

# Sidebar pour la navigation
st.sidebar.title("Menu de navigation")
sections = st.sidebar.radio("Choisissez une section", (
    "Présentation du sujet", 
    "Présentation des données", 
    "Analyse des données", 
    "Préprocessing", 
    "Modèles entraînés", 
    "Analyse du meilleur modèle", 
    "Conclusion",
    "Démonstration"
))

# Afficher la section choisie
if sections == "Présentation du sujet":
    presentation_sujet.display()

elif sections == "Présentation des données":
    presentation_donnees.display()

elif sections == "Analyse des données":
    analyse_donnees.display()

elif sections == "Préprocessing":
    preprocessing.display()

elif sections == "Modèles entraînés":
    modelisation.display()

elif sections == "Analyse du meilleur modèle":
    meilleur_modele.display()

elif sections == "Conclusion":
    conclusion.display()
elif sections == "Demonstration":
    demonstration.display()
