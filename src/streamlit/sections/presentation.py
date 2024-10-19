import streamlit as st
import plotly.express as px
import pandas as pd

def display():
    st.title("Reconnaissance de plantes - Détection des maladies")

    tabs = st.tabs(["Contexte", "Problème", "Enjeux"])

    with tabs[0]:
        st.header("Contexte")
        
        col1, col2 = st.columns([2,1])
        
        with col1:
            st.markdown("""
            - La détection des maladies des plantes est cruciale pour l'agriculture moderne
            - Impact significatif sur la sécurité alimentaire mondiale
            - Le changement climatique augmente les risques de propagation des maladies
            - Utilisation de l'intelligence artificielle pour résoudre ce défi
            """)
        
        with col2:
            st.image("images/IMG_5931.jpg", caption="Exemple de maladie des plantes")

    with tabs[1]:
        st.header("Problème")
        
        col1, col2 = st.columns([2,1])
        
        with col1:
            st.markdown("""
            - Nécessité d'un diagnostic rapide et précis des maladies des plantes
            - Défi : variabilité des symptômes et des conditions environnementales
            - Objectif : développer un système automatisé de détection basé sur l'image
            - Application de techniques avancées de vision par ordinateur
            """)
        
        with col2:
            st.image("images/ai_diagnosis.webp", caption="Diagnostic automatisé")

    with tabs[2]:
        st.header("Enjeux")
        
        col1, col2 = st.columns([1,1])
        
        with col1:
            st.markdown("""
            - Amélioration des rendements agricoles
            - Réduction des pertes économiques pour les agriculteurs
            - Optimisation de l'utilisation des pesticides
            - Contribution à la durabilité des pratiques agricoles
            - Avancement de la recherche en agritech
            """)
        
        with col2:
            st.markdown("""
                ### Impact potentiel

                - La détection automatisée peut améliorer les récoltes de 2% à 4% à court terme, avec un potentiel d'optimisation jusqu'à 20% à long terme.
                - Réduction significative du nombre de traitements chimiques, générant des économies importantes sur les intrants.
                - Certains modèles d'apprentissage profond atteignent une précision de prédiction de 94% pour la détection des maladies.
                """)
            

if __name__ == "__main__":
    display()
