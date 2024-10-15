import streamlit as st

def display():
    st.title("Présentation du sujet")
    st.markdown("""
    ### Sujet du projet
    Le projet consiste à classifier des images de plantes malades et saines...
    
    ### Enjeux
    - Réduire le temps de diagnostic
    - Utiliser un modèle d'IA pour automatiser la reconnaissance des maladies
    - Impact potentiel sur l'industrie agricole
    """)
