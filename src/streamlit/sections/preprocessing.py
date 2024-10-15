import streamlit as st

def display():
    st.title("Préprocessing")
    st.write("Description et justification du preprocessing effectué sur les données...")
    st.markdown("""
    - Suppression des doublons
    - Redimensionnement des images
    - Normalisation des données
    """)
