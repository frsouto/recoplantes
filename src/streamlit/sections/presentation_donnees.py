import streamlit as st
import pandas as pd

def display():
    st.title("Présentation des données")
    st.write("Description des données utilisées dans le projet")
    
    # Exemple d'affichage de la volumétrie des données
    df = pd.read_csv('chemin/vers/ton_fichier_donnees.csv')
    st.write("Dimensions du dataset : ", df.shape)
    st.write(df.head())
