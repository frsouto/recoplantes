import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

def display():
    st.title("Analyse des données")
    st.write("Analyse exploratoire des données avec DataViz")
    
    # Exemple de graphique
    data = sns.load_dataset('iris')  # Remplace avec tes données
    sns.pairplot(data, hue='species')
    st.pyplot(plt)
