import streamlit as st
from PIL import Image
import os

# Introduction à la partie Préprocessing
def display():
    # Introduction
    st.title("⚙️ Préprocessing des données")
    st.write("""
Le prétraitement est une étape clé du machine learning. Il se divise en deux phases :
1. **Prétraitement manuel** pour améliorer la qualité des images.
2. **Prétraitement automatisé** pour préparer les données pour l'entraînement.
""")

    # Créer des onglets pour les différentes sections
    tab1, tab2, tab3 = st.tabs(["Prétraitement Manuel", "Prétraitement Automatisé", "Conclusion"])

    # Contenu de l'onglet "Prétraitement Manuel"
    with tab1:
        st.subheader("📝 Prétraitement Manuel")
        
        st.write("""
        Cette phase consiste à nettoyer le dataset avant l'entraînement :
        - Suppression des images dupliquées.
        - Élimination des images floues ou mal cadrées.
        """)

        st.write("**1. Suppression des Doublons**")
        st.write("""
        Les doublons ont été supprimés en utilisant un hash MD5. Cela évite les biais lors de l'entraînement.
        """)

        st.write("**2. Suppression des Outliers**")
        st.write("""
        Les images de mauvaise qualité (floues, surexposées, etc.) ont été retirées. Cela améliore la qualité des données utilisées.
        """)


    # Contenu de l'onglet "Prétraitement Automatisé"
    with tab2:
        with tab2:
            st.subheader("⚙️ Prétraitement Automatisé")
            
            st.write("""
            Cette phase applique des transformations automatiques pour préparer les images :
            - **Redimensionnement** des images en 224x224.
            - **Normalisation** des valeurs des pixels entre 0 et 1.
            - **Division** du dataset en ensemble d'entraînement et de validation.
            """)




    # Contenu de l'onglet "Conclusion"
    with tab3:
        st.subheader("📊 Conclusion")
        
        st.write("""
        Grâce à ce prétraitement, les données sont prêtes pour l'entraînement. Cela garantit un dataset propre, de bonne qualité, 
        et améliore la performance du modèle.
        """)


# Appel de la fonction display pour afficher la page dans Streamlit
if __name__ == "__main__":
    display_preprocessing()
