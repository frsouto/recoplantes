import streamlit as st
from PIL import Image
import os

# Introduction à la partie Préprocessing
def display():
    st.title("Préprocessing des données")
    st.write(
        '''
        Le prétraitement des données est une étape cruciale dans notre pipeline de machine learning.
        Il est divisé en deux phases : un prétraitement manuel et un prétraitement automatisé.
        Le prétraitement manuel vise à améliorer la qualité des données, tandis que le prétraitement automatisé prépare les données pour qu'elles puissent être utilisées par les modèles de deep learning.
        '''
    )

    # Section 1: Prétraitement Manuel
    st.subheader("Prétraitement Manuel")
    st.write(
        '''
        Le prétraitement manuel des données inclut les étapes de nettoyage avant que les images ne soient introduites dans le pipeline d'entraînement. Ces étapes permettent de garantir la qualité du dataset et de supprimer les images inadaptées ou redondantes.
        '''
    )

    # Étape 1: Suppression des doublons
    st.subheader("Suppression des Doublons")
    st.write(
        '''
        Avant de commencer l'entraînement, nous avons vérifié et supprimé les images dupliquées dans le dataset en utilisant une méthode de hash MD5. Cette étape permet d'éviter que des images similaires ne biaisent l'entraînement du modèle.
        '''
    )

    # Étape 2: Suppression des outliers
    st.subheader("Suppression des Outliers")
    st.write(
        '''
        Les images présentant des défauts de qualité (floues, mal cadrées, surexposées, etc.) ont été identifiées manuellement et supprimées. Cela garantit que seules les images de haute qualité sont utilisées pour l'entraînement, améliorant ainsi la performance globale du modèle.
        '''
    )

    # Section 2: Prétraitement Automatisé
    st.subheader("Prétraitement Automatisé")
    st.write(
        '''
        Après avoir nettoyé les données avec les étapes manuelles, nous avons appliqué un ensemble de transformations automatiques aux images avant l'entraînement. Ces étapes sont réalisées directement dans le code et incluent le redimensionnement, la normalisation et la division des données.
        '''
    )

    # Étape 1: Redimensionnement des images
    st.subheader("Redimensionnement des Images")
    st.write(
        '''
        Toutes les images sont redimensionnées à une taille de 224x224 pixels. Cela standardise la taille des entrées du modèle et permet une compatibilité avec les architectures CNN prédéfinies. Les images originales peuvent être plus grandes ou plus petites, mais elles sont uniformisées pour optimiser le processus d'apprentissage.
        '''
    )
    

    # Étape 2: Normalisation des pixels
    st.subheader("Normalisation des Pixels")
    st.write(
        '''
        Les valeurs des pixels des images ont été normalisées en les divisant par 255, ce qui les transforme dans une plage de [0, 1].
        Cette normalisation aide le modèle à mieux comprendre les données et accélère l'entraînement en stabilisant les activations des couches du réseau.
        '''
    )

    

    # Étape 4: Division du dataset
    st.subheader("Division du Dataset")
    st.write(
        '''
        Le dataset a été divisé en deux parties : un ensemble d'entraînement (80%) et un ensemble de validation (20%). Cela permet de tester la performance du modèle sur des données qu'il n'a pas vues pendant l'entraînement, garantissant une bonne généralisation.
        '''
    )

    st.write("---")

    # Conclusion
    st.subheader("Conclusion")
    st.write(
        '''
        Le prétraitement des données, à la fois manuel et automatisé, nous permet d'obtenir un dataset propre et bien formaté, prêt pour l'entraînement du modèle. Grâce à ces étapes, nous maximisons la qualité des données et améliorons les performances du modèle final.
        '''
    )

# Appel de la fonction display pour afficher la page dans Streamlit
if __name__ == "__main__":
    display_preprocessing()
