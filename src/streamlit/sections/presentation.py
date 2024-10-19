# contexte_motivation.py

import streamlit as st

def display():
    st.title("Contexte et motivation")

    tabs = st.tabs(["Introduction", "Objectif", "Impact", "Approche"])

    with tabs[0]:
        st.header("Introduction")
        col1, col2 = st.columns([3, 2])
        with col1:
            st.write("""
            La détection des maladies des plantes est un enjeu crucial pour l'agriculture moderne. 
            Les maladies végétales peuvent entraîner :
            - **Pertes de rendement importantes**
            - **Menace pour la sécurité alimentaire**
            - **Coûts élevés pour les agriculteurs** (traitements et prévention)

            Avec le changement climatique, la propagation de nouvelles maladies ou l'intensification 
            des maladies existantes devient un risque accru.
            """)
        with col2:
            st.image("images/IMG_5931.jpg", caption="Impact des maladies sur les cultures")

    with tabs[1]:
        st.header("Objectif du projet")
        st.write("""
        L'objectif de ce projet est de développer un modèle capable de détecter automatiquement 
        les maladies végétales à partir d'images de feuilles. Cela peut aider les agriculteurs à :
        - Prendre des décisions précoces et informées
        - Réduire les pertes économiques
        - Améliorer l'efficacité des traitements
        """)

    with tabs[2]:
        st.header("Impact potentiel")
        st.write("""
        Une détection précoce et précise des maladies peut avoir un impact significatif sur :
        - **La productivité agricole**
        - **La sécurité alimentaire**
        - **La réduction de l'utilisation de pesticides**
        - **La durabilité des pratiques agricoles**
        """)

    with tabs[3]:
        st.header("Approche")
        st.write("""
        Notre approche utilise des techniques de vision par ordinateur et d'apprentissage profond pour :
        1. Analyser des images de feuilles
        2. Identifier les caractéristiques visuelles associées à différentes maladies
        3. Classifier les images dans l'une des 38 catégories, représentant soit des maladies spécifiques, soit un état sain

        Cette solution automatisée permettrait une détection rapide et à grande échelle des 
        maladies des plantes, offrant aux agriculteurs un outil précieux pour la gestion de leurs cultures.
        """)

if __name__ == "__main__":
    display()
