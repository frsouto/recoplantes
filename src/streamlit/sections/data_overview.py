import streamlit as st
from PIL import Image

def display():
    # Titre de la page
    st.title("Présentation des Données du Projet : Reconnaissance de Maladies des Plantes")

    # Création des onglets
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Introduction", 
        "Volumétrie", 
        "Distribution", 
        "Exemples", 
        "Qualité", 
        "Conclusion"
    ])

    # Onglet 1: Introduction au Dataset
    with tab1:
        st.header("Introduction")
        st.write("""
        Ce projet utilise le dataset **PlantVillage** disponible sur Kaggle, avec plus de 50 000 images de feuilles de plantes. 
        Ces images sont réparties en 38 catégories représentant différentes maladies ou des feuilles saines. 
        Le dataset est utilisé pour entraîner un modèle de deep learning capable de détecter automatiquement les maladies des plantes.
        """)
        st.image("images/dataset_overview.png", caption="Aperçu du dataset PlantVillage")

    # Onglet 2: Volumétrie des Données et Exploration
    with tab2:
        st.header("Volumétrie des Données")
        st.write("""
        Le dataset contient un grand nombre d'images réparties inégalement selon les classes. 
        Vous pouvez consulter un aperçu des dimensions du dataset et quelques images représentatives.
        """)
        st.subheader("Dimensions du Dataset")
        st.write("Le dataset contient plus de 50 000 images réparties en 38 classes différentes.")

    # Onglet 3: Distribution des Classes
    with tab3:
        st.header("Distribution des Classes de Maladies")
        st.write("""
        La répartition des images n'est pas équilibrée : certaines maladies sont sur-représentées, 
        comme **Tomato Yellow Leaf Curl Virus**, tandis que d'autres maladies ont moins d'exemples, ce qui pourrait introduire des biais dans l'entraînement du modèle.
        """)
        st.image("images/class_distribution.png", caption="Distribution des Classes de Maladies dans le Dataset")

    # Onglet 4: Exemples d'Images
    with tab4:
        st.header("Exemples d'Images du Dataset")
        st.write("""
        Le dataset contient des images représentant différentes maladies. Voici quelques exemples d'images des classes les plus représentées.
        Utilisez le sélecteur ci-dessous pour voir des images correspondant à une maladie spécifique.
        """)
        disease_class = st.selectbox(
            "Sélectionnez une classe de maladie",
            ["Tomato Yellow Leaf Curl Virus", "Apple Cedar Rust", "Citrus Greening", "Healthy"]
        )
        if disease_class == "Tomato Yellow Leaf Curl Virus":
            st.image("images/tomato_yellow_leaf_curl.png", caption="Tomato Yellow Leaf Curl Virus")
        elif disease_class == "Apple Cedar Rust":
            st.image("images/apple_cedar_rust.png", caption="Apple Cedar Rust")
        elif disease_class == "Citrus Greening":
            st.image("images/citrus_greening.png", caption="Citrus Greening")
        else:
            st.image("images/healthy_leaf.png", caption="Healthy Leaf")

    # Onglet 5: Analyse de la Qualité des Images
    with tab5:
        st.header("Analyse de la Qualité des Images")
        st.write("""
        Une analyse approfondie a été réalisée pour évaluer la qualité des images du dataset. 
        Des métriques comme la netteté, la luminosité et le bruit ont été mesurées pour garantir que seules des images de haute qualité sont utilisées pour l'entraînement du modèle.
        """)
        st.image("images/sharpness_analysis.png", caption="Analyse de la Netteté des Images")
        if st.button("Télécharger le rapport complet sur la qualité des images"):
            with open("reports/image_quality_report.pdf", "rb") as file:
                st.download_button(label="Télécharger le rapport", data=file, file_name="image_quality.pdf", mime="application/pdf")

    # Onglet 6: Conclusion Préliminaire
    with tab6:
        st.header("Conclusion Préliminaire")
        st.write("""
        L'analyse des données montre un fort déséquilibre entre les classes ainsi qu'une qualité d'image variable. 
        Ces facteurs doivent être pris en compte lors de l'entraînement du modèle pour améliorer les performances de classification.
        """)
        if st.button("Voir les prochaines étapes du projet"):
            st.write("""
            Les prochaines étapes incluent :
            - La gestion des déséquilibres de classes via des techniques d'augmentation de données.
            - L'amélioration de la qualité du dataset en éliminant les doublons et les images de mauvaise qualité.
            - La validation des modèles avec des approches de transfer learning.
            """)

# Appel de la fonction display pour afficher la page dans Streamlit
if __name__ == '__main__':
    display()
