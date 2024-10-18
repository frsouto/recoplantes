import streamlit as st

def display():
    # Titre principal
    st.title("📝 Conclusion et Perspectives")

    # Résumé du projet
    st.subheader("Résumé du projet")
    st.write("""
    Ce projet avait pour objectif de détecter automatiquement les maladies des plantes à partir d'images de feuilles. 
    Plusieurs modèles ont été testés, et le modèle **MobileNetV1**, avec du **transfer learning**, a obtenu les meilleurs résultats avec une précision de **99,71%** sur l'ensemble de validation. 
    Malgré des défis tels que le déséquilibre des classes et des erreurs de segmentation, ce modèle a bien géré la classification.
    """)

    # Points clés
    st.subheader("Points clés")
    st.write("""
    - **Performance du modèle :** Le modèle a bien réussi à généraliser sur des maladies distinctes, mais des confusions subsistent entre certaines maladies aux symptômes similaires, comme la brûlure précoce et tardive des tomates.
    - **Segmentation des images :** La segmentation des feuilles n'a pas toujours amélioré les performances du modèle et a parfois conduit à des erreurs, ce qui souligne la nécessité d'améliorer cette étape.
    - **Interprétabilité :** Des outils comme **GradCAM** et **LIME** ont aidé à comprendre les erreurs du modèle en mettant en lumière des zones d'images non pertinentes utilisées pour la classification.
    """)

    # Perspectives futures
    st.subheader("Perspectives futures")
    st.write("""
    - **Augmentation des données :** Ajouter davantage de données, surtout pour les classes moins représentées, et utiliser des techniques d'augmentation comme les **GANs** (Generative Adversarial Networks) pour générer de nouvelles images synthétiques.
    - **Amélioration de la segmentation :** Optimiser le processus de segmentation pour se concentrer sur les zones pertinentes et minimiser les erreurs liées au bruit ou aux zones non informatives.
    - **Exploration de nouveaux modèles :** Tester des architectures plus avancées comme **EfficientNet** ou les **Vision Transformers** pour améliorer la reconnaissance des maladies dans des contextes plus déséquilibrés et complexes.
    """)
