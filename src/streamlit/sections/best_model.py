import streamlit as st

def display():
    # Titre principal
    st.title("🏆 Analyse du Meilleur Modèle")

    # Introduction générale du meilleur modèle
    st.write("""
    Le modèle choisi pour le diagnostic des maladies des plantes est basé sur MobileNetV1 avec Transfer Learning. 
    Ce modèle s'est révélé être le plus performant parmi ceux testés, surpassant des architectures comme VGG16 et ResNet, grâce à sa capacité à généraliser efficacement tout en ayant des exigences computationnelles réduites.
    """)

    # Résultats principaux
    st.subheader("📊 Principaux résultats")
    st.write("""
    - **Accuracy d'entraînement** : 100%
    - **Accuracy de validation** : 99.71%
    - **Perte de validation (Validation Loss)** : 0.0149

    Cela montre une excellente capacité de généralisation et une performance quasi-parfaite sur l'ensemble de validation.
    """)

    # Optimisations appliquées
    st.subheader("🔧 Optimisations appliquées")
    st.write("""
    - **Transfer Learning** avec MobileNetV1 pré-entraîné sur ImageNet, ce qui a permis de tirer parti des caractéristiques extraites sur un large dataset d'images variées.
    - **GlobalAveragePooling** pour réduire la dimensionnalité des caractéristiques et éviter le sur-apprentissage.
    - **Dropout à 20%** pour améliorer la régularisation et éviter l'overfitting.
    """)

    # Pourquoi MobileNetV1
    st.subheader("📱 Pourquoi MobileNetV1 ?")
    st.write("""
    Le modèle MobileNetV1 a été choisi pour son efficacité à extraire des caractéristiques complexes avec une architecture légère, particulièrement adaptée aux dispositifs à faible puissance de calcul. 
    L'utilisation de Transfer Learning a également permis de réduire considérablement le temps d'entraînement tout en augmentant les performances globales.
    """)

    # Prochaines étapes
    st.subheader("🚀 Prochaines étapes")
    st.write("""
    Pour continuer à améliorer les performances du modèle, des techniques d'augmentation de données sont envisagées, telles que des transformations aléatoires (flip, rotation, zoom), afin de traiter plus efficacement les classes minoritaires et d'enrichir la diversité des données d'entraînement.
    """)
