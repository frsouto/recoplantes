import streamlit as st

def display():

    # Titre principal
    st.title("📊 Modèles Entraînés")
    
    # Introduction générale
    st.write("""
    Dans le cadre de notre projet de diagnostic des maladies des plantes, plusieurs modèles de Deep Learning ont été testés et optimisés. 
    Nous présentons ici un résumé des modèles utilisés, les hyperparamètres choisis, ainsi que les défis spécifiques rencontrés pendant l'entraînement.
    """)

    # Section pour le CNN
    st.subheader("🧠 Convolutional Neural Network (CNN)")
    st.write("""
    Le CNN est une architecture classique adaptée à l'analyse d'images. Elle a été notre premier choix pour établir une base solide dans la classification des maladies des plantes.
    """)
    st.markdown("**Caractéristiques :**")
    st.write("""
    - Couches : Conv2D, MaxPooling, Dense layers.
    - Régularisation : Dropout à 20%.
    """)
    st.markdown("**Hyperparamètres :**")
    st.write("""
    - Batch size : 32
    - Learning rate : 0.0001
    - Nombre d'époques : 20 (early stopping à l'époque 13 pour éviter l'overfitting).
    """)
    st.markdown("**Défis rencontrés :**")
    st.write("""
    - **Overfitting** après 13 époques, nécessitant une régularisation supplémentaire via Dropout.
    - Performance globale solide avec une accuracy de validation de 91%, mais nécessité d'améliorer les classes minoritaires.
    """)

    # Section pour MobileNet
    st.subheader("📱 MobileNetV1 - Transfer Learning")
    st.write("""
    MobileNetV1, utilisé avec du transfer learning, a permis une grande amélioration des performances tout en réduisant le temps d'entraînement.
    """)
    st.markdown("**Caractéristiques :**")
    st.write("""
    - Architecture légère avec GlobalAveragePooling pour réduire la dimensionnalité.
    - Utilisation de Dropout à 20%.
    """)
    st.markdown("**Hyperparamètres :**")
    st.write("""
    - Batch size : 32
    - Learning rate : 0.0001
    - Nombre d'époques : 20
    """)
    st.markdown("**Résultats :**")
    st.write("""
    - **Accuracy d'entraînement :** 100%
    - **Accuracy de validation :** 99.71% après stabilisation de la loss à 0.0149.
    """)

    # Section pour ResNet
    st.subheader("🔗 ResNet50 - Résultats mitigés")
    st.write("""
    ResNet50 a été testé pour sa capacité à capturer des détails complexes. Cependant, les contraintes de calcul ont limité les possibilités d'ajustement des hyperparamètres.
    """)
    st.markdown("**Défis rencontrés :**")
    st.write("""
    - Temps d'entraînement prolongé sur Google Colab, ce qui a ralenti les itérations.
    - Difficulté à stabiliser les performances sur certaines classes malgré la puissance de l'architecture.
    """)

# Appel de la fonction pour afficher les détails des modèles
if __name__ == "__main__":
    display()

