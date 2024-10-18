
import streamlit as st

def display():
    # Titre principal
    st.title("Diagnostic des Maladies des Plantes - Un Projet pour l'Agriculture de Précision")
    
    # Introduction et contexte
    st.markdown("""
    ### Problématique :
    - **Problème majeur** : Maladies des plantes = menace pour l'agriculture moderne.
    - **Impact global** : Sécurité alimentaire  en jeu.
    - **Limites actuelles** : Méthodes manuelles, coûteuses, et lentes.
    """)

    # Organisation du contenu en colonnes pour plus de compacité
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Enjeu :
        Le développement d'une solution de diagnostic automatisée pour détecter les maladies des plantes rapidement et avec précision, 
        en utilisant des techniques de Deep Learning basées sur l'analyse d'images.
        """)

    with col2:
        st.markdown("""
        ### Solution proposée :
        Utilisation de techniques de vision par ordinateur et de réseaux de neurones convolutifs pour classer les maladies à partir d'images 
        de feuilles.
        """)



    # Détails supplémentaires
    st.markdown("""
    ### Détails supplémentaires :
    
    **Objectif** : Développer un modèle d'apprentissage performant pour améliorer le diagnostic des maladies des plantes.  
    
    **Impact attendu** : Accélérer la prise de décision dans l'agriculture et améliorer les rendements.
    """)

# Appel de la fonction pour afficher la page
if __name__ == "__main__":
    display()
