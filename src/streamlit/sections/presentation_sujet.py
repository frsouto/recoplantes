import streamlit as st

def display():
    # Titre et sous-titre
    st.subheader("Un projet appliqué à l'agriculture de précision")
    
    # Introduction concise
    st.markdown("""
    ### Problématique :
    Les maladies des plantes représentent un enjeu majeur pour l'agriculture. Elles causent des pertes économiques importantes 
    et menacent la sécurité alimentaire mondiale. Les méthodes traditionnelles de diagnostic reposent souvent sur des inspections 
    manuelles qui sont coûteuses, lentes et nécessitent une expertise humaine pointue.
    
    ### Enjeu :
    Le développement d'une méthode automatisée, rapide et précise pour diagnostiquer les maladies des plantes à partir de simples 
    images de feuilles, en utilisant le Deep Learning.
    
    ### Solution proposée :
    Utilisation de techniques avancées de vision par ordinateur pour classer les maladies à partir d'un jeu de données d'images de feuilles, 
    permettant aux agriculteurs de diagnostiquer rapidement et efficacement les maladies.
    """)

    # Illustration
    st.image("images/plantnetwork.webp", caption="Feuilles et technologie : Deep Learning pour diagnostiquer les maladies", use_column_width=True)

    # Détails supplémentaires sous l'image
    st.markdown("""
    ### Détails supplémentaires :
    
    **Objectif du projet** : Créer un modèle d'apprentissage profond capable de classer les maladies des plantes avec une précision élevée, 
    en réduisant les erreurs de diagnostic.
    
    **Impact attendu** : Accélérer les processus décisionnels dans les exploitations agricoles, améliorer les rendements et réduire les pertes économiques.
    """)
    
# Appel de la fonction pour afficher la page
if __name__ == "__main__":
    display()
