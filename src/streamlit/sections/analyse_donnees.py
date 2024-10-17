import streamlit as st

# Fonction principale d'affichage de l'analyse
def display():
    st.title("Analyse des données")
    st.write("Sélectionnez une métrique pour voir la distribution correspondante.")

    # Liste des métriques disponibles
    options = {
        "Netteté": "images/sharpness_distribution.png",
        "Contraste": "images/contrast_distribution.png",
        "Luminosité": "images/brightness_distribution.png",
        "Bruit": "images/noise_distribution.png",
        "Saturation": "images/saturation_distribution.png",
        "Ratio de Zoom": "images/ratiozoom_distribution.png",
        "SNR (Signal to Noise Ratio)": "images/snr_distribution.png",
        "Entropie": "images/entropy_distribution.png"
    }

    # Sélecteur pour choisir la métrique
    choice = st.selectbox("Choisissez une métrique à visualiser :", list(options.keys()))

    # Afficher l'image correspondant à la métrique sélectionnée
    st.image(options[choice], caption=f"Distribution de {choice}")

# Appel de la fonction display pour afficher la page dans Streamlit
if __name__ == "__main__":
    display()
