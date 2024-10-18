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

    # Organisation du contenu avec des colonnes (image à gauche, texte à droite)
    col1, col2 = st.columns([2, 3])  # Ajuster la taille des colonnes, col1 pour l'image, col2 pour le texte

    # Afficher l'image dans la première colonne
    with col1:
        st.image(options[choice], caption=f"Distribution de {choice}")

    with col2:
        if choice == "Netteté":
            st.subheader("Netteté")
            st.write("""
                La netteté mesure la clarté des détails dans l'image. 
                Un score bas indique une image floue, ce qui peut rendre difficile la détection 
                des caractéristiques visuelles des maladies. Les images nettes facilitent l'analyse automatique.
            """)

        elif choice == "Contraste":
            st.subheader("Contraste")
            st.write("""
                Le contraste mesure la différence entre les zones claires et sombres de l'image.
                Un contraste élevé aide à distinguer les détails importants des feuilles, tandis qu'un contraste faible
                peut rendre la différenciation des zones atteintes difficile.
            """)

        elif choice == "Luminosité":
            st.subheader("Luminosité")
            st.write("""
                La luminosité représente la clarté ou l'obscurité globale de l'image. Une luminosité trop faible
                peut cacher des détails importants, tandis qu'une luminosité excessive peut saturer l'image,
                rendant la détection des caractéristiques plus difficile. Une luminosité équilibrée est essentielle 
                pour une bonne qualité d'image.
            """)

        elif choice == "Bruit":
            st.subheader("Bruit")
            st.write("""
                Le bruit représente les anomalies ou les distorsions dans une image qui peuvent réduire la qualité globale.
                Des niveaux élevés de bruit peuvent masquer les caractéristiques importantes de la feuille, 
                rendant plus difficile la détection des maladies. Un bruit minimal garantit une image plus claire et plus facile à analyser.
            """)

        elif choice == "Saturation":
            st.subheader("Saturation")
            st.write("""
                La saturation mesure l'intensité des couleurs dans une image. Une saturation élevée signifie que les couleurs 
                sont plus vives, tandis qu'une saturation faible donne des couleurs plus ternes. Une saturation bien équilibrée 
                aide à détecter les anomalies de couleur sur les feuilles, comme les taches ou les zones décolorées.
            """)

        elif choice == "Ratio de Zoom":
            st.subheader("Ratio de Zoom")
            st.write("""
                Le ratio de zoom indique à quel point l'image est agrandie par rapport à sa taille réelle.
                Un ratio de zoom élevé peut aider à voir des détails microscopiques, mais peut aussi omettre le contexte général 
                de la feuille. Un bon équilibre de zoom permet de visualiser à la fois les détails et la feuille entière.
            """)

        elif choice == "SNR (Signal to Noise Ratio)":
            st.subheader("SNR (Signal to Noise Ratio)")
            st.write("""
                Le rapport signal-bruit (SNR) est une mesure de la clarté d'une image par rapport au bruit de fond.
                Un SNR élevé signifie que les détails importants sont bien visibles par rapport au bruit. 
                Un SNR faible indique que le bruit interfère avec les caractéristiques importantes, ce qui complique 
                la détection des maladies.
            """)

        elif choice == "Entropie":
            st.subheader("Entropie")
            st.write("""
                L'entropie mesure la complexité et la quantité d'information contenue dans une image.
                Une entropie élevée indique une grande variété de textures et de détails, ce qui peut être utile pour 
                détecter des anomalies complexes. Une entropie faible suggère une image plus uniforme, 
                où les détails sont peut-être moins nombreux.
            """)


# Appel de la fonction display pour afficher la page dans Streamlit
if __name__ == "__main__":
    display()
