Plant Disease Recognition Project
==============================

Ce dépôt contient le projet de reconnaissance de maladies des plantes développé dans le cadre de la promotion Février 2024 de DataScientest. 
L’objectif principal est de créer un modèle performant pour détecter automatiquement des maladies végétales à partir d’images de feuilles. 

L'organisation du projet suit une structure modulaire, facilitant la navigation et la reproductibilité pour les futures itérations et analyses. 
Vous trouverez ci-dessous une vue d'ensemble des principaux dossiers et fichiers du projet.

Project Organization
==============================

    .
├── LICENSE                   # Licence du projet
├── README.md                 # Documentation principale du projet
├── models                    # Modèles entraînés et fichiers de sauvegarde
│   └── CNNFirstComplet.keras # Exemple de modèle CNN sauvegardé
├── notebooks                 # Notebooks Jupyter pour les différentes étapes du projet
│   ├── 01_exploration        # Exploration des données et visualisation
│   ├── 02_preparation        # Préparation et prétraitement des données
│   └── 03_modelisation       # Entraînement, fine-tuning et évaluation des modèles
├── reports                   # Rapports et fichiers de résultats
│   ├── figures               # Figures et visualisations générées pour le rapport
│   └── fev24cds_plants_modelisation_v1.0.pdf # Rapport de modélisation
├── requirements.txt          # Liste des dépendances du projet
└── src                       # Code source du projet
    ├── features              # Transformation et extraction des caractéristiques
    │   └── build_features.py # Script de préparation des données
    ├── models                # Scripts pour entraîner et faire des prédictions
    │   ├── predict_model.py
    │   └── train_model.py
    ├── streamlit             # Application Streamlit pour l'interprétabilité
    │   ├── app.py            # Point d'entrée de l'application Streamlit
    │   └── components        # Composants spécifiques de l'application
    └── visualization         # Scripts de visualisation des résultats et architecture
        └── visualize.py


--------
Instructions pour lancer l'application Streamlit
--------

```bash
# Clone the repository
git clone https://github.com/DataScientest-Studio/fev24_cds_plants.git

# Create and activate the virtual environment
python3 -m venv plantes
source plantes/bin/activate

# Install dependencies
cd fev24_cds_plants
pip install -r requirements.txt

# Launch the Streamlit application
cd src/streamlit
streamlit run interpretabilite.py

