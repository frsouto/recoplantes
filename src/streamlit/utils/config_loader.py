import json

def load_model_configs(config_file):
    """Charger la configuration des modèles depuis un fichier JSON"""
    with open(config_file, "r") as file:
        return json.load(file)