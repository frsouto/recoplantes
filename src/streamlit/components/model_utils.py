# model_utils.py

import tensorflow as tf
import numpy as np

def load_model(model_name):
    """Charge un modèle TensorFlow sauvegardé."""
    return tf.keras.models.load_model(model_name)

def create_lime_predictor(model, model_choice, model_configs):
    """Crée une fonction de prédiction adaptée pour LIME."""
    def predictor(images):
        processed_images = []
        for img in images:
            if img.dtype != np.float32:
                img = img.astype(np.float32)
            if img.max() > 1.0:
                img = img / 255.0
            if model_configs[model_choice]["preprocessing"] == "mobilenet":
                img = img * 255.0
                img = tf.keras.applications.mobilenet.preprocess_input(img)
            processed_images.append(img)
        
        batch = np.stack(processed_images)
        return model.predict(batch)
    
    return predictor
