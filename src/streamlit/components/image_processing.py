# image_processing.py

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import lime
from lime import lime_image

def preprocess_image(img, model_choice, for_lime=False, model_configs=None):
    """Prétraite l'image selon le modèle choisi."""
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_configs[model_choice]["preprocessing"] == "simple":
        return img_array / 255.0
    else:  # Pour MobileNet
        if for_lime:
            # Pour LIME, on normalise simplement entre 0 et 1
            return img_array / 255.0
        else:
            # Pour la prédiction normale, on utilise le prétraitement MobileNet
            return tf.keras.applications.mobilenet.preprocess_input(img_array)


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, intensity=4):
    """Génère la heatmap Grad-CAM."""
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        pred_index = pred_index.numpy()
        if isinstance(pred_index, np.ndarray):
            if pred_index.size == 1:
                pred_index = pred_index.item()
            else:
                pred_index = pred_index[0]
                
        class_channel = predictions[0][pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = heatmap * intensity
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.3):
    """Superpose la heatmap Grad-CAM sur l'image originale."""
    img = img.resize((224, 224))
    img_array = np.array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.cm.get_cmap("inferno")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img_array
    superimposed_img = image.array_to_img(superimposed_img)
    
    return superimposed_img

def explain_with_lime(img_array, model, model_choice, model_configs):
    """Génère l'explication LIME."""
    predictor = create_lime_predictor(model, model_choice, model_configs)
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array[0].astype('double'),  # Assurer que l'image est en double précision
        predictor,
        top_labels=1,
        hide_color=0,
        num_samples=15
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=True
    )
    
    # S'assurer que l'image est normalisée entre 0 et 1
    if temp.max() > 1.0:
        temp = temp / 255.0
        
    return mark_boundaries(temp, mask)

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
