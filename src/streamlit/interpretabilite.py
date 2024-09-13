import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Liste des classes
class_names = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Charger le modèle pré-entraîné
model = tf.keras.models.load_model('CNNNasNet.keras')
# last_conv_layer_name = 'conv_pw_13_relu'  # Nom de la dernière couche convolutionnelle de votre modèle
# last_conv_layer_name = 'block7a_expand_activation'
last_conv_layer_name = 'separable_conv_2_normal_left1_12'
img_size = (224, 224)

# Fonction pour générer la heatmap Grad-CAM avec intensification
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None, intensity=4):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0], axis=1).numpy()[0]
        class_channel = predictions[0][:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = heatmap * intensity
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Fonction pour superposer la heatmap Grad-CAM sur l'image originale
def overlay_gradcam(img_path, heatmap, alpha=0.3):
    img = Image.open(img_path)
    img = img.resize(img_size)
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

# Fonction pour générer l'explication LIME
def explain_with_lime(img_array, model):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(img_array[0], model.predict, top_labels=1, hide_color=0, num_samples=15)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,
                                                hide_rest=True)
    lime_img = mark_boundaries(temp, mask)
    return lime_img

# Interface de l'application
st.title("Classification d'Image avec Visualisation Interprétable")

# Ajout d'un sélecteur pour choisir la méthode d'interprétabilité
option = st.radio(
    "Choisissez la méthode d'interprétabilité que vous souhaitez afficher :",
    ('Grad-CAM', 'LIME'))

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"], key="uploader1")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize(img_size)
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    
    st.write(f"Classe prédite : {predicted_class_name}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption='Image originale', use_column_width=True)
    
    with col2:
        if option == 'Grad-CAM':
            # Générer la heatmap Grad-CAM
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, intensity=3)
            gradcam_image = overlay_gradcam(uploaded_file, heatmap, alpha=0.7)
            st.image(gradcam_image, caption=f'Image Grad-CAM pour {predicted_class_name}', use_column_width=True)
        
        elif option == 'LIME':
            # Générer l'explication LIME
            lime_img = explain_with_lime(img_array, model)
            st.image(lime_img, caption=f'Explication LIME pour {predicted_class_name}', use_column_width=True)
