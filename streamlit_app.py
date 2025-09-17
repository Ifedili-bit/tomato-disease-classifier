import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("plant_disease_model.h5")
    return model

model = load_model()

# Class names (same order as training)
class_names = [
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# App title
st.title("🍅 Tomato Leaf Disease Classifier")
st.markdown("Upload a tomato leaf image and let AI predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))   # MobileNetV2 expects 224x224
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    class_id = np.argmax(score)
    confidence = 100 * np.max(score)

    # Show result
    st.subheader(f"Prediction: **{class_names[class_id]}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
