import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

# Load the trained model
MODEL_PATH = "Toothbrush_checker_model.h5"
model = load_model(MODEL_PATH)

st.title("Toothbrush Quality Checker 🪥")

uploaded_file = st.file_uploader("Upload an image of a toothbrush", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3-channel RGB
    image = image.resize((150, 150))  # Fixed size to match model input (150x150)
    img_array = np.array(image, dtype=np.float32) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    try:
        # Preprocess the image
        img_array = preprocess_image(image)

        # Show a loading spinner during prediction
        with st.spinner("Analyzing toothbrush quality..."):
            prediction = model.predict(img_array)
            class_names = ["Good", "Defective"]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100


st.markdown("### 🔍 Verdict: **{predicted_class} Toothbrush** ({confidence:.2f}% confidence)")
