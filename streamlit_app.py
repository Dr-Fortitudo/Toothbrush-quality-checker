import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Toothbrush_checker_model.h5") 
    return model

model = load_model()

st.title("Toothbrush Quality Checker 🪥")

uploaded_file = st.file_uploader("Upload an image of a toothbrush", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image to match model input
    img_resized = image.resize((224, 224))  # assuming the model was trained on 224x224 images
    img_array = np.array(img_resized) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Make prediction
    prediction = model.predict(img_array)
    class_names = ["Defective", "Good"]  # Adjust order based on your model
    predicted_class = class_names[np.argmax(prediction)]

    st.markdown(f"### 🔍 Verdict: **{predicted_class} Toothbrush**")
