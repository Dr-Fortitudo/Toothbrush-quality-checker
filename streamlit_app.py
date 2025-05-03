import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.layers import TFSMLayer
from keras import Sequential

# Load the trained model using TFSMLayer
MODEL_PATH = "model.savedmodel"
layer = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
model = Sequential([layer])

# Streamlit UI
st.title("Toothbrush Quality Checker 🪥")

uploaded_file = st.file_uploader("Upload an image of a toothbrush", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))  # Match model's expected input size
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    try:
        img_array = preprocess_image(image)

        with st.spinner("Analyzing toothbrush quality..."):
            prediction = model({"sequential_9_input": img_array})  # Use correct input name
            class_names = ["Good", "Defective"]
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

        st.markdown(f"### 🔍 Verdict: **{predicted_class} Toothbrush** ({confidence:.2f}% confidence)")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
