import tensorflow as tf
import streamlit as st

model_path = "model.savedmodel"
loaded = tf.saved_model.load(model_path)

# Display input signature in Streamlit
signature = loaded.signatures["serving_default"].structured_input_signature
st.write("Model Input Signature:", signature)
