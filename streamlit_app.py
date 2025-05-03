import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
MODEL_PATH = "model.savedmodel"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
import os

print("Current working directory:", os.getcwd())
print("Files/folders:", os.listdir())
