import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
from keras.layers import TFSMLayer
from keras import Sequential

# Load the trained model
MODEL_PATH = "model.savedmodel"
layer = TFSMLayer(MODEL_PATH, call_endpoint="serving_default")
model = Sequential([layer])

loaded = tf.saved_model.load(MODEL_PATH)
print(loaded.signatures.keys())
