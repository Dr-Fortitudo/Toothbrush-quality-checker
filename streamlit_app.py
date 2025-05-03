import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf

loaded = tf.saved_model.load(MODEL_PATH)
print(loaded.signatures.keys())
