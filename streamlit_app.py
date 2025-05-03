import tensorflow as tf

model_path = "model.savedmodel"
loaded = tf.saved_model.load(model_path)

# List available input keys
print(loaded.signatures["serving_default"].structured_input_signature)
