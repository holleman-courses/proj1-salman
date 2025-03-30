import tensorflow as tf

# Load the trained Keras model.
model = tf.keras.models.load_model("wristwatch_model.h5")

# Convert the model to TensorFlow Lite format.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model.
with open("wristwatch_model.tflite", "wb") as f:
    f.write(tflite_model)
print("Model converted to TFLite and saved as wristwatch_model.tflite")
