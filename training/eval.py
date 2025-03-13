import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os

# dataset path
data_dir="dataset"
img_size=(224, 224)

# load the trained model
model=load_model("model.h5")

# preprocess the data for the evaluation
datagen = ImageDataGenerator(rescale=1./255)

eval_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=1,
    class_mode="binary",
    shuffle=False
)

# Evaluationg the model
loss,accuracy= model.evaluate(eval_data, verbose=1)
print(f"\nTest Accuracy: {accuracy:.4f}")

# Get class labels
class_indices = eval_data.class_indices
class_labels = {v: k for k, v in class_indices.items()} 

# display results for each image prediction
print("\nPredictions:")
filenames=eval_data.filenames
predictions=model.predict(eval_data)

for i, pred in enumerate(predictions):
    predicted_class = class_labels[round(pred[0])] 
    print(f"Image: {filenames[i]} --> Predicted as: {predicted_class} (Confidence: {pred[0]:.4f})")
