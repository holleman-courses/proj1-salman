import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# dataset paths
data_dir = "dataset"
img_size = (224, 224)
# load the pre-trained MobileNetV2 model ( I choose this model because it predict well in case of a small dataset)
base_model = keras.applications.MobileNetV2(
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False  # Remove classification head
)
# freeze base model layers
base_model.trainable = False

# add a custom classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),  # Reduce parameters
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")  # Binary classification
])
# compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# data preprocessing with augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=2,
    class_mode="binary",
    subset="training"
)
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=2,
    class_mode="binary",
    subset="validation"
)
# train the model
model.fit(train_data, validation_data=val_data, epochs=10, verbose=1)
model.save("model.h5")
print("Training with MobileNetV2 completed. Model saved as model.h5")
