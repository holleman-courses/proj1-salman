import tensorflow as tf
import numpy as np
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D,
                                     BatchNormalization, Activation,
                                     MaxPooling2D, GlobalAveragePooling2D,
                                     Flatten, Dense)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def enhanced_cnn():
    # Input: 64x64 RGB images.
    inputs = Input(shape=(64, 64, 3))
    
    # Block 1: Standard convolution (downsampling)
    x = Conv2D(8, kernel_size=3, strides=2, padding='same',
               kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Output shape: 32x32x8
    
    # Block 2: Depthwise separable convolution block
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                        depthwise_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, kernel_size=1, strides=1, padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Output shape: 32x32x16
    
    # Block 3: Another depthwise separable block with downsampling.
    x = DepthwiseConv2D(kernel_size=3, strides=2, padding='same',
                        depthwise_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=1, strides=1, padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Output shape: 16x16x32
    
    # Block 4: One more depthwise separable block (without downsampling)
    x = DepthwiseConv2D(kernel_size=3, strides=1, padding='same',
                        depthwise_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=1, strides=1, padding='same',
               kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Output shape: 16x16x32
    
    # Global average pooling and final classification.
    x = GlobalAveragePooling2D()(x)  # shape: (batch, 32)
    outputs = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Dataset paths (update as needed)
train_dir = "C:/Users/pcsal/4127/project1/dataset/train"
val_dir   = "C:/Users/pcsal/4127/project1/dataset/val"

# Training parameters.
img_height, img_width = 64, 64
batch_size = 32
epochs = 20

# Data generators.
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Build and compile the model.
model = enhanced_cnn()
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
print("Total parameters:", model.count_params())

history = model.fit(
    train_generator,
    steps_per_epoch=math.ceil(train_generator.samples / batch_size),
    validation_data=val_generator,
    validation_steps=math.ceil(val_generator.samples / batch_size),
    epochs=epochs
)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

model.save("wristwatch_model.h5")
print("Model saved as wristwatch_model.h5")

###############################################
# Now, convert the saved model to a fully quantized TFLite model.
###############################################

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for _ in range(100):
        sample_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
        yield [sample_input]

converter.representative_dataset = representative_dataset

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

quant_model_file = "wristwatch_model_quant.tflite"
with open(quant_model_file, "wb") as f:
    f.write(tflite_model)
print("Quantized model saved as", quant_model_file)

###############################################
# Generate a C header file from the quantized TFLite model.
###############################################

def convert_to_header(tflite_file, header_file):
    with open(tflite_file, "rb") as f:
        tflite_data = f.read()
    
    c_array = "#include <Arduino.h>\n\n"
    c_array += "const unsigned char model_data[] PROGMEM = {\n"
    hex_values = [f"0x{b:02x}" for b in tflite_data]
    line_length = 16
    lines = []
    for i in range(0, len(hex_values), line_length):
        lines.append("  " + ", ".join(hex_values[i:i+line_length]))
    c_array += ",\n".join(lines) + "\n};\n\n"
    c_array += f"const unsigned int model_data_len = {len(tflite_data)};\n"
    
    with open(header_file, "w") as f:
        f.write(c_array)
    print(f"Header file generated: {header_file}")

header_file = "model_data.h"
convert_to_header(quant_model_file, header_file)
