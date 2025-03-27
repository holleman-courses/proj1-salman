import tensorflow as tf
import numpy as np

# Load your trained model.
model = tf.keras.models.load_model("wristwatch_model.h5")

# Create the TFLite converter.
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable optimizations for quantization.
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide a representative dataset to ensure full integer quantization.
def representative_dataset():
    for _ in range(100):
        # Create a random input sample with shape (1, 64, 64, 3) as float32.
        # Ideally, you should use samples from your training/validation data.
        sample_input = np.random.rand(1, 64, 64, 3).astype(np.float32)
        yield [sample_input]

converter.representative_dataset = representative_dataset

# Force the converter to use full integer quantization.
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensor types to uint8 (or int8, if preferred)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

# Convert the model.
tflite_model = converter.convert()

# Save the quantized model.
with open("wristwatch_model_quant.tflite", "wb") as f:
    f.write(tflite_model)
print("Quantized model saved as wristwatch_model_quant.tflite")

# Optionally, generate a header file for Arduino deployment.
def convert_to_header(tflite_file, header_file):
    with open(tflite_file, "rb") as f:
        tflite_data = f.read()
    
    # Create the C array string with PROGMEM so it goes into flash memory.
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

convert_to_header("wristwatch_model_quant.tflite", "model_data.h")
