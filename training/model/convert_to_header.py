# convert_to_header.py
def convert_to_header(tflite_file, header_file):
    # Read the tflite model file as binary
    with open(tflite_file, "rb") as f:
        tflite_data = f.read()
    
    # Create a C array as a string.
    c_array = "unsigned char model_data[] = {"
    # Process the binary data and convert each byte to a hex representation.
    hex_values = [f"0x{b:02x}" for b in tflite_data]
    # Split into lines for better readability.
    line_length = 16
    lines = []
    for i in range(0, len(hex_values), line_length):
        lines.append("  " + ", ".join(hex_values[i:i+line_length]))
    c_array += "\n" + ",\n".join(lines) + "\n};\n\n"
    
    # Also include the model length.
    c_array += f"unsigned int model_data_len = {len(tflite_data)};\n"
    
    # Write the C array to the header file.
    with open(header_file, "w") as f:
        f.write(c_array)
    print(f"Header file generated: {header_file}")

if __name__ == "__main__":
    convert_to_header("wristwatch_model.tflite", "model_data.h")
