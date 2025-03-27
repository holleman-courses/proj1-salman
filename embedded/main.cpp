#include <Arduino.h>
#include "TensorFlowLite.h"       
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model_data.h"  // Your fully quantized model header with PROGMEM

// Define a tensor arena size; adjust as needed.
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Updated image dimensions for the simplified model.
constexpr int kImageWidth = 64;
constexpr int kImageHeight = 64;
constexpr int kImageChannels = 3;
constexpr int kImageBufferSize = kImageWidth * kImageHeight * kImageChannels;

// Global pointers for TensorFlow Lite objects.
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port connection.
  }
  Serial.println("Starting Inference for Quantized Model (64x64 Input)...");

  // Set up error reporter.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Load the model from the embedded C array.
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version (%d) does not match supported version (%d).",
                           model->version(), TFLITE_SCHEMA_VERSION);
    while (1);
  }
  Serial.println("Model loaded successfully.");

  // Create an op resolver.
  static tflite::AllOpsResolver resolver;

  // Build the interpreter.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory for the model's tensors.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed.");
    while (1);
  }
  Serial.println("Tensors allocated.");

  // Get pointers to the model's input and output tensors.
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Print the input tensor shape.
  Serial.print("Input tensor shape: ");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.print(input_tensor->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  // Print quantization parameters for debugging.
  if (input_tensor->type == kTfLiteInt8) {
    Serial.print("Input scale: ");
    Serial.println(input_tensor->params.scale);
    Serial.print("Input zero point: ");
    Serial.println(input_tensor->params.zero_point);
  }
  if (output_tensor->type == kTfLiteInt8) {
    Serial.print("Output scale: ");
    Serial.println(output_tensor->params.scale);
    Serial.print("Output zero point: ");
    Serial.println(output_tensor->params.zero_point);
  }
}

// This function simulates capturing an image for a quantized model.
// 'desired_norm' is a normalized brightness value (0 to 1) you wish to simulate.
bool captureAndPreprocessImage(float desired_norm) {
  int num_pixels = kImageBufferSize;
  if (input_tensor->type == kTfLiteInt8) {
    int8_t* input_data = input_tensor->data.int8;
    float scale = input_tensor->params.scale;
    int zero_point = input_tensor->params.zero_point;
    // Compute the quantized value corresponding to desired_norm.
    int8_t quantized_value = int8_t(round(desired_norm / scale)) + zero_point;
    for (int i = 0; i < num_pixels; i++) {
      input_data[i] = quantized_value;
    }
  } else {
    // Fallback if model isn't quantized (should not happen in this case).
    float* input_data = input_tensor->data.f;
    for (int i = 0; i < num_pixels; i++) {
      input_data[i] = desired_norm;
    }
  }
  return true;
}

void loop() {
  Serial.println("----- New Inference Cycle -----");

  // Simulate an image with normalized brightness 0.8.
  if (!captureAndPreprocessImage(0.8)) {
    Serial.println("Image capture failed, skipping inference.");
    delay(1000);
    return;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed.");
    delay(1000);
    return;
  }

  float watch_probability = 0.0;
  float notwatch_probability = 0.0;
  if (output_tensor->type == kTfLiteInt8) {
    int8_t* output_data = output_tensor->data.int8;
    float out_scale = output_tensor->params.scale;
    int out_zero_point = output_tensor->params.zero_point;
    watch_probability = (output_data[0] - out_zero_point) * out_scale;
    notwatch_probability = (output_data[1] - out_zero_point) * out_scale;
  } else {
    watch_probability = output_tensor->data.f[0];
    notwatch_probability = output_tensor->data.f[1];
  }

  Serial.print("Watch probability: ");
  Serial.println(watch_probability, 4);
  Serial.print("Not Watch probability: ");
  Serial.println(notwatch_probability, 4);

  if (watch_probability > notwatch_probability) {
    Serial.println("Detected: Wristwatch");
  } else {
    Serial.println("Detected: Not Watch");
  }

  delay(2000);
}
