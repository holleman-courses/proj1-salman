#include <Arduino.h>
#undef swap  // Undefine any swap macro that might conflict

// Use the Harvard TinyMLx library's camera interface.
#include <Arduino_OV767X_TinyMLx.h>  // This header comes with Harvard_TinyMLx

#include "TensorFlowLite.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "model_data.h"  // Your fully quantized model header with PROGMEM


// The image provider code will use the global "Camera" object defined by TinyMLShield.
//extern TinyMLShield Camera;  // Provided by the Harvard_TinyMLx library

// Tensor arena size (adjust as needed)
constexpr int kTensorArenaSize = 100 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

// Our model expects a 96x96 grayscale image.
constexpr int kModelWidth = 64;
constexpr int kModelHeight = 64;
constexpr int kModelChannels = 3; // grayscale
constexpr int kImageBufferSize = kModelWidth * kModelHeight * kModelChannels;

// Global TFLite pointers.
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

TfLiteStatus GetImage(tflite::ErrorReporter* error_reporter, int image_width,
  int image_height, int channels, int8_t* image_data) {
// For QCIF grayscale, the camera returns 176x144 data.
const int qcif_width = 176;
const int qcif_height = 144;
byte data[qcif_width * qcif_height];  // raw camera data

static bool g_is_camera_initialized = false;
if (!g_is_camera_initialized) {
// Initialize camera: Use QCIF, GRAYSCALE, 5 fps, OV7675 module.
if (!Camera.begin(QCIF, GRAYSCALE, 5, OV7675)) {
TF_LITE_REPORT_ERROR(error_reporter, "Failed to initialize camera!");
return kTfLiteError;
}
g_is_camera_initialized = true;
}

// Capture a frame.
Camera.readFrame(data);

// Debug: print a few raw pixel values before cropping.
Serial.print("Raw pixel [0]: ");
Serial.println(data[0]);
Serial.print("Raw pixel [50]: ");
Serial.println(data[50]);

// Crop the center 96x96 region.
int min_x = (qcif_width - image_width) / 2;
int min_y = (qcif_height - image_height) / 2;
int index = 0;

// To get some statistics, compute min, max, and sum.
int min_pixel = 255;
int max_pixel = 0;
unsigned long sum = 0;

for (int y = min_y; y < min_y + image_height; y++) {
for (int x = min_x; x < min_x + image_width; x++) {
int pixel = data[y * qcif_width + x];
if (pixel < min_pixel) min_pixel = pixel;
if (pixel > max_pixel) max_pixel = pixel;
sum += pixel;
// Convert to signed int8 (subtract 128)
image_data[index++] = static_cast<int8_t>(pixel) - 128;
}
}

float avg = sum / float(image_width * image_height);
Serial.print("Cropped image min pixel: ");
Serial.println(min_pixel);
Serial.print("Cropped image max pixel: ");
Serial.println(max_pixel);
Serial.print("Cropped image average pixel: ");
Serial.println(avg);

return kTfLiteOk;
}


void setup() {
  Serial.begin(115200);
  while (!Serial) { }
  Serial.println("Starting Inference with TinyMLShield and Quantized Model...");

  // Set up error reporter.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Load the model.
  const tflite::Model* model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema version (%d) does not match supported version (%d).",
                           model->version(), TFLITE_SCHEMA_VERSION);
    while (1);
  }
  Serial.println("Model loaded successfully.");

  // Create op resolver.
  static tflite::AllOpsResolver resolver;

  // Build the interpreter.
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate tensors.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    error_reporter->Report("AllocateTensors() failed.");
    while (1);
  }
  Serial.println("Tensors allocated.");

  // Get pointers to input and output tensors.
  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  // Print input tensor shape.
  Serial.print("Input tensor shape: ");
  for (int i = 0; i < input_tensor->dims->size; i++) {
    Serial.print(input_tensor->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  // Print quantization parameters.
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
  
  Serial.println("Setup complete.");
}

void loop() {
  Serial.println("----- New Inference Cycle -----");

  // Get an image from the camera.
  if (GetImage(nullptr, kModelWidth, kModelHeight, kModelChannels, input_tensor->data.int8) != kTfLiteOk) {
    Serial.println("Image capture failed, skipping inference.");
    delay(1000);
    return;
  }
  
  // Run inference.
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Inference failed.");
    delay(1000);
    return;
  }
  
  // Dequantize outputs.
  int8_t* output_data = output_tensor->data.int8;
  float out_scale = output_tensor->params.scale;
  int out_zero_point = output_tensor->params.zero_point;
  float watch_probability = (output_data[0] - out_zero_point) * out_scale;
  float notwatch_probability = (output_data[1] - out_zero_point) * out_scale;
  
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
