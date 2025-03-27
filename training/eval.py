# eval.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import time

# Set the validation dataset directory and image dimensions.
val_dir = "C:/Users/pcsal/4127/project1/dataset/val"
img_height, img_width = 96, 96

# Load the trained model.
model = load_model("wristwatch_model.h5")

# Prepare the evaluation data generator.
eval_datagen = ImageDataGenerator(rescale=1.0/255)
eval_generator = eval_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Evaluate the model on the validation set.
loss, test_accuracy = model.evaluate(eval_generator, verbose=1)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Get predictions for all images.
predictions = model.predict(eval_generator)
y_true = np.array(eval_generator.labels)
y_pred = np.array([np.argmax(pred) for pred in predictions])

# Assuming class 1 ("watch") is the target.
TP = np.sum((y_true == 1) & (y_pred == 1))
FN = np.sum((y_true == 1) & (y_pred == 0))
TN = np.sum((y_true == 0) & (y_pred == 0))
FP = np.sum((y_true == 0) & (y_pred == 1))

FRR = FN / (TP + FN) if (TP + FN) > 0 else 0  # False Rejection Rate for target.
FPR = FP / (FP + TN) if (FP + TN) > 0 else 0  # False Positive Rate for target.

# Number of parameters.
num_params = model.count_params()

# Attempt to compute MACs using keras_flops, but catch errors if they occur.
try:
    import keras_flops
    macs = keras_flops.get_flops(model, batch_size=1)
except Exception as e:
    print("Could not compute MACs due to error:", e)
    macs = "N/A (error computing MACs)"

# Measure inference time on a single image (average over 50 runs).
eval_generator.reset()
batch_data, _ = next(eval_generator)  # use next() to get one batch
num_runs = 50
inference_times = []
for i in range(num_runs):
    start = time.time()
    _ = model.predict(batch_data)
    end = time.time()
    inference_times.append(end - start)
avg_inference_time = sum(inference_times) / len(inference_times)
fps = 1 / avg_inference_time if avg_inference_time > 0 else 0

# Input tensor shape (as defined by the model).
input_shape = (img_height, img_width, 3)

# Create and print summary table.
print("\nSummary Table:")
print("-------------------------------------------------------------")
print(f"Metric                          | Value")
print("-------------------------------------------------------------")
# Training and validation accuracies were printed in train.py.
print(f"Training Accuracy (TF)          | 0.9062")  # example value
print(f"Validation Accuracy (TF)        | 0.7645")  # example value
print(f"Test Accuracy (TF)              | {test_accuracy:.4f}")
print(f"False Rejection Rate (FRR)      | {FRR:.4f}")
print(f"False Positive Rate (FPR)       | {FPR:.4f}")
print(f"Number of Parameters            | {num_params}")
print(f"MACs                            | {macs}")
print(f"Input Tensor Shape              | {input_shape}")
print(f"Average Inference Time (ms)     | {avg_inference_time*1000:.2f}")
print(f"Sampling Rate (fps)             | {fps:.2f}")
print("-------------------------------------------------------------")
