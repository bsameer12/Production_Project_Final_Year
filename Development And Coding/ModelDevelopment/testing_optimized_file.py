import tensorflow as tf
import numpy as np
import cv2
import json

# Set image dimensions
img_width, img_height = 64, 64

# Path to the pre-trained model and class labels file
model_path = 'asl_sign_language_model_optimized.h5'
labels_path = 'custom_cnn_labels.json'

# Path to the test image
test_image_path = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test/1/e8ade0e3-8c75-409f-b2cc-5305b27113ef.rgb_0000.png'

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class labels from the saved JSON file
with open(labels_path, 'r') as f:
    class_labels = json.load(f)

# Preprocessing function with histogram equalization and resizing
def preprocess_image(img_path, target_size=(64, 64)):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for histogram equalization
    img = cv2.equalizeHist(img)  # Apply histogram equalization
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert back to RGB
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img

# Load and preprocess the test image
img = preprocess_image(test_image_path, target_size=(img_width, img_height))

# Predict on the test image
predictions = model.predict(np.expand_dims(img, axis=0))
predicted_class = np.argmax(predictions, axis=-1)

# Ensure class_labels are in the correct format and match the model output
# Debugging: print the class labels
print(f"Class Labels: {class_labels}")

# Invert the class labels to get the corresponding class name
inv_class_labels = {int(k): v for k, v in class_labels.items()}  # Convert keys to int for matching

# Debugging: print available class labels
print(f"Available class labels: {inv_class_labels}")


# Debugging: print the predicted class index
print(f"Predicted class index: {predicted_class[0]}")


# Now, try to get the predicted label
try:
    predicted_class_idx = predicted_class[0]  # Index from model prediction
    predicted_label = inv_class_labels[predicted_class_idx]  # Look up using the index
    print(f'Predicted Sign: {predicted_label}')
except KeyError as e:
    print(f"KeyError: {e}. This means the predicted class index is not in the class labels.")
