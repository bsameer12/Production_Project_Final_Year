import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle

# Set image dimensions
img_width, img_height = 64, 64  # You can adjust based on your dataset size

# Load the trained model
def load_model():
    model = tf.keras.models.load_model('asl_sign_language_model_MobileNetV2_and_data_augmentation.h5')  # Load the entire model
    return model

# Load class labels for predictions
def load_class_labels():
    with open('class_labels.pkl', 'rb') as f:
        class_labels = pickle.load(f)
    inv_class_labels = {v: k for k, v in class_labels.items()}  # Reverse the dictionary for easy lookup
    return inv_class_labels

# Function for making predictions on a new image
def predict_sign_language(model, inv_class_labels, img_path):
    # Load and preprocess the image for prediction
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale the image

    # Predict the class (letter) for the input image
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=-1)

    # Get the corresponding label
    predicted_label = inv_class_labels[predicted_class[0]]

    return predicted_label

# Load the model and class labels
model = load_model()
inv_class_labels = load_class_labels()

# Path to the test image (update this with your own test image path)
test_image_path = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test/A_test.jpg'  # Replace with your test image path

# Predict the sign for the test image
predicted_label = predict_sign_language(model, inv_class_labels, test_image_path)

print(f'Predicted Sign: {predicted_label}')
