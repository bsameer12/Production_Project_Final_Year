import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('asl_sign_language_model.h5')

# Path to the test image
test_image_path = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test/Z/Z_test.jpg'

# Set image dimensions
img_width, img_height = 64, 64  # You can adjust based on your dataset size

# Load and preprocess the test image
img = image.load_img(test_image_path, target_size=(img_width, img_height))  # Resize image to match model input size
img_array = image.img_to_array(img)  # Convert the image to a numpy array
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (model expects a batch of images)
img_array /= 255.0  # Rescale the image

# Predict with the loaded model
predictions = loaded_model.predict(img_array)
predicted_class = np.argmax(predictions, axis=-1)

# Get the class labels from the training generator
class_labels = {0: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 1: '10',
                10: 'A', 11: 'B', 12: 'C', 13: 'D', 36: 'DELETE', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
                18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 37: 'Nothing', 24: 'O' ,25: 'P', 26: 'Q', 27: 'R', 28: 'S', 38 : 'SPACE',
                29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z' }

# Map the predicted class index to the corresponding label
predicted_label = class_labels[predicted_class[0]]

# Print the predicted label
print(f'Predicted Sign: {predicted_label}')

