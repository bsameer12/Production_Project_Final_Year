import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Set image dimensions
img_width, img_height = 64, 64  # You can adjust based on your dataset size

# Path to your dataset
train_dir = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_train'
test_image_path = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test/A/A_test.jpg'  # Replace with the correct path to your test image

# ImageDataGenerator for data augmentation (helps improve model generalization)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,       # Rescale pixel values to the range [0,1]
    rotation_range=20,   # Random rotations
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2, # Random vertical shift
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load and preprocess the data for training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',  # For multi-class classification
)

# Create a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=300,  # You can increase this for better accuracy
)

# Save the model after training
model.save('asl_sign_language_model.h5')  # You can specify any file name or path

# Now, let's predict for a new individual test image (e.g., 'A_test.jpg')
# Load and preprocess the test image
img = image.load_img(test_image_path, target_size=(img_width, img_height))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Rescale the image

# Predict the class (letter) for the input image
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=-1)

# Get the corresponding letter
class_labels = train_generator.class_indices
inv_class_labels = {v: k for k, v in class_labels.items()}
predicted_label = inv_class_labels[predicted_class[0]]

print(f'Predicted Sign: {predicted_label}')
