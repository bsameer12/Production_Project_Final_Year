import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import json

# Set image dimensions
img_width, img_height = 64, 64

# Path to your dataset
train_dir = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_train'
test_image_path = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test/A/A_test.jpg'

# Preprocessing function with histogram equalization and resizing
def preprocess_image(img_path, target_size=(64, 64)):
    # Read image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Histogram Equalization for contrast enhancement
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale for histogram equalization
    img = cv2.equalizeHist(img)  # Apply histogram equalization
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert back to RGB

    # Resize image to target size
    img = cv2.resize(img, target_size)

    # Normalize the image
    img = img.astype("float32") / 255.0

    return img

# ImageDataGenerator for data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,   # Increased rotation range
    width_shift_range=0.3,  # Increased horizontal shift
    height_shift_range=0.3, # Increased vertical shift
    shear_range=0.3,     # Increased shear range
    zoom_range=0.3,      # Increased zoom range
    brightness_range=[0.8, 1.2],  # Added brightness augmentation
    horizontal_flip=True,  # Flip images horizontally
    fill_mode='nearest',  # Fill the empty pixels after transformation
    channel_shift_range=30.0,   # Randomly shift RGB channels
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load and preprocess the data for training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',  # Multi-class classification
)

from tensorflow.keras.layers import Input

model = tf.keras.models.Sequential([
    # Input Layer
    Input(shape=(img_width, img_height, 3)),

    # First Convolution Layer
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Second Convolution Layer
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Third Convolution Layer
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Fourth Convolution Layer
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D((2, 2)),

    # Global Average Pooling (to handle varying image sizes)
    tf.keras.layers.GlobalAveragePooling2D(),

    # Fully Connected Layer
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),

    # Output Layer
    tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')
])


# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Train the model with early stopping and learning rate scheduler
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=3, min_lr=1e-6)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=200,  # You can increase this for better accuracy
    callbacks=[early_stopping, lr_scheduler]
)

# Save the model after training
model.save('asl_sign_language_model_optimized.h5')

# Save class labels to a dictionary
custom_cnn_labels = {value: key for key, value in train_generator.class_indices.items()}

# Optionally, save the labels to a file (e.g., JSON)
with open('custom_cnn_labels.json', 'w') as f:
    json.dump(custom_cnn_labels, f)

# Now, let's predict for a new individual test image (e.g., 'A_test.jpg')
img = preprocess_image(test_image_path, target_size=(img_width, img_height))  # Use preprocessed image
predictions = model.predict(np.expand_dims(img, axis=0))
predicted_class = np.argmax(predictions, axis=-1)

# Get the corresponding label
inv_class_labels = {v: k for k, v in custom_cnn_labels.items()}
predicted_label = inv_class_labels[predicted_class[0]]

print(f'Predicted Sign: {predicted_label}')
