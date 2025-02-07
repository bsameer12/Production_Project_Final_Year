import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Set image dimensions
img_width, img_height = 64, 64  # You can adjust based on your dataset size

# Path to your dataset
train_dir = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_train'
test_image_path = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test/A/A_test.jpg'  # Replace with the correct path to your test image

# ImageDataGenerator for data augmentation
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,       # Rescale pixel values to the range [0,1]
    rotation_range=20,   # Random rotations
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2, # Random vertical shift
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.2, 1.0],  # Randomly adjust brightness
    channel_shift_range=20.0  # Randomly change brightness in each color channel
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load and preprocess the data for training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',  # For multi-class classification
)

# Print the discovered classes
print("Discovered Classes:", train_generator.class_indices)

# List of classes discovered in the dataset
class_names = list(train_generator.class_indices.keys())
print("Class Names:", class_names)

# Using Pretrained ResNet50 model for feature extraction and fine-tuning
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Add custom layers for the ASL dataset
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
x = Dense(train_generator.num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Freeze the layers of the pre-trained model (we will only train the top layers)
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Set up EarlyStopping and ModelCheckpoint callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)

# Saving the entire model (architecture + weights + optimizer) during training
checkpoint = ModelCheckpoint('asl_sign_language_resnet_model.h5', save_best_only=True, save_weights_only=False)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=220,  # You can increase this for better accuracy
    validation_data=test_datagen.flow_from_directory(
        '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test',
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical',
    ),
    validation_steps=10,
    callbacks=[early_stop, checkpoint]  # Add callbacks here
)

# Save the class labels (mapping between index and label)
class_labels = train_generator.class_indices
with open('class_labels_resnet.pkl', 'wb') as f:
    pickle.dump(class_labels, f)

# Save the model
model.save('asl_sign_language_resnet_model_using_ResNet50.h5')  # Save the entire model

# Load and preprocess the test image
test_image = image.load_img(test_image_path, target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image / 255.0  # Rescale to the range [0, 1]

# Predict the class of the test image
predictions = model.predict(test_image)
predicted_class = np.argmax(predictions, axis=1)
predicted_label = class_names[predicted_class[0]]

# Print the prediction result
print(f"Predicted Class: {predicted_label}")
