import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# Set image dimensions
img_width, img_height = 64, 64  # You can adjust based on your dataset size

# Path to your dataset
train_dir = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_train'
test_image_path = '../ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test/A_test.jpg'  # Replace with the correct path to your test image

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
    channel_shift_range=20.0
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Load and preprocess the data for training
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='categorical',  # For multi-class classification
)

# Using Pretrained MobileNetV2 model for feature extraction and fine-tuning
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

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
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('asl_sign_language_model.h5', save_best_only=True, save_weights_only=True)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=200,  # You can increase this for better accuracy
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
with open('class_labels.pkl', 'wb') as f:
    pickle.dump(class_labels, f)

# Save the model
model.save('asl_sign_language_model_MobileNetV2_and_data_augmentation.h5')  # Save the entire model
