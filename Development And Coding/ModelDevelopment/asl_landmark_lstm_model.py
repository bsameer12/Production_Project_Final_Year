import os
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

# Paths to training and test image folders
train_dir = '/Users/sameer/Desktop/Production_Project_Final_Year/Development And Coding/ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_train'
test_dir = '/Users/sameer/Desktop/Production_Project_Final_Year/Development And Coding/ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test'

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Data augmentation (rotation, brightness adjustment)
def augment_image(image):
    angle = np.random.uniform(-10, 10)
    brightness = np.random.uniform(0.7, 1.3)
    M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    image = cv2.warpAffine(image, M, (224, 224))
    image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    return image

# Extract normalized landmarks (63 values)
def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = augment_image(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]
        normed = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in hand.landmark]
        return [coord for pt in normed for coord in pt]
    return None

# Load data with consistent label map and extract features
def prepare_data_from_folder(folder_path, existing_label_map=None):
    data, labels = [], []
    label_map = existing_label_map or {}
    current_label = max(label_map.values()) + 1 if existing_label_map else 0
    for subfolder in sorted(os.listdir(folder_path)):
        sub_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(sub_path): continue
        if subfolder not in label_map:
            if existing_label_map: continue
            label_map[subfolder] = current_label
            current_label += 1
        print(f"📂 Loading: {subfolder}")
        count = 0
        for img_file in os.listdir(sub_path):
            if count >= 1000: break
            img_path = os.path.join(sub_path, img_file)
            landmarks = extract_landmarks(img_path)
            if landmarks:
                data.append(landmarks)
                labels.append(label_map[subfolder])
                count += 1
    return np.array(data), np.array(labels), label_map

# Load and prepare data
x_data, y_data, label_map = prepare_data_from_folder(train_dir)
num_classes = len(label_map)
y_data_cat = to_categorical(y_data, num_classes)

# Stratified train/val split
x_train, x_val, y_train_cat, y_val_cat = train_test_split(x_data, y_data_cat, test_size=0.1, stratify=y_data, random_state=42)

# Repeat for temporal modeling
x_train_seq = np.repeat(x_train[:, np.newaxis, :], 10, axis=1)
x_val_seq = np.repeat(x_val[:, np.newaxis, :], 10, axis=1)

# Build Functional model with Attention
input_layer = layers.Input(shape=(10, x_train.shape[1]))
lstm_out = layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(input_layer)
attention_out = layers.Attention()([lstm_out, lstm_out])
dropout_1 = layers.Dropout(0.4)(attention_out)
lstm_out_2 = layers.LSTM(64, kernel_regularizer=regularizers.l2(0.001))(dropout_1)
dropout_2 = layers.Dropout(0.3)(lstm_out_2)
dense_1 = layers.Dense(64, activation='relu')(dropout_2)
output_layer = layers.Dense(num_classes, activation='softmax')(dense_1)

model = models.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.summary()

# Train model
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
history = model.fit(x_train_seq, y_train_cat, validation_data=(x_val_seq, y_val_cat),
                    epochs=50, batch_size=64,
                    callbacks=[early_stop, lr_schedule])

# Save model
model.save("asl_landmark_lstm_model.keras")

# Accuracy / Loss Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Over Epochs")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Load and evaluate on test data
x_test, y_test, _ = prepare_data_from_folder(test_dir, existing_label_map=label_map)
y_test_cat = to_categorical(y_test, num_classes)
x_test_seq = np.repeat(x_test[:, np.newaxis, :], 10, axis=1)

test_loss, test_acc, test_prec, test_rec = model.evaluate(x_test_seq, y_test_cat)
print(f"\n📁 Final Test Accuracy: {test_acc:.4f} | Loss: {test_loss:.4f}")
print(f"📌 Precision: {test_prec:.4f} | Recall: {test_rec:.4f}")

# Predictions & analysis
y_pred = model.predict(x_test_seq)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_cat, axis=1)
label_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]

# Debug: Print mismatch info
print(f"🧪 Unique y_true labels: {np.unique(y_true)}")
print(f"🧪 Number of target names: {len(label_names)}")

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label_names, labels=np.unique(y_true)))

conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Filter label names to only include those in y_true
filtered_labels = [label_names[i] for i in np.unique(y_true)]

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=filtered_labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - ASL Landmark Model")
plt.show()

# Confidence plots
confidence_scores = np.max(y_pred, axis=1)
predicted_labels = [label_names[i] for i in y_pred_classes]

plt.figure(figsize=(10, 5))
sns.histplot(confidence_scores, bins=30, kde=True, color='skyblue')
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.boxplot(x=predicted_labels, y=confidence_scores, palette="Spectral")
plt.xticks(rotation=90)
plt.title("Confidence per Predicted Class")
plt.xlabel("Predicted Label")
plt.ylabel("Confidence Score")
plt.tight_layout()
plt.show()
