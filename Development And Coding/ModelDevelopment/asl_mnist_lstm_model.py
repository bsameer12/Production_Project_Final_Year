import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

# Load dataset
train_df = pd.read_csv("/Users/sameer/Desktop/Production_Project_Final_Year/Development And Coding/archive/sign_mnist_train.csv")
test_df = pd.read_csv("/Users/sameer/Desktop/Production_Project_Final_Year/Development And Coding/archive/sign_mnist_test.csv")

# Label remapping (remove J/Z)
valid_labels = sorted(list(set(train_df['label']) | set(test_df['label'])))
label_map = {old: new for new, old in enumerate(valid_labels)}
train_df['label'] = train_df['label'].map(label_map)
test_df['label'] = test_df['label'].map(label_map)

# Prepare image data
x_train = train_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_df["label"].values
x_test = test_df.drop("label", axis=1).values.reshape(-1, 28, 28, 1) / 255.0
y_test = test_df["label"].values

# One-hot encoding
num_classes = len(label_map)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Simulate 10-frame sequences by repeating frames
x_train_seq = np.repeat(x_train[:, np.newaxis, :, :, :], 10, axis=1)
x_test_seq = np.repeat(x_test[:, np.newaxis, :, :, :], 10, axis=1)

# Model definition: CNN + LSTM + Attention
input_layer = layers.Input(shape=(10, 28, 28, 1))
cnn = layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu'))(input_layer)
cnn = layers.TimeDistributed(layers.MaxPooling2D((2, 2)))(cnn)
cnn = layers.TimeDistributed(layers.Flatten())(cnn)

# LSTM with attention-like self-query
lstm_out = layers.LSTM(128, return_sequences=True)(cnn)
attention = layers.Attention()([lstm_out, lstm_out])
lstm_out = layers.LSTM(64)(attention)
dense = layers.Dense(64, activation='relu')(lstm_out)
output = layers.Dense(num_classes, activation='softmax')(dense)

model = models.Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
history = model.fit(x_train_seq, y_train_cat,
                    validation_split=0.1,
                    epochs=50,
                    batch_size=64,
                    callbacks=[early_stop])

# Save the model in recommended format
model.save("asl_mnist_cnn_lstm_attention_model.keras")

# Plot accuracy & loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label="Train Accuracy")
plt.plot(history.history['val_accuracy'], label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label="Train Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate
test_loss, test_acc = model.evaluate(x_test_seq, y_test_cat)
print(f"\n✅ Final Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

# Confusion matrix
y_pred = model.predict(x_test_seq)
y_pred_classes = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)
labels = [chr(i + 65) for i in valid_labels]

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=labels)
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix - CNN + LSTM + Attention Model")
plt.show()

