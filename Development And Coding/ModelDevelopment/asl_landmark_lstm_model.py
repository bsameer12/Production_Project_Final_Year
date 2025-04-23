import os
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    log_loss,
    balanced_accuracy_score,
    matthews_corrcoef,
    cohen_kappa_score,
    average_precision_score
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf

# ─── Paths to data ─────────────────────────────────────────────────────────────
train_dir = '/Users/sameer/Desktop/Production_Project_Final_Year/Development And Coding/ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_train'
test_dir  = '/Users/sameer/Desktop/Production_Project_Final_Year/Development And Coding/ASL(American_Sign_Language)_Alphabet_Dataset/ASL_Alphabet_Dataset/asl_alphabet_test'

# ─── MediaPipe hand detector ───────────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands    = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# ─── Data augmentation ─────────────────────────────────────────────────────────
def augment_image(image):
    angle      = np.random.uniform(-10, 10)
    brightness = np.random.uniform(0.7, 1.3)
    M = cv2.getRotationMatrix2D((112, 112), angle, 1.0)
    image = cv2.warpAffine(image, M, (224, 224))
    image = np.clip(image * brightness, 0, 255).astype(np.uint8)
    return image

# ─── Landmark extraction ───────────────────────────────────────────────────────
def extract_landmarks(image_path):
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, (224, 224))
    img = augment_image(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        hand  = results.multi_hand_landmarks[0]
        wrist = hand.landmark[0]
        normed = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in hand.landmark]
        return [coord for pt in normed for coord in pt]
    return None

# ─── Load & prepare data ───────────────────────────────────────────────────────
def prepare_data_from_folder(folder_path, existing_label_map=None, max_per_class=1000):
    data, labels = [], []
    label_map    = existing_label_map or {}
    counter      = max(label_map.values()) + 1 if existing_label_map else 0

    for subfolder in sorted(os.listdir(folder_path)):
        sub_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(sub_path): continue
        if subfolder not in label_map:
            if existing_label_map: continue
            label_map[subfolder] = counter
            counter += 1

        print(f"📂 Loading: {subfolder}")
        count = 0
        for img_file in os.listdir(sub_path):
            if count >= max_per_class: break
            lm = extract_landmarks(os.path.join(sub_path, img_file))
            if lm is not None:
                data.append(lm)
                labels.append(label_map[subfolder])
                count += 1

    return np.array(data), np.array(labels), label_map

# ─── Prepare training/validation ───────────────────────────────────────────────
x_data, y_data, label_map = prepare_data_from_folder(train_dir)
num_classes               = len(label_map)
y_data_cat                = to_categorical(y_data, num_classes)

x_train, x_val, y_train_cat, y_val_cat = train_test_split(
    x_data, y_data_cat, test_size=0.1, stratify=y_data, random_state=42
)

x_train_seq = np.repeat(x_train[:, np.newaxis, :], 10, axis=1)
x_val_seq   = np.repeat(x_val[:,   np.newaxis, :], 10, axis=1)

# ─── Build model ────────────────────────────────────────────────────────────────
inp = layers.Input(shape=(10, x_train.shape[1]))
l1  = layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(0.001))(inp)
att = layers.Attention()([l1, l1])
d1  = layers.Dropout(0.4)(att)
l2  = layers.LSTM(64, kernel_regularizer=regularizers.l2(0.001))(d1)
d2  = layers.Dropout(0.3)(l2)
h   = layers.Dense(64, activation='relu')(d2)
out = layers.Dense(num_classes, activation='softmax')(h)

model = models.Model(inputs=inp, outputs=out)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
model.summary()

# ─── Train ─────────────────────────────────────────────────────────────────────
es  = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
rp  = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)

history = model.fit(
    x_train_seq, y_train_cat,
    validation_data=(x_val_seq, y_val_cat),
    epochs=50, batch_size=64,
    callbacks=[es, rp]
)
model.save("asl_landmark_lstm_model.keras")

# ─── Training Curves ────────────────────────────────────────────────────────────
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'],   label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title("Accuracy Over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Over Epochs"); plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.tight_layout(); plt.show()

# ─── Prepare test data ────────────────────────────────────────────────────────
x_test, y_test, _   = prepare_data_from_folder(test_dir, existing_label_map=label_map)
y_test_cat          = to_categorical(y_test, num_classes)
x_test_seq          = np.repeat(x_test[:, np.newaxis, :], 10, axis=1)

# ─── 1) Core Metrics & Advanced Metrics ────────────────────────────────────────
test_loss, test_acc, test_prec, test_rec = model.evaluate(x_test_seq, y_test_cat, verbose=0)
y_pred_prob = model.predict(x_test_seq)
y_pred_cls  = np.argmax(y_pred_prob, axis=1)
y_true_cls  = np.argmax(y_test_cat,  axis=1)

# Compute additional metrics
ll        = log_loss(y_test_cat,       y_pred_prob)
bal_acc   = balanced_accuracy_score(y_true_cls, y_pred_cls)
mcc       = matthews_corrcoef(y_true_cls, y_pred_cls)
kappa     = cohen_kappa_score(y_true_cls, y_pred_cls)
ap_macro  = average_precision_score(y_test_cat, y_pred_prob, average='macro')

def top_k_accuracy(y_true, y_prob, k=3):
    topk = np.argsort(y_prob, axis=1)[:, -k:]
    return np.mean([y_true[i] in topk[i] for i in range(len(y_true))])

top3_acc = top_k_accuracy(y_true_cls, y_pred_prob, k=3)

# Print them
print("\n" + "="*40)
print(f"Test Loss           : {test_loss:.4f}")
print(f"Test Accuracy       : {test_acc:.4f}")
print(f"Precision (macro)   : {test_prec:.4f}")
print(f"Recall    (macro)   : {test_rec:.4f}")
print(f"Log Loss            : {ll:.4f}")
print(f"Balanced Accuracy   : {bal_acc:.4f}")
print(f"MCC                 : {mcc:.4f}")
print(f"Cohen's Kappa       : {kappa:.4f}")
print(f"Avg Precision       : {ap_macro:.4f}")
print(f"Top-3 Accuracy      : {top3_acc:.4f}")
print("="*40 + "\n")

# ─── 2) Confusion Matrix ────────────────────────────────────────────────────────
labels_pres = np.unique(y_true_cls)
names_pres  = [k for k,v in sorted(label_map.items(), key=lambda x:x[1]) if v in labels_pres]
cm = confusion_matrix(y_true_cls, y_pred_cls, labels=labels_pres)

plt.figure(figsize=(10,8))
disp = ConfusionMatrixDisplay(cm, display_labels=names_pres)
disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
plt.title("Confusion Matrix — Test Set")
plt.xlabel("Predicted Label"); plt.ylabel("True Label")
plt.tight_layout(); plt.show()

# ─── 3) Per-Class Bar Chart ────────────────────────────────────────────────────
report = classification_report(
    y_true_cls, y_pred_cls,
    labels=labels_pres,
    target_names=names_pres,
    output_dict=True
)
metrics = ['precision','recall','f1-score']
x = np.arange(len(names_pres))
width = 0.3

plt.figure(figsize=(14,5))
for i, m in enumerate(metrics):
    vals = [report[name][m] for name in names_pres]
    plt.bar(x + i*width, vals, width, label=m.capitalize())
plt.xticks(x + width, names_pres, rotation=90)
plt.ylim(0,1.05)
plt.title("Per-Class Precision, Recall & F1-Score")
plt.xlabel("Class"); plt.ylabel("Score"); plt.legend()
plt.tight_layout(); plt.show()

# ─── 4) ROC Curves ─────────────────────────────────────────────────────────────
plt.figure(figsize=(8,8))
for idx, cls in enumerate(labels_pres):
    fpr, tpr, _ = roc_curve(y_test_cat[:, cls], y_pred_prob[:, cls])
    auc_score   = roc_auc_score(y_test_cat[:, cls], y_pred_prob[:, cls])
    plt.plot(fpr, tpr, lw=1, label=f"{names_pres[idx]} (AUC={auc_score:.2f})")
plt.plot([0,1],[0,1],'k--',lw=1)
plt.title("Multiclass ROC Curves (OvR)")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
plt.tight_layout(); plt.show()

# ─── 5) Calibration (Reliability Diagram) ──────────────────────────────────────
y_true_flat = y_test_cat.ravel()
y_prob_flat = y_pred_prob.ravel()
frac_pos, mean_pred = calibration_curve(y_true_flat, y_prob_flat, n_bins=10)

plt.figure(figsize=(6,6))
plt.plot(mean_pred, frac_pos, 's-', label='Aggregated')
plt.plot([0,1],[0,1],'k--', label='Perfect')
plt.title("Reliability Diagram")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.legend(); plt.tight_layout(); plt.show()

# ─── 6) Additional Metrics Bar Chart ──────────────────────────────────────────
add_metrics = {
    'Log Loss': ll,
    'Bal. Accuracy': bal_acc,
    'MCC': mcc,
    "Cohen's Kappa": kappa,
    'Avg Precision': ap_macro,
    'Top-3 Acc': top3_acc
}

plt.figure(figsize=(8,5))
plt.bar(add_metrics.keys(), add_metrics.values())
plt.title("Additional Evaluation Metrics")
plt.ylabel("Score"); plt.xticks(rotation=45, ha='right')
plt.ylim(0,1.05)
plt.tight_layout(); plt.show()
