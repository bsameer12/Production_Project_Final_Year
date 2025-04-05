import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import time
import pyttsx3
import csv
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# CSV logging
csv_filename = "asl_predictions_log.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Prediction", "Confidence", "Confidence_Level"])

# Load the trained landmark-based LSTM model
model = load_model("asl_landmark_lstm_model.keras")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Class labels from training
label_map = ['1','10','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
class_labels = {i: label for i, label in enumerate(label_map)}

# Settings
SEQ_LENGTH = 10
frame_buffer = deque(maxlen=SEQ_LENGTH)
confidence_history = deque(maxlen=50)
prev_time = time.time()
last_spoken = ""

# Start webcam
cap = cv2.VideoCapture(1)
print("📷 ASL Live Landmark Recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not available.")
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = "No Hand"
    confidence = 0.0
    top3_preds = []
    model_input_landmarks = None

    if result.multi_hand_landmarks:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            h, w, _ = frame.shape
            x_min, x_max = int(min(x_vals) * w), int(max(x_vals) * w)
            y_min, y_max = int(min(y_vals) * h), int(max(y_vals) * h)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(frame, f"Hand #{idx+1}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            wrist = hand_landmarks.landmark[0]
            normed = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in hand_landmarks.landmark]
            landmark_vector = [coord for pt in normed for coord in pt]

            model_input_landmarks = landmark_vector.copy()
            frame_buffer.append(landmark_vector)

    # Predict if enough frames are collected
    if len(frame_buffer) == SEQ_LENGTH:
        input_seq = np.array(frame_buffer).reshape(1, SEQ_LENGTH, -1)
        predictions = model.predict(input_seq, verbose=0)[0]
        confidence = float(np.max(predictions))
        predicted_class = int(np.argmax(predictions))
        prediction = class_labels.get(predicted_class, "Unknown")

        confidence_history.append(confidence)

        # Top 3 predictions
        top3_indices = predictions.argsort()[-3:][::-1]
        top3_preds = [(class_labels[i], predictions[i]) for i in top3_indices]

        # Confidence label
        conf_label = "Low"
        if confidence > 0.8:
            conf_label = "High"
        elif confidence > 0.5:
            conf_label = "Medium"

        # Speak only new confident predictions
        if confidence > 0.7 and prediction != last_spoken:
            engine.say(prediction)
            engine.runAndWait()
            last_spoken = prediction

        # Log to CSV
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), prediction, f"{confidence:.4f}", conf_label])

    # Label with black color
    color = (0, 0, 0)

    cv2.putText(frame, f'{prediction} ({confidence:.2f}) [{conf_label}]', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA)

    # Top 3 predictions
    for i, (label, conf) in enumerate(top3_preds):
        cv2.putText(frame, f"{i+1}. {label}: {conf:.2f}", (10, 80 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show input landmarks
    if model_input_landmarks:
        for idx, val in enumerate(model_input_landmarks[:21]):
            cv2.putText(frame, f"L{idx}:{val:.2f}", (10, 200 + idx * 18),
                        cv2.FONT_HERSHEY_PLAIN, 1, color, 1)

    # Confidence history graph
    if len(confidence_history) > 1:
        conf_plot = np.array(confidence_history)
        graph_height, graph_width = 100, 200
        graph = np.ones((graph_height, graph_width, 3), dtype=np.uint8) * 255
        for i in range(1, len(conf_plot)):
            pt1 = (int((i - 1) * graph_width / len(conf_plot)), int(graph_height * (1 - conf_plot[i - 1])))
            pt2 = (int(i * graph_width / len(conf_plot)), int(graph_height * (1 - conf_plot[i])))
            cv2.line(graph, pt1, pt2, (0, 0, 0), 2)
        frame[-graph_height:, -graph_width:] = graph

    # FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("📖 ASL Landmark Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
engine.stop()
