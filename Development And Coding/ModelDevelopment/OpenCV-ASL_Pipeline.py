import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from collections import deque

# Load ASL model
model = tf.keras.models.load_model('asl_sign_language_model.h5')
img_width, img_height = 64, 64

# Class labels
class_labels = {
    0: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 1: '10',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 36: 'DELETE', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
    18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 37: 'Nothing', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 38: 'SPACE', 29: 'T', 30: 'U', 31: 'V',
    32: 'W', 33: 'X', 34: 'Y', 35: 'Z'
}

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Smoothing buffer
prediction_buffer = deque(maxlen=10)

# Start webcam
cap = cv2.VideoCapture(1)
print("📷 Improved ASL Recognition started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera frame not available.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction = "No Hand"
    confidence = 0.0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            h, w, _ = frame.shape
            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

            x_min, x_max = max(min(x_list) - 30, 0), min(max(x_list) + 30, w)
            y_min, y_max = max(min(y_list) - 30, 0), min(max(y_list) + 30, h)

            # Extract and preprocess hand ROI
            hand_roi = frame[y_min:y_max, x_min:x_max]
            if hand_roi.size == 0:
                continue

            # Normalize & resize
            resized = cv2.resize(hand_roi, (img_width, img_height))
            input_array = image.img_to_array(resized)
            input_array = np.expand_dims(input_array, axis=0) / 255.0

            # Show what the model sees
            cv2.imshow("🧠 Input to Model", resized)

            # Predict
            predictions = model.predict(input_array)
            confidence = float(np.max(predictions))
            predicted_class = np.argmax(predictions)

            # Debug info
            raw_label = class_labels.get(predicted_class, "Unknown")
            print(f"[DEBUG] Prediction: {raw_label}, Confidence: {confidence:.4f}")

            # Optional: log to file
            # with open("confidence_log.txt", "a") as f:
            #     f.write(f"{raw_label},{confidence:.4f}\n")

            # Confidence threshold lowered to 0.3 for testing
            if confidence > 0.3:
                prediction_buffer.append(predicted_class)
                final_class = max(set(prediction_buffer), key=prediction_buffer.count)
                prediction = class_labels.get(final_class, "Unknown")
            else:
                prediction = "Low Confidence"

            # Draw overlay
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            cv2.putText(frame, f'{prediction} ({confidence:.2f})', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display final output
    cv2.imshow("📖 ASL Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
