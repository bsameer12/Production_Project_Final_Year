import matplotlib
matplotlib.use("Agg")

import customtkinter as ctk
from tkinter import StringVar, filedialog
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
import matplotlib.pyplot as plt
import time
import subprocess
import csv
import os
import threading
from PIL import Image, ImageTk, ImageOps
import io
from openpyxl import Workbook
from queue import Queue, Empty

# === CONFIG ===
SEQ_LENGTH = 10
COOLDOWN_PERIOD = 30
predicting = False
tts_enabled = True
frame_buffer = deque(maxlen=SEQ_LENGTH)
confidence_history = deque(maxlen=50)
prediction_history = []
prediction_log_data = []
spoken_timestamps = {}

# === Load Model & Setup ===
model = load_model("asl_landmark_lstm_model.keras")
label_map = ['1','10','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','del','nothing','space']
class_labels = {i: label for i, label in enumerate(label_map)}
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# === TTS using macOS 'say' command ===
tts_queue = Queue()

def tts_worker():
    while True:
        try:
            prediction = tts_queue.get(timeout=1)
            if prediction and tts_enabled:
                subprocess.run(["say", prediction])
        except Empty:
            continue

threading.Thread(target=tts_worker, daemon=True).start()

# === CSV Logging ===
csv_filename = "asl_predictions_log.csv"
if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Prediction", "Confidence", "Confidence_Level"])

# === UI Setup ===
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
app = ctk.CTk()
app.geometry("1152x940")
app.title("ASL Landmark Recognition")
cap = cv2.VideoCapture(0)

# === UI Elements ===
tts_var = StringVar(value="on")
def toggle_tts():
    global tts_enabled
    tts_enabled = tts_var.get() == "on"

tts_switch = ctk.CTkSwitch(app, text="Text-to-Speech", variable=tts_var, command=toggle_tts, onvalue="on", offvalue="off")
tts_switch.place(x=10, y=10)
tts_switch.select()

video_label = ctk.CTkLabel(app, text="", width=640, height=480)
video_label.place(x=10, y=50)

prediction_box = ctk.CTkLabel(app, text="Prediction: -\nConfidence: 0.00\nLevel: -", fg_color="green", text_color="black", width=240, height=80, corner_radius=10, font=("Arial", 16))
prediction_box.place(x=330, y=60)

snapshot_box = ctk.CTkLabel(app, text="", width=200, height=200)
snapshot_box.place(x=680, y=50)

start_btn = ctk.CTkButton(app, text="Start", command=lambda: set_predicting(True))
stop_btn = ctk.CTkButton(app, text="Stop", command=lambda: set_predicting(False))
exit_btn = ctk.CTkButton(app, text="Exit", command=app.quit)
export_btn = ctk.CTkButton(app, text="Export to Excel", command=lambda: export_to_excel())

start_btn.place(x=950, y=70)
stop_btn.place(x=950, y=110)
exit_btn.place(x=950, y=150)
export_btn.place(x=950, y=190)

# Label above the Stats Section
stats_title_label = ctk.CTkLabel(app, text="Prediction Stats", font=("Arial", 14, "bold"))
stats_title_label.place(x=680, y=280)

# Stats Label
stats_label = ctk.CTkLabel(app, text="FPS: 0.0\nConfidence: 0.0\nTop 3:\nPredicted: -", justify="left")
stats_label.place(x=680, y=310)


# Label above the Hand Landmark Textbox
landmark_label = ctk.CTkLabel(app, text="Hand Landmark", font=("Arial", 14, "bold"))
landmark_label.place(x=900, y=280)

# Hand Landmark Textbox
landmark_output = ctk.CTkTextbox(app, width=240, height=200)
landmark_output.place(x=900, y=310)
landmark_output.configure(state="disabled")

# Label above the Prediction History Log (Left Side)
history_label = ctk.CTkLabel(app, text="Prediction History", font=("Arial", 14, "bold"))
history_label.place(x=10, y=540)

# Prediction History Textbox (Left Side)
history_log = ctk.CTkTextbox(app, width=540, height=150)
history_log.place(x=10, y=570)
history_log.insert("0.0", "[Time] Prediction (Confidence)\n")
history_log.configure(state="disabled")

# Label above the Histogram Box (Right Side)
histogram_label = ctk.CTkLabel(app, text="Prediction Confidence Histogram", font=("Arial", 14, "bold"))
histogram_label.place(x=580, y=540)

# Histogram Box (Right Side)
histogram_box = ctk.CTkLabel(app, text="", width=340, height=150)
histogram_box.place(x=580, y=570)




# === Supporting Functions ===
def set_predicting(val):
    global predicting
    predicting = val

def export_to_excel():
    filename = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")])
    if filename:
        wb = Workbook()
        ws = wb.active
        ws.append(["Timestamp", "Prediction", "Confidence", "Confidence_Level"])
        for row in prediction_log_data:
            ws.append(row)
        wb.save(filename)

def plot_histogram(data):
    plt.clf()
    keys = list(data.keys())
    values = list(data.values())
    plt.bar(keys, values, color='gold')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf).resize((300, 150))
    return ImageTk.PhotoImage(img)

def update_ui(prediction, confidence, level, top3, frame, snapshot, landmarks):
    prediction_box.configure(text=f"Prediction: {prediction}\nConfidence: {confidence:.2f}\nLevel: {level}")
    if prediction in ['A', 'B', 'C']:
        prediction_box.configure(fg_color="green")
    elif prediction in ['del', 'nothing', 'space']:
        prediction_box.configure(fg_color="red")
    else:
        prediction_box.configure(fg_color="blue")

    stats = f"FPS: {frame['fps']:.2f}\nConfidence: {confidence:.2f}\nTop 3:\n"
    for i, (lbl, conf) in enumerate(top3):
        stats += f"{i+1}. {lbl}: {conf:.2f}\n"
    stats += f"Predicted: {prediction}"
    stats_label.configure(text=stats)

    img = Image.fromarray(cv2.cvtColor(frame["img"], cv2.COLOR_BGR2RGB)).resize((640, 480))
    video_img = ImageTk.PhotoImage(img)
    video_label.configure(image=video_img)
    video_label.imgtk = video_img

    hist_img = plot_histogram(Counter(prediction_history[-50:]))
    histogram_box.configure(image=hist_img)
    histogram_box.imgtk = hist_img

    gray = cv2.cvtColor(snapshot, cv2.COLOR_RGB2GRAY)
    snap = ImageOps.invert(Image.fromarray(gray).resize((200, 200)))
    snap_img = ImageTk.PhotoImage(snap)
    snapshot_box.configure(image=snap_img)
    snapshot_box.imgtk = snap_img

    landmark_output.configure(state="normal")
    landmark_output.delete("0.0", "end")
    for i, lm in enumerate(landmarks):
        landmark_output.insert("end", f"ID {i:2d}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}\n")

    landmark_output.configure(state="disabled")

    history_log.configure(state="normal")
    history_log.insert("end", f"[{time.strftime('%H:%M:%S')}] {prediction} ({confidence:.2f})\n")
    history_log.see("end")
    history_log.configure(state="disabled")

# === Frame Updater Thread ===
def update_frame():
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()
        prediction, confidence = "-", 0.0
        model_input_landmarks = None
        top3 = []
        snapshot = rgb.copy()

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            wrist = hand_landmarks.landmark[0]
            normed = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in hand_landmarks.landmark]
            landmark_vector = [c for pt in normed for c in pt]
            model_input_landmarks = landmark_vector.copy()
            if predicting:
                frame_buffer.append(landmark_vector)

        if predicting and len(frame_buffer) == SEQ_LENGTH:
            input_seq = np.array(frame_buffer).reshape(1, SEQ_LENGTH, -1)
            preds = model.predict(input_seq, verbose=0)[0]
            confidence = float(np.max(preds))
            prediction_idx = int(np.argmax(preds))
            prediction = class_labels[prediction_idx]
            confidence_history.append(confidence)
            prediction_history.append(prediction)

            top3_indices = preds.argsort()[-3:][::-1]
            top3 = [(class_labels[i], preds[i]) for i in top3_indices]

            level = "High" if confidence > 0.8 else "Medium" if confidence > 0.5 else "Low"
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            prediction_log_data.append([ts, prediction, round(confidence, 4), level])

            if (prediction not in spoken_timestamps or time.time() - spoken_timestamps[prediction] > COOLDOWN_PERIOD) and confidence > 0.7:
                spoken_timestamps[prediction] = time.time()
                if tts_enabled:
                    tts_queue.put(prediction)

            with open(csv_filename, "a", newline='') as file:
                writer = csv.writer(file)
                writer.writerow([ts, prediction, f"{confidence:.4f}", level])

            app.after(0, update_ui, prediction, confidence, level, top3, {"fps": fps, "img": frame}, snapshot, result.multi_hand_landmarks[0].landmark)
        else:
            app.after(0, update_ui, "-", 0.0, "-", [], {"fps": fps, "img": frame}, snapshot, [])

        time.sleep(0.01)

# === Launch Frame Thread & Start App ===
threading.Thread(target=update_frame, daemon=True).start()
app.mainloop()
cap.release()
cv2.destroyAllWindows()
