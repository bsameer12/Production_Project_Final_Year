import customtkinter as ctk
from tkinter import StringVar

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Main App
app = ctk.CTk()
app.title("ASL Landmark Recognition")
app.geometry("1150x760")

# Text-to-Speech Toggle
tts_var = StringVar(value="on")
tts_switch = ctk.CTkSwitch(
    master=app, text="Text-to-Speech", variable=tts_var, onvalue="on", offvalue="off"
)
tts_switch.select()
tts_switch.place(x=10, y=10)

# Video Frame placeholder
video_frame = ctk.CTkLabel(app, text="", width=640, height=480, corner_radius=10)
video_frame.place(x=10, y=50)

# ID + Confidence Overlay
id_conf_box = ctk.CTkLabel(
    app,
    text="ID: 0\nHigh Confidence",
    fg_color="green",
    corner_radius=8,
    font=("Arial", 16),
    text_color="black",
    width=160,
    height=60
)
id_conf_box.place(x=440, y=60)

# Model Input Snapshot
snapshot = ctk.CTkLabel(app, text="Model Input Snapshot", width=160, height=200)
snapshot.place(x=680, y=50)

# Start/Stop/Exit Buttons
start_btn = ctk.CTkButton(app, text="Start")
stop_btn = ctk.CTkButton(app, text="Stop")
exit_btn = ctk.CTkButton(app, text="Exit", command=app.quit)

start_btn.place(x=860, y=50)
stop_btn.place(x=860, y=100)
exit_btn.place(x=860, y=150)

# Stats and Predictions
stats_label = ctk.CTkLabel(app, text="FPS: 23.8\nConfidence: 0.96\nTop 3: 0.96\n2: 8: 0.02\n3: T: 0.01", justify="left")
stats_label.place(x=680, y=260)

# Landmark Output List
output_title = ctk.CTkLabel(app, text="Output", font=("Arial", 14, "bold"))
output_title.place(x=860, y=210)

landmark_output = ctk.CTkTextbox(app, width=250, height=200)
landmark_output.place(x=860, y=240)
landmark_output.insert("0.0", "".join([f"Landmark {i}: (0.00, 0.00, 0.00)\n" for i in range(10)]))
landmark_output.configure(state="disabled")

# Histogram
histogram_label = ctk.CTkLabel(app, text="Histogram", font=("Arial", 14, "bold"))
histogram_label.place(x=860, y=450)
histogram_box = ctk.CTkLabel(app, text="", width=250, height=150, corner_radius=10)
histogram_box.place(x=860, y=480)

# Output log
output_log = ctk.CTkTextbox(app, width=640, height=80)
output_log.place(x=10, y=550)
output_log.insert("0.0", "[20:56:35] 8 (High)")
output_log.configure(state="disabled")

# History Label
history_label = ctk.CTkLabel(app, text="History", font=("Arial", 14, "bold"))
history_label.place(x=680, y=580)

app.mainloop()
