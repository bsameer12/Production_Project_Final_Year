import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Toplevel
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('asl_sign_language_model.h5')

# Class labels dictionary
class_labels = {
    0: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 1: '10',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 36: 'DELETE', 14: 'E', 15: 'F', 16: 'G', 17: 'H',
    18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 37: 'Nothing', 24: 'O', 25: 'P',
    26: 'Q', 27: 'R', 28: 'S', 38: 'SPACE', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X',
    34: 'Y', 35: 'Z'
}


# Function to browse and select image, then make prediction
def browse_and_predict():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename(title="Select an Image",
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])

    if not file_path:
        return

    try:
        # Open a Toplevel window to display the image preview
        preview_window = Toplevel(root)
        preview_window.title("Image Preview")

        # Load and display the image
        img = Image.open(file_path)
        img.thumbnail((200, 200))  # Resize the image for the preview window
        img_display = ImageTk.PhotoImage(img)

        # Create a Label to display the image
        label = tk.Label(preview_window, image=img_display)
        label.image = img_display  # Keep a reference to the image
        label.pack(padx=10, pady=10)

        # Set image dimensions (modify if your model requires different dimensions)
        img_width, img_height = 64, 64

        # Load and preprocess the image for prediction
        img = image.load_img(file_path, target_size=(img_width, img_height))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Rescale the image

        # Predict with the model
        predictions = loaded_model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=-1)

        # Map the predicted class index to the corresponding label
        predicted_label = class_labels[predicted_class[0]]

        # Show the result in a messagebox
        messagebox.showinfo("Prediction Result", f"Predicted Sign: {predicted_label}")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Create a Tkinter window
root = tk.Tk()
root.title("ASL Sign Language Prediction")
root.geometry("400x200")

# Add a button to browse and predict
browse_button = tk.Button(root, text="Browse Image", command=browse_and_predict, width=20)
browse_button.pack(pady=40)

# Start the Tkinter event loop
root.mainloop()
