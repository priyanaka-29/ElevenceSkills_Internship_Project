
import cv2
import numpy as np
import os

# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Load trained model
model = load_model("model/car_color_model.keras")

# Classes
classes = [
    'matiz black',
    'matiz blue',
    'matiz red',
    'rio black',
    'rio blue',
    'rio red',
    'tiggo black',
    'tiggo blue',
    'tiggo red'
]

# Create window
window = tk.Tk()

window.title("Car Colour Detection System")

window.geometry("700x700")

window.configure(bg="lightblue")

# Title
title = tk.Label(window,
                 text="AI Car Colour Detection",
                 font=("Arial", 20, "bold"),
                 bg="lightblue")

title.pack(pady=20)

# Image preview
image_label = tk.Label(window)
image_label.pack()

# Result label
result_label = tk.Label(window,
                        text="Prediction: ",
                        font=("Arial", 16, "bold"),
                        bg="lightblue")

result_label.pack(pady=20)

# Upload image function
def upload_image():

    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    # Show image
    img = Image.open(file_path)

    img = img.resize((450, 300))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)

    image_label.image = photo

    # Predict image
    image = cv2.imread(file_path)

    image = cv2.resize(image, (64, 64))

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    class_index = np.argmax(prediction)

    predicted_color = classes[class_index]

    confidence = np.max(prediction) * 100

    # Show prediction
    result_label.config(
        text=f"Prediction: {predicted_color} ({confidence:.2f}%)"
    )

# Open detection window
def open_detection():

    os.system("python realtime.py")

# Upload button
upload_btn = tk.Button(window,
                       text="Upload Image",
                       command=upload_image,
                       font=("Arial", 14),
                       bg="green",
                       fg="white")

upload_btn.pack(pady=10)

# Detection button
detect_btn = tk.Button(window,
                       text="Open Detection",
                       command=open_detection,
                       font=("Arial", 14),
                       bg="blue",
                       fg="white")

detect_btn.pack(pady=10)

# Exit button
exit_btn = tk.Button(window,
                     text="Exit",
                     command=window.destroy,
                     font=("Arial", 14),
                     bg="red",
                     fg="white")

exit_btn.pack(pady=10)

# Run GUI
window.mainloop()
