id="guifinal1"
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import pyttsx3
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Voice
engine = pyttsx3.init()

# Load model
model = load_model(
    "model/sign_language_model.keras",
    compile=False
)

# Classes
classes = ['A', 'B', 'C', 'D', 'E', 'V']

# Window
window = tk.Tk()

window.title("Sign Language Detection")

window.geometry("700x700")

window.configure(bg="lightblue")

# Title
title = tk.Label(window,
                 text="AI Sign Language Detection",
                 font=("Arial",20,"bold"),
                 bg="lightblue")

title.pack(pady=20)

# Image label
image_label = tk.Label(window)

image_label.pack()

# Result label
result_label = tk.Label(window,
                        text="Prediction",
                        font=("Arial",18,"bold"),
                        bg="lightblue")

result_label.pack(pady=20)

# Upload image
def upload_image():

    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    # Display image
    img = Image.open(file_path)

    img = img.resize((400,300))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)

    image_label.image = photo

    # Prediction
    image = cv2.imread(file_path)

    image = cv2.resize(image, (64,64))

    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    class_index = np.argmax(prediction)

    label = classes[class_index]

    confidence = np.max(prediction) * 100

    result_label.config(
        text=f'Prediction: {label} ({confidence:.2f}%)'
    )

    engine.say(label)

    engine.runAndWait()

# Open realtime camera
def open_camera():

    os.system("python realtime.py")

# Upload button
upload_btn = tk.Button(window,
                       text="Upload Image",
                       command=upload_image,
                       font=("Arial",16,"bold"),
                       bg="green",
                       fg="white")

upload_btn.pack(pady=10)

# Camera button
camera_btn = tk.Button(window,
                       text="Open Camera",
                       command=open_camera,
                       font=("Arial",16,"bold"),
                       bg="blue",
                       fg="white")

camera_btn.pack(pady=10)

# Exit button
exit_btn = tk.Button(window,
                     text="Exit",
                     command=window.destroy,
                     font=("Arial",16,"bold"),
                     bg="red",
                     fg="white")

exit_btn.pack(pady=20)

window.mainloop()
