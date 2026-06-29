id="animalall1"
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load YOLO model
model = YOLO("yolov8n.pt")

# Carnivorous animals
carnivores = ['dog', 'cat', 'bear']

# Window
window = tk.Tk()

window.title("Animal Detection System")

window.geometry("900x700")

window.configure(bg="lightblue")

# Title
title = tk.Label(window,
                 text="AI Animal Detection",
                 font=("Arial",20,"bold"),
                 bg="lightblue")

title.pack(pady=20)

# Image preview
image_label = tk.Label(window)

image_label.pack()

# Result label
result_label = tk.Label(window,
                        text="Detection Results",
                        font=("Arial",16,"bold"),
                        bg="lightblue")

result_label.pack(pady=20)

# Upload image function
def upload_image():

    file_path = filedialog.askopenfilename()

    if not file_path:
        return

    frame = cv2.imread(file_path)

    results = model(frame)

    carnivore_count = 0

    for result in results:

        boxes = result.boxes

        for box in boxes:

            cls = int(box.cls[0])

            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Carnivores
            if label in carnivores:

                color = (0,0,255)

                carnivore_count += 1

            else:

                color = (255,0,0)

            # Rectangle
            cv2.rectangle(frame,
                          (x1,y1),
                          (x2,y2),
                          color,
                          3)

            # Label
            cv2.putText(frame,
                        label,
                        (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2)

    # Popup
    messagebox.showinfo(
        "Carnivorous Animals",
        f"Detected: {carnivore_count}"
    )

    # Convert image
    frame_rgb = cv2.cvtColor(frame,
                             cv2.COLOR_BGR2RGB)

    img = Image.fromarray(frame_rgb)

    img = img.resize((700,500))

    photo = ImageTk.PhotoImage(img)

    image_label.config(image=photo)

    image_label.image = photo

    result_label.config(
        text=f"Carnivorous Animals: {carnivore_count}"
    )

# Open camera
def open_camera():

    import realtime

# Buttons
upload_btn = tk.Button(window,
                       text="Upload Image",
                       command=upload_image,
                       font=("Arial",16,"bold"),
                       bg="green",
                       fg="white")

upload_btn.pack(pady=10)

camera_btn = tk.Button(window,
                       text="Open Camera",
                       command=open_camera,
                       font=("Arial",16,"bold"),
                       bg="blue",
                       fg="white")

camera_btn.pack(pady=10)

exit_btn = tk.Button(window,
                     text="Exit",
                     command=window.destroy,
                     font=("Arial",16,"bold"),
                     bg="red",
                     fg="white")

exit_btn.pack(pady=20)

window.mainloop()
