id="fullrt1"
import cv2
import numpy as np
from ultralytics import YOLO
import os

# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model
from tkinter import filedialog

# Load YOLO model
yolo_model = YOLO("yolov8n.pt")

# Load CNN model
color_model = load_model("model/car_color_model.keras")

# Upload image
file_path = filedialog.askopenfilename()

# Read image
frame = cv2.imread(file_path)

# YOLO detection
results = yolo_model(frame, verbose=False)

car_count = 0
people_count = 0

# Process detections
for result in results:

    boxes = result.boxes

    for box in boxes:

        cls = int(box.cls[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # PERSON DETECTION
        if cls == 0:

            people_count += 1

            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          (0, 255, 255),
                          2)

        # CAR DETECTION
        elif cls == 2:

            car_count += 1

            roi = frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue

            # Resize ROI
            roi_resized = cv2.resize(roi, (64, 64))

            # Average color detection
            avg_color = cv2.mean(roi_resized)

            blue_value = avg_color[0]
            green_value = avg_color[1]
            red_value = avg_color[2]

            # Detect dominant color
            if red_value > blue_value and red_value > green_value:

                predicted_color = "red"

            elif blue_value > red_value and blue_value > green_value:

                predicted_color = "blue"

            elif red_value < 80 and green_value < 80 and blue_value < 80:

                predicted_color = "black"

            else:

                predicted_color = "silver"

            # Rectangle color condition
            # Red rectangle for blue cars
            if predicted_color == "blue":

                color = (0, 0, 255)

            # Blue rectangle for others
            else:

                color = (255, 0, 0)

            # Draw rectangle
            cv2.rectangle(frame,
                          (x1, y1),
                          (x2, y2),
                          color,
                          3)

# Show counts
cv2.putText(frame,
            f'Cars: {car_count}',
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)

cv2.putText(frame,
            f'People: {people_count}',
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2)

cv2.putText(frame,
            f'Color: {predicted_color}',
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2)

# Show output
cv2.imshow("Car Colour Detection", frame)

cv2.waitKey(0)

cv2.destroyAllWindows()
