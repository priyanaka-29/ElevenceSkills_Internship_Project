id="realfinal1"
import cv2
import numpy as np
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

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Rectangle
    cv2.rectangle(frame,
                  (100,100),
                  (300,300),
                  (0,255,0),
                  2)

    # ROI
    roi = frame[100:300, 100:300]

    roi = cv2.resize(roi, (64,64))

    roi = roi / 255.0

    roi = np.expand_dims(roi, axis=0)

    # Prediction
    prediction = model.predict(roi, verbose=0)

    class_index = np.argmax(prediction)

    label = classes[class_index]

    confidence = np.max(prediction) * 100

    text = f'{label} ({confidence:.2f}%)'

    # Display
    cv2.putText(frame,
                text,
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2)

    cv2.imshow("Realtime Sign Detection", frame)

    # Voice
    if confidence > 90:

        engine.say(label)

        engine.runAndWait()

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
