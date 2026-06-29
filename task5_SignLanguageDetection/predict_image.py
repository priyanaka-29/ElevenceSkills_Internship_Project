import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/sign_language_model.h5")

classes = ['A','B','C','D','E']

image = cv2.imread("test.jpg")

image = cv2.resize(image, (64,64))
image = image / 255.0
image = np.expand_dims(image, axis=0)

prediction = model.predict(image)

label = classes[np.argmax(prediction)]

print("Predicted Sign:", label)