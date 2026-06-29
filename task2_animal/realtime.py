id="animalall2"
import cv2
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Carnivores
carnivores = ['dog', 'cat', 'bear']

# Camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Prediction
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

    # Count text
    cv2.putText(frame,
                f'Carnivores: {carnivore_count}',
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2)

    cv2.imshow("Realtime Animal Detection", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):

        break

cap.release()

cv2.destroyAllWindows()
