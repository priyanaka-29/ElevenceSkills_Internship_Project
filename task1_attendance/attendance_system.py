
import sys
print("Loading libraries...")

try:
    import cv2
    print(" cv2 loaded")
except:
    print(" Run: pip install opencv-python==4.8.0.76")
    sys.exit()

try:
    import pandas as pd
    print(" pandas loaded")
except:
    print(" Run: pip install pandas")
    sys.exit()

try:
    import numpy as np
    print(" numpy loaded")
except:
    print(" Run: pip install numpy")
    sys.exit()

try:
    import openpyxl
    print(" openpyxl loaded")
except:
    print(" Run: pip install openpyxl")
    sys.exit()

# Try emotion detection libraries
EMOTION_METHOD = None

try:
    from deepface import DeepFace
    EMOTION_METHOD = "deepface"
    print(" deepface loaded - using for emotion")
except:
    print(" deepface not available")

if EMOTION_METHOD is None:
    try:
        from fer import FER
        emotion_detector = FER(mtcnn=False)
        EMOTION_METHOD = "fer"
        print(" fer loaded - using for emotion")
    except:
        print(" fer not available")

if EMOTION_METHOD is None:
    print(" No emotion library - will use basic detection")
    print(" Run: pip install deepface")

from datetime import datetime
import json
import os

print("\n ALL MAIN LIBRARIES LOADED!")
print("="*50)
print("   ATTENDANCE SYSTEM WITH EMOTION DETECTION")
print("="*50)


TEST_MODE    = True   # True = works anytime
START_HOUR   = 9
START_MINUTE = 30
END_HOUR     = 10
END_MINUTE   = 0


# Check model files
if not os.path.exists("face_model.yml"):
    print("\n face_model.yml NOT FOUND!")
    print("Run: python train_model.py first!")
    sys.exit()

if not os.path.exists("label_map.json"):
    print("\n label_map.json NOT FOUND!")
    print("Run: python train_model.py first!")
    sys.exit()

print(" Model files found!")

# Load models
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("label_map.json") as f:
    label_map = json.load(f)

print(f" Students: {list(label_map.values())}")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Emotion detector setup
if EMOTION_METHOD == "fer":
    emotion_detector = FER(mtcnn=False)

# Emotion list for basic detection
EMOTIONS = ["Neutral", "Happy", "Sad", "Angry", "Surprised"]

def detect_emotion(face_img):
    """Detect emotion using available method"""

    # Method 1: DeepFace
    if EMOTION_METHOD == "deepface":
        try:
            result = DeepFace.analyze(
                face_img,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            if isinstance(result, list):
                return result[0]['dominant_emotion']
            return result['dominant_emotion']
        except:
            return "Neutral"

    # Method 2: FER
    elif EMOTION_METHOD == "fer":
        try:
            result = emotion_detector.detect_emotions(face_img)
            if result:
                scores = result[0]['emotions']
                return max(scores, key=scores.get)
            return "Neutral"
        except:
            return "Neutral"

    # Method 3: Basic (no library needed)
    else:
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            avg_brightness = np.mean(gray)
            if avg_brightness > 150:
                return "Happy"
            elif avg_brightness > 100:
                return "Neutral"
            else:
                return "Sad"
        except:
            return "Neutral"

# Setup attendance
all_students = list(label_map.values())
attendance = {}
for name in all_students:
    attendance[name] = {
        "Name"    : name,
        "Status"  : "Absent",
        "Emotion" : "N/A",
        "Time"    : "N/A"
    }
marked = set()

print(f" Tracking: {all_students}")
print(f" Emotion method: {EMOTION_METHOD or 'Basic'}")

def is_time_ok():
    if TEST_MODE:
        return True
    now   = datetime.now()
    start = now.replace(hour=START_HOUR,   minute=START_MINUTE, second=0)
    end   = now.replace(hour=END_HOUR,     minute=END_MINUTE,   second=0)
    return start <= now <= end

def save_attendance():
    df   = pd.DataFrame(list(attendance.values()))
    date = datetime.now().strftime("%Y-%m-%d")

    csv_file   = f"attendance_{date}.csv"
    excel_file = f"attendance_{date}.xlsx"

    df.to_csv(csv_file, index=False)
    df.to_excel(excel_file, index=False)

    print(f"\n CSV saved:   {csv_file}")
    print(f" Excel saved: {excel_file}")
    print("\n" + "="*50)
    print(" FINAL ATTENDANCE:")
    print("="*50)
    print(df.to_string(index=False))
    print("="*50)

# Open camera
print("\n Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Camera not found!")
    sys.exit()

print(" Camera opened!")
print(" TEST MODE ON - works at any time")
print("\n Look at camera to mark attendance")
print("Press Q to quit and save\n")

# ── MAIN LOOP 
emotion_counter = 0   # Check emotion every 10 frames (faster)

while True:
    if not is_time_ok():
        print(" Time window ended. Saving...")
        break

    ret, frame = cap.read()
    if not ret:
        print(" Camera error!")
        break

    emotion_counter += 1
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces   = face_detector.detectMultiScale(gray, 1.2, 5)
    now_str = datetime.now().strftime("%H:%M:%S")

    # Top black bar
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 55), (0,0,0), -1)
    cv2.putText(frame,
        f"Time:{now_str} | Present:{len(marked)}/{len(all_students)} | Press Q to save",
        (8, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,0), 2)

    for (x, y, w, h) in faces:
        face_gray  = gray[y:y+h, x:x+w]
        face_color = frame[y:y+h, x:x+w]

        try:
            student_id, confidence = recognizer.predict(face_gray)
        except:
            continue

        if confidence < 70:
            name  = label_map.get(str(student_id), "Unknown")
            color = (0, 255, 0)

            # Get emotion (every 10 frames to keep fast)
            emotion = "Detecting..."
            if emotion_counter % 10 == 0:
                emotion = detect_emotion(face_color)

            # Mark attendance
            if name not in marked:
                final_emotion = detect_emotion(face_color)
                attendance[name]["Status"]  = "Present"
                attendance[name]["Emotion"] = final_emotion
                attendance[name]["Time"]    = now_str
                marked.add(name)
                print(f"✅ PRESENT: {name} | {final_emotion} | {now_str}")

            # Show stored emotion
            stored_emotion = attendance[name]["Emotion"]
            label = f"{name} | {stored_emotion}"

        else:
            name  = "Unknown"
            label = "Unknown"
            color = (0, 0, 255)

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-32), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    # Show present list
    y_pos = 70
    for pname in marked:
        cv2.putText(frame, f"✓ {pname}", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        y_pos += 28

    cv2.imshow("Attendance System - Press Q to Save & Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nSaving attendance...")
        break

cap.release()
cv2.destroyAllWindows()
save_attendance()