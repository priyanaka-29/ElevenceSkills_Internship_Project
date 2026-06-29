# FILE NAME: collect_photos.py

import sys

# Check cv2 first
try:
    import cv2
    print(" cv2 loaded successfully")
except ImportError:
    print(" cv2 not found!")
    print("Run this command first:")
    print("pip install opencv-python==4.8.0.76")
    sys.exit()

import os

print("="*40)
print("  STUDENT PHOTO COLLECTOR")
print("="*40)

student_name = input("\nEnter student name (example: Priyanka): ").strip()

if not student_name:
    print(" Name cannot be empty!")
    exit()

# Create folder
save_path = f"dataset/{student_name}"
os.makedirs(save_path, exist_ok=True)
print(f"\n Folder created: {save_path}")

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Cannot open camera!")
    print("Make sure your webcam is connected.")
    exit()

count = 0
total_needed = 25

print(f"\n📷 Camera is ON!")
print(f"Press SPACE to capture photo ({total_needed} needed)")
print(f"Press Q to quit\n")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print(" Camera read failed!")
        break
    
    # Instructions on screen
    display = frame.copy()
    
    # Black bar at top
    cv2.rectangle(display, (0, 0), (640, 60), (0, 0, 0), -1)
    
    cv2.putText(display,
        f"Student: {student_name}   Photos: {count}/{total_needed}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(display,
        "SPACE = Take Photo    Q = Quit",
        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    # Progress bar
    progress = int((count / total_needed) * 620)
    cv2.rectangle(display, (0, 60), (progress, 75), (0, 255, 0), -1)
    
    cv2.imshow("Photo Collector", display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord(' '):  # SPACE
        count += 1
        filename = os.path.join(save_path, f"img{count}.jpg")
        cv2.imwrite(filename, frame)  # Save original (not display)
        print(f" Photo {count} saved → {filename}")
        
        if count >= total_needed:
            print(f"\n🎉 All {total_needed} photos collected!")
            break
    
    elif key == ord('q') or key == 27:  # Q or ESC
        print(f"\nStopped. Saved {count} photos.")
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*40}")
print(f" DONE for student: {student_name}")
print(f" Photos saved in: {save_path}/")
print(f" Total photos: {count}")
print(f"{'='*40}")

if count < 10:
    print("\n  WARNING: Less than 10 photos!")
    print("Please collect at least 20 photos for good accuracy.")
else:
    print("\n Good! Now run for next student or run train_model.py")