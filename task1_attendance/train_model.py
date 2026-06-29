# FILE NAME: train_model.py
# PURPOSE: Train LBPH face recognition model
# METHODOLOGY: Local Binary Patterns Histogram
# COMPARISON: LBPH vs Eigenfaces vs Fisherfaces

import cv2
import os
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*50)
print(" TRAINING FACE RECOGNITION MODEL")
print(" Method: LBPH Algorithm")
print("="*50)

# Face detector 
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

# THREE models for comparison 
model_lbph  = cv2.face.LBPHFaceRecognizer_create()
model_eigen = cv2.face.EigenFaceRecognizer_create()
model_fisher = cv2.face.FisherFaceRecognizer_create()

def get_training_data(dataset_folder):
    face_samples       = []
    face_samples_sized = []   # fixed size for Eigen/Fisher
    ids                = []
    label_map          = {}
    student_id         = 0

    students = [d for d in os.listdir(dataset_folder)
                if os.path.isdir(os.path.join(dataset_folder, d))]

    print(f"\nStudents found: {students}")

    for student_name in students:
        folder = os.path.join(dataset_folder, student_name)
        label_map[student_name] = student_id
        print(f"\nProcessing: {student_name} (ID:{student_id})")

        count = 0
        for photo in os.listdir(folder):
            if not photo.lower().endswith(
                    ('.jpg','.jpeg','.png')):
                continue

            img = cv2.imread(os.path.join(folder, photo))
            if img is None:
                continue

            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_sized = cv2.resize(face_roi, (100, 100))

                face_samples.append(face_roi)
                face_samples_sized.append(face_sized)
                ids.append(student_id)
                count += 1

        print(f"  → {count} faces processed")
        student_id += 1

    return face_samples, face_samples_sized, ids, label_map

# Get training data
faces, faces_sized, ids, label_map = \
    get_training_data("dataset/")

if not faces:
    print("❌ No face data found!")
    print("Run collect_my_photos.py first!")
    exit()

print(f"\nTotal face samples: {len(faces)}")
print("Training models... please wait...")

# Train all 3 models 
ids_array       = np.array(ids)
sized_array     = faces_sized

print("\n[1/3] Training LBPH...")
model_lbph.train(faces, ids_array)
print(" LBPH trained!")

print("[2/3] Training Eigenfaces...")
try:
    model_eigen.train(sized_array, ids_array)
    print(" Eigenfaces trained!")
    eigen_ok = True
except Exception as e:
    print(f" Eigenfaces: {e}")
    eigen_ok = False

print("[3/3] Training Fisherfaces...")
try:
    model_fisher.train(sized_array, ids_array)
    print(" Fisherfaces trained!")
    fisher_ok = True
except Exception as e:
    print(f" Fisherfaces: {e}")
    fisher_ok = False

# Quick accuracy test 
print("\nTesting accuracy on training data...")

lbph_correct   = 0
eigen_correct  = 0
fisher_correct = 0
total          = 0

for face, face_s, true_id in zip(
        faces, faces_sized, ids):
    total += 1

    pred_lbph, _ = model_lbph.predict(face)
    if pred_lbph == true_id:
        lbph_correct += 1

    if eigen_ok:
        pred_eigen, _ = model_eigen.predict(face_s)
        if pred_eigen == true_id:
            eigen_correct += 1

    if fisher_ok:
        pred_fisher, _ = model_fisher.predict(face_s)
        if pred_fisher == true_id:
            fisher_correct += 1

lbph_acc   = lbph_correct   / total * 100
eigen_acc  = eigen_correct  / total * 100 if eigen_ok  else 0
fisher_acc = fisher_correct / total * 100 if fisher_ok else 0

print(f"\n📊 ACCURACY COMPARISON:")
print(f"  LBPH       : {lbph_acc:.1f}%  ← Selected")
print(f"  Eigenfaces : {eigen_acc:.1f}%")
print(f"  Fisherfaces: {fisher_acc:.1f}%")

# ── Save best model ──────────────────────────────────────────
model_lbph.save("face_model.yml")

id_to_name = {str(v): k for k, v in label_map.items()}
with open("label_map.json", "w") as f:
    json.dump(id_to_name, f)

# ── Save comparison chart ────────────────────────────────────
os.makedirs("visualizations", exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor('#f8f9fa')

models  = ['LBPH\n(Selected)', 'Eigenfaces', 'Fisherfaces']
accs    = [lbph_acc, eigen_acc, fisher_acc]
colors  = ['#2ecc71', '#e74c3c', '#3498db']

bars = ax.bar(models, accs, color=colors,
              width=0.4, edgecolor='black', linewidth=1.2)

ax.set_title('Model Comparison - Training Accuracy\nTask 1: Attendance System',
             fontsize=13, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_ylim(0, 115)
ax.axhline(y=80, color='orange', linestyle='--',
           linewidth=2, label='80% threshold')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_facecolor('#ffffff')

for bar, val in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1.5,
        f'{val:.1f}%',
        ha='center', fontsize=13, fontweight='bold'
    )

plt.tight_layout()
plt.savefig("visualizations/model_comparison.png",
            dpi=150, bbox_inches='tight')
print("\n Model comparison chart saved!")

print("\n" + "="*50)
print(" TRAINING COMPLETE!")
print("="*50)
print(f"Students: {list(label_map.keys())}")
print(f"Best Model: LBPH ({lbph_acc:.1f}% accuracy)")
print("Files saved: face_model.yml, label_map.json")
print("Chart saved: visualizations/model_comparison.png")
print("\nNext: python attendance_system.py")
print("="*50)