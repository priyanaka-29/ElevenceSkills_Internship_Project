# FILE NAME: evaluate_model.py
# PURPOSE: Evaluate model - confusion matrix, accuracy,
#          precision, recall, F1 score

import cv2
import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("="*50)
print("  MODEL EVALUATION - CONFUSION MATRIX")
print("="*50)

try:
    import seaborn as sns
    from sklearn.metrics import (
        confusion_matrix,
        classification_report,
        accuracy_score
    )
    print(" Evaluation libraries loaded!")
except:
    print(" Run: pip install seaborn scikit-learn")
    exit()

# Load model
if not os.path.exists("face_model.yml"):
    print(" face_model.yml not found!")
    print("Run train_model.py first!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_model.yml")

with open("label_map.json") as f:
    label_map = json.load(f)

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    'haarcascade_frontalface_default.xml'
)

student_names = list(label_map.values())
print(f"\nStudents: {student_names}")
print("Testing model on dataset...\n")

# Test on all photos
true_labels = []
pred_labels = []

for sid, sname in label_map.items():
    folder = os.path.join("dataset", sname)

    if not os.path.exists(folder):
        print(f" Folder not found: {folder}")
        continue

    photos = [f for f in os.listdir(folder)
              if f.lower().endswith(('.jpg','.jpeg','.png'))]

    print(f"Testing: {sname} ({len(photos)} photos)")
    correct = 0

    for photo in photos:
        img = cv2.imread(os.path.join(folder, photo))
        if img is None:
            continue

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            try:
                pred_id, conf = recognizer.predict(face_roi)
                true_labels.append(int(sid))
                pred_labels.append(pred_id)

                if pred_id == int(sid):
                    correct += 1
            except:
                continue

    total_tested = len([t for t in true_labels
                        if t == int(sid)])
    if total_tested > 0:
        acc = correct / total_tested * 100
        print(f"   {sname}: {correct}/{total_tested} = {acc:.1f}%")

if not true_labels:
    print(" No test results!")
    exit()

# Metrics
overall_acc = accuracy_score(true_labels, pred_labels)

print(f"\n{'='*50}")
print(f" OVERALL ACCURACY: {overall_acc*100:.2f}%")
print(f"{'='*50}")

print("\n DETAILED CLASSIFICATION REPORT:")
print(classification_report(
    true_labels,
    pred_labels,
    target_names=student_names,
    zero_division=0
))

# Create visualizations
os.makedirs("visualizations", exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle(
    f'Model Evaluation - Task 1\n'
    f'Overall Accuracy: {overall_acc*100:.1f}% | '
    f'ElevanceSkills Internship',
    fontsize=13, fontweight='bold'
)

# Chart 1: Confusion Matrix
ax1 = axes[0]
cm  = confusion_matrix(true_labels, pred_labels)

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=student_names,
    yticklabels=student_names,
    ax=ax1,
    linewidths=2,
    linecolor='white',
    annot_kws={"size": 14, "weight": "bold"}
)
ax1.set_title(
    f'Confusion Matrix\nAccuracy: {overall_acc*100:.1f}%',
    fontsize=13, fontweight='bold'
)
ax1.set_ylabel('True Label',     fontsize=11)
ax1.set_xlabel('Predicted Label', fontsize=11)

# Chart 2: Per student accuracy
ax2 = axes[1]
ax2.set_facecolor('#ffffff')

per_acc = []
for sid, sname in label_map.items():
    indices = [i for i, t in enumerate(true_labels)
               if t == int(sid)]
    if indices:
        correct = sum(1 for i in indices
                     if pred_labels[i] == int(sid))
        acc = correct / len(indices) * 100
        per_acc.append((sname, acc, len(indices)))

names   = [x[0] for x in per_acc]
accs    = [x[1] for x in per_acc]
samples = [x[2] for x in per_acc]
colors  = ['#2ecc71' if a >= 70 else '#e74c3c'
           for a in accs]

bars = ax2.bar(names, accs, color=colors,
               edgecolor='black', width=0.4,
               linewidth=1.2)

ax2.set_title('Accuracy per Student',
              fontsize=13, fontweight='bold')
ax2.set_ylabel('Accuracy (%)', fontsize=11)
ax2.set_ylim(0, 115)
ax2.axhline(y=80, color='orange',
            linestyle='--', linewidth=2,
            label='80% threshold')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for bar, acc, n in zip(bars, accs, samples):
    ax2.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1.5,
        f'{acc:.1f}%\n({n} imgs)',
        ha='center', fontsize=11, fontweight='bold'
    )

plt.tight_layout()
out = "visualizations/confusion_matrix.png"
plt.savefig(out, dpi=150, bbox_inches='tight',
            facecolor='#f8f9fa')
print(f"\n Confusion matrix saved: {out}")
plt.show()

print("\n" + "="*50)
print(" EVALUATION COMPLETE!")
print("="*50)
print(f"Overall Accuracy : {overall_acc*100:.2f}%")
print(f"Charts saved in  : visualizations/")
print("="*50)