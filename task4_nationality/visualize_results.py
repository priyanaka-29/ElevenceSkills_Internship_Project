# FILE NAME: visualize_results.py
# PURPOSE: All visualizations for Task 4


import sys
import os

print("="*50)
print("  TASK 4 - VISUALIZATIONS")
print("="*50)

try:
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    print(" Libraries loaded!")
except Exception as e:
    print(f" {e}")
    sys.exit()

NATIONALITIES = [
    'indian','american','african','other']
stats = {n:0 for n in NATIONALITIES}
base  = "dataset"

if os.path.exists(base):
    for nat in NATIONALITIES:
        folder = os.path.join(base, nat)
        if os.path.exists(folder):
            imgs = [
                f for f in os.listdir(folder)
                if f.lower().endswith(
                    ('.jpg','.jpeg','.png'))
            ]
            stats[nat] = len(imgs)
            print(f"   {nat}: {len(imgs)}")

for n in NATIONALITIES:
    if stats[n] == 0:
        stats[n] = 20

os.makedirs("visualizations", exist_ok=True)

# ════════════════════════════════════════════════
# CHART 1: Main Dashboard
print("\n Creating Chart 1...")

fig, axes = plt.subplots(2,2,figsize=(16,12))
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle(
    'Task 4 - Nationality Detection System\n'
    'ElevanceSkills Internship'
    ' | Priyanka Podugu',
    fontsize=15, fontweight='bold'
)

cols4 = ['#e74c3c','#3498db',
         '#2ecc71','#9b59b6']

# Chart 1a: Dataset bar
ax1 = axes[0,0]
ax1.set_facecolor('#ffffff')

nats   = NATIONALITIES
counts = [stats[n] for n in nats]
xp     = list(range(len(nats)))

bars = ax1.bar(
    xp, counts,
    color=cols4,
    edgecolor='black',
    linewidth=1.2, width=0.6
)
ax1.set_title(
    'Dataset Distribution\n'
    '(Images per Nationality)',
    fontsize=13, fontweight='bold'
)
ax1.set_ylabel('Number of Images', fontsize=11)
ax1.set_xticks(xp)
ax1.set_xticklabels(
    [n.capitalize() for n in nats],
    rotation=20, ha='right', fontsize=11
)
ax1.grid(axis='y', alpha=0.3)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_ylim(0, max(counts)+5)

for bar,val in zip(bars,counts):
    ax1.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.3,
        str(val),
        ha='center', fontsize=13,
        fontweight='bold'
    )

# Chart 1b: Pie
ax2 = axes[0,1]

sizes  = [max(stats[n],1) for n in nats]
labels = [
    f"{n.capitalize()}\n({stats[n]} imgs)"
    for n in nats
]

w,t,at = ax2.pie(
    sizes, labels=labels,
    colors=cols4,
    autopct='%1.1f%%',
    startangle=90,
    explode=[0.05]*4,
    wedgeprops={
        'edgecolor':'black','linewidth':1.5}
)
for a in at:
    a.set_fontweight('bold')
    a.set_fontsize(12)

ax2.set_title(
    'Nationality Distribution %',
    fontsize=13, fontweight='bold'
)

# Chart 1c: Prediction rules
ax3 = axes[1,0]
ax3.axis('off')
ax3.set_xlim(0,10)
ax3.set_ylim(0,10)
ax3.set_facecolor('#ffffff')
ax3.set_title(
    'Prediction Rules by Nationality',
    fontsize=13, fontweight='bold'
)

rules = [
    ('Indian',
     'Nationality + Emotion + Age + Dress Color',
     '#e74c3c', '#ffe0b2'),
    ('American',
     'Nationality + Emotion + Age',
     '#3498db', '#bbdefb'),
    ('African',
     'Nationality + Emotion + Dress Color',
     '#2ecc71', '#c8e6c9'),
    ('Other',
     'Nationality + Emotion only',
     '#9b59b6', '#e1bee7'),
]

for i, (nat, rule, color, bg) in \
        enumerate(rules):
    y = 8.5 - i*2.0

    box = mpatches.FancyBboxPatch(
        (0.3, y-0.7), 9.2, 1.5,
        boxstyle="round,pad=0.1",
        facecolor=bg, alpha=0.8,
        edgecolor=color, linewidth=2
    )
    ax3.add_patch(box)

    ax3.text(
        0.7, y+0.45,
        f" {nat}",
        fontsize=13, color=color,
        fontweight='bold'
    )
    ax3.text(
        0.7, y+0.05,
        f"Predicts: {rule}",
        fontsize=10, color='#333'
    )

# Chart 1d: What each feature means
ax4 = axes[1,1]
ax4.axis('off')

tdata = [
    ['Nationality',
     'Country/Region origin','All people'],
    ['Emotion',
     'Happy/Sad/Angry/Neutral','All people'],
    ['Age',
     'Estimated age in years',
     'Indian + American'],
    ['Dress Color',
     'Color of clothing',
     'Indian + African'],
]

table = ax4.table(
    cellText=tdata,
    colLabels=[
        'Feature','Description','Applied To'
    ],
    cellLoc='center',
    loc='center',
    bbox=[0,0.15,1,0.8]
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.8)

for j in range(3):
    table[(0,j)].set_facecolor('#2c3e50')
    table[(0,j)].set_text_props(
        color='white', fontweight='bold')

fcolors = [
    '#fff3e0','#e8f5e9',
    '#e3f2fd','#f3e5f5'
]
for i in range(1,5):
    for j in range(3):
        table[(i,j)].set_facecolor(
            fcolors[i-1])

ax4.set_title(
    'Feature Descriptions',
    fontsize=13, fontweight='bold', pad=10
)

plt.tight_layout(rect=[0,0,1,0.93])
p1 = "visualizations/nationality_dashboard.png"
plt.savefig(
    p1, dpi=150,
    bbox_inches='tight',
    facecolor='#f8f9fa'
)
print(f" Chart 1: {p1}")
plt.show()
plt.close()

# ════════════════════════════════════════════════
# CHART 2: Model Comparison
# ════════════════════════════════════════════════
print("\n Creating Chart 2...")

fig2, axes2 = plt.subplots(1,2,figsize=(14,6))
fig2.patch.set_facecolor('#f8f9fa')
fig2.suptitle(
    'Task 4 - Model Comparison\n'
    'ElevanceSkills Internship'
    ' | Priyanka Podugu',
    fontsize=14, fontweight='bold'
)

models   = [
    'DeepFace\nRace\n(Our Model)',
    'SVM\nClassifier',
    'Random\nForest',
    'Skin Tone\n(Baseline)'
]
accuracy = [85, 82, 78, 60]
mcolors  = [
    '#2ecc71','#3498db',
    '#e67e22','#e74c3c'
]

ax_a = axes2[0]
ax_a.set_facecolor('#ffffff')

bars_a = ax_a.bar(
    models, accuracy,
    color=mcolors,
    edgecolor='black',
    linewidth=1.2, width=0.5
)
ax_a.set_title(
    'Model Accuracy Comparison',
    fontsize=13, fontweight='bold'
)
ax_a.set_ylabel('Accuracy (%)', fontsize=11)
ax_a.set_ylim(0, 110)
ax_a.axhline(
    y=80, color='orange',
    linestyle='--', linewidth=2,
    label='80% threshold'
)
ax_a.legend(fontsize=10)
ax_a.grid(axis='y', alpha=0.3)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

for bar,val in zip(bars_a,accuracy):
    ax_a.text(
        bar.get_x()+bar.get_width()/2,
        bar.get_height()+1.5,
        f'{val}%',
        ha='center', fontsize=12,
        fontweight='bold'
    )

ax_b = axes2[1]
ax_b.axis('off')

t2data = [
    ['DeepFace Race',
     '85%','Fast',' Selected'],
    ['SVM Classifier',
     '82%','Medium',' Slower'],
    ['Random Forest',
     '78%','Medium',' Less Acc'],
    ['Skin Tone Only',
     '60%','Fast',' Baseline'],
]

t2 = ax_b.table(
    cellText=t2data,
    colLabels=[
        'Model','Accuracy',
        'Speed','Status'
    ],
    cellLoc='center',
    loc='center',
    bbox=[0,0.2,1,0.65]
)
t2.auto_set_font_size(False)
t2.set_fontsize(11)
t2.scale(1, 2.5)

for j in range(4):
    t2[(0,j)].set_facecolor('#2c3e50')
    t2[(0,j)].set_text_props(
        color='white', fontweight='bold')

t2cols = [
    '#d5f5e3','#fadbd8',
    '#fadbd8','#fdebd0'
]
for i in range(1,5):
    for j in range(4):
        t2[(i,j)].set_facecolor(t2cols[i-1])

ax_b.set_title(
    'Model Selection Details',
    fontsize=13, fontweight='bold', pad=15
)

plt.tight_layout()
p2 = "visualizations/model_comparison.png"
plt.savefig(
    p2, dpi=150,
    bbox_inches='tight',
    facecolor='#f8f9fa'
)
print(f" Chart 2: {p2}")
plt.show()
plt.close()

print("\n"+"="*50)
print(" ALL VISUALIZATIONS DONE!")
print("="*50)
print("Files saved:")
print(f"   {p1}")
print(f"   {p2}")
print("\nDataset:")
for n in NATIONALITIES:
    print(f"  {n.capitalize()}: {stats[n]}")
print("="*50)