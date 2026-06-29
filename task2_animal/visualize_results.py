id="an2"
import matplotlib.pyplot as plt

# Models
models = ['YOLOv8\n(Selected)',
          'Faster R-CNN',
          'SSD MobileNet']

# Accuracy values
accuracy = [97.85, 92.40, 88.75]

# Colors
colors = ['limegreen',
          'red',
          'dodgerblue']

# Figure
plt.figure(figsize=(10,6))

# Bars
bars = plt.bar(models,
               accuracy,
               color=colors,
               edgecolor='black',
               width=0.5)

# Threshold line
plt.axhline(y=80,
            color='orange',
            linestyle='--',
            linewidth=2,
            label='80% threshold')

# Title
plt.title(
    'Model Comparison - Training Accuracy\nTask 2: Animal Detection System',
    fontsize=18,
    fontweight='bold'
)

# Y label
plt.ylabel('Accuracy (%)',
           fontsize=16)

# Limit
plt.ylim(0,110)

# Grid
plt.grid(axis='y',
         linestyle='--',
         alpha=0.5)

# Text labels
for bar, value in zip(bars, accuracy):

    plt.text(bar.get_x() + bar.get_width()/2,
             value + 1,
             f'{value:.2f}%',
             ha='center',
             fontsize=16,
             fontweight='bold')

# Legend
plt.legend(fontsize=12)

# Save graph
plt.savefig("model_comparison.png")

# Show graph
plt.show()

print("Visualization generated successfully")
