id="graph1"
import matplotlib.pyplot as plt

# Models
models = ['CNN\n(Selected)',
          'ResNet50',
          'MobileNetV2']

# Accuracy values
accuracy = [96.21, 92.43, 88.35]

# Colors
colors = ['limegreen',
          'red',
          'dodgerblue']

# Figure size
plt.figure(figsize=(10,6))

# Bar graph
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
    'Model Comparison - Training Accuracy\nTask 5: Sign Language Detection System',
    fontsize=18,
    fontweight='bold'
)

# Y label
plt.ylabel('Accuracy (%)',
           fontsize=16)

# Y limit
plt.ylim(0,110)

# Grid
plt.grid(axis='y',
         linestyle='--',
         alpha=0.5)

# Accuracy text
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

print("Graph generated successfully")
