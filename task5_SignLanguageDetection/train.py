id="trainfinal1"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense

# Dataset path
dataset_path = "dataset"

# Image generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

# Train data
train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Validation data
val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# CNN model
model = Sequential()

model.add(Conv2D(32,
                 (3,3),
                 activation='relu',
                 input_shape=(64,64,3)))

model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,
                 (3,3),
                 activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dense(6, activation='softmax'))

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=3
)

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
model.save("model/sign_language_model.keras")

print("Model Saved Successfully")

# Accuracy graph
plt.plot(history.history['accuracy'],
         label='Train')

plt.plot(history.history['val_accuracy'],
         label='Validation')

plt.title("Model Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.savefig("accuracy_graph.png")

plt.show()
