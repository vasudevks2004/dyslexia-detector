import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# Load and preprocess your dataset
# Replace this with actual data loading
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()  # Example dataset

X_train = X_train.reshape(-1, 28, 28, 1) / 255.0  # Normalize
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Define CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Change output layer based on your dataset
])

# Compile Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Save Model
model.save('trained_model_CNN1.h5')

print("Model trained and saved as 'trained_model_CNN1.h5'")
