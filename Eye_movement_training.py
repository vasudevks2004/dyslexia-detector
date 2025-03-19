import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

def show_history_graph(history):
    # Summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
    # Summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
data = []
label = []
cw_directory = os.getcwd()
folder = cw_directory + '/eye dataset'

for filename in os.listdir(folder):
    sub_dir = os.path.join(folder, filename)
    for img_name in os.listdir(sub_dir):
        img_dir = os.path.join(sub_dir, img_name)
        print(int(filename), img_dir)
        
        img = cv2.imread(img_dir)
        img = cv2.resize(img, (128, 128))
        
        if img is not None:
            data.append(img / 255.0)  # Normalize pixel values
            label.append(int(filename))

# Convert to NumPy arrays
data = np.array(data)
label = np.array(label)

# Convert labels to one-hot encoding
num_classes = len(set(label))  # Number of unique classes
label = to_categorical(label, num_classes)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(data, label, test_size=0.20, random_state=42)

def train_CNN(X_train, Y_train, X_test, Y_test, num_classes):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Updated activation

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',  # Updated loss function
                  metrics=['accuracy'])

    history = model.fit(X_train, Y_train, epochs=20, validation_data=(X_test, Y_test))

    show_history_graph(history)
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
    
    print("Testing Accuracy:", test_acc)
    print("Testing Loss:", test_loss)

    model.save('eye_movement_trained.h5')
    return model

# Train CNN Model
model_CNN = train_CNN(X_train, Y_train, X_test, Y_test, num_classes)

# Predict
Y_CNN = model_CNN.predict(X_test)
