import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import cv2

DATASET_DIR = "archive"
IMG_SIZE = (24, 24)
BATCH_SIZE = 32
EPOCHS = 10

def load_images():
    images, labels = [], []

    for img_name in os.listdir(os.path.join(DATASET_DIR, "closed_eye")):
        img_path = os.path.join(DATASET_DIR, "closed_eye", img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(0)

    for img_name in os.listdir(os.path.join(DATASET_DIR, "open_eye")):
        img_path = os.path.join(DATASET_DIR, "open_eye", img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(1)

    images = np.array(images, dtype="float32") / 255.0
    images = images.reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
    labels = np.array(labels)
    return images, labels

print("Loading images...")
X, y = load_images()
print(f"Loaded {len(X)} images")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(24, 24, 1)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

print("Training...")
model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, y_test))

loss, acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {acc:.4f}")

model.save("eye_state_model.keras")
print("Model saved as eye_state_model.keras")
