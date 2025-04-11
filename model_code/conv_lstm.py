import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, BatchNormalization
from sklearn.model_selection import train_test_split
import pickle

def calculate_pixel_accuracy(y_true, y_pred, threshold=0.5):
    y_pred_bin = (y_pred > threshold).astype(np.uint8)
    y_true_bin = (y_true > threshold).astype(np.uint8)
    correct = np.sum(y_pred_bin == y_true_bin)
    total = np.prod(y_true.shape)
    return correct / total

# ---------------------------
# ConvLSTM Model Definition
# ---------------------------
def build_conv_lstm(image_shape=(224, 224, 1), time_steps=10):
    model = models.Sequential([
        layers.Input(shape=(time_steps, image_shape[0], image_shape[1], image_shape[2])),
        ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True),
        BatchNormalization(),
        ConvLSTM2D(filters=16, kernel_size=(3, 3), padding='same', return_sequences=False),
        BatchNormalization(),
        Conv2D(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ---------------------------------
# Data Loading from Folder
# ---------------------------------
def load_patient_sequence(patient_path, image_size=(224, 224), time_steps=10):
    frames = sorted(
        [f for f in os.listdir(patient_path) if f.startswith("frame_") and f.endswith(".png")],
        key=lambda x: int(x.split("_")[1].split(".")[0])  # handles frame_0.png, frame_01.png, etc.
    )

    if len(frames) < time_steps + 1:
        return None, None

    X = []
    for i in range(time_steps):
        img = cv2.imread(os.path.join(patient_path, frames[i]))
        img = cv2.resize(img, image_size)
        X.append(img)

    y = cv2.imread(os.path.join(patient_path, frames[time_steps]))
    y = cv2.resize(y, image_size)
    y = cv2.cvtColor(y, cv2.COLOR_BGR2GRAY)  # output is grayscale
    y = y[..., np.newaxis]  # add channel dimension

    return np.array(X), y

# ---------------------------------
# Load all sequences
# ---------------------------------
def load_dataset(data_dir, image_size=(224, 224), time_steps=10):
    X_data = []
    y_data = []

    for patient_folder in os.listdir(data_dir):
        patient_path = os.path.join(data_dir, patient_folder)
        if os.path.isdir(patient_path):
            X, y = load_patient_sequence(patient_path, image_size, time_steps)
            if X is not None and y is not None:
                X_data.append(X)
                y_data.append(y)

    return np.array(X_data), np.array(y_data)

# -------------------------------
# Main training logic
# -------------------------------
if __name__ == "__main__":
    data_path = "synthetic images/training"
    time_steps = 10
    image_shape = (224, 224, 3)

    print("Loading data...")
    X_data, y_data = load_dataset(data_path, image_size=image_shape[:2], time_steps=time_steps)
    print(f"Loaded {len(X_data)} samples.")

    if len(X_data) == 0:
        raise ValueError("No valid training data found. Please check folder structure and frame naming.")

    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    print("Building model...")
    model = build_conv_lstm(image_shape=image_shape, time_steps=time_steps)

    print("Training model...")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=2)

    # Evaluate model on validation data
    print("Evaluating accuracy on validation set...")
    y_pred = model.predict(X_val)

    accuracy = calculate_pixel_accuracy(y_val, y_pred)
    print(f"Pixel-level Accuracy on validation set: {accuracy * 100:.2f}%")

    # --------------------------
    # Save the model as pickle
    # --------------------------
    print("Saving model...")
    with open("conv_lstm_tumor_growth_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("Model training and saving complete!")
