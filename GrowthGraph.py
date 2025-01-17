import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, BatchNormalization, LSTM
import matplotlib.pyplot as plt

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or cannot be opened: {image_path}")
    
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    normalized_image = resized_image / 255.0
    rgb_image = cv2.cvtColor((normalized_image * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)

    return rgb_image / 255.0  # Normalize to [0, 1]

# Logistic Growth Function (for tumor progression)
def logistic_growth(V0, V_max, r, t):
    return V_max / (1 + ((V_max - V0) / V0) * np.exp(-r * t))

# Generate synthetic tumor data
def generate_synthetic_data_logistic(image, time_steps, V0=50, V_max=500, r=0.1):
    generated_images = []
    tumor_areas = []

    for t in range(time_steps):
        tumor_area = logistic_growth(V0, V_max, r, t)
        tumor_mask = np.zeros((image.shape[0], image.shape[1]))

        radius = int(np.sqrt(tumor_area / np.pi))
        center = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.circle(tumor_mask, center, radius, 1, -1)

        synthetic_image = image.copy()
        synthetic_image[:, :, 0] += tumor_mask
        synthetic_image = np.clip(synthetic_image, 0, 1)

        generated_images.append(synthetic_image)
        tumor_areas.append(tumor_area)

    return np.array(generated_images), np.array(tumor_areas)

# Prepare LSTM training data
def prepare_lstm_data(tumor_areas):
    X = tumor_areas[:-1].reshape(-1, 1, 1)  # Features (previous time steps)
    y = tumor_areas[1:].reshape(-1, 1)  # Target (next time step)
    return X, y

# Define and train the LSTM model
def train_lstm_model(X, y, epochs=200, batch_size=1):
    lstm_model = models.Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(1, 1)))
    lstm_model.add(layers.Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=epochs, batch_size=batch_size)
    return lstm_model

def main():
    training_folder = r"C:\\Users\\crdik\\Downloads\\mpw\\dataset and backend\\Training1"
    testing_folder = r"C:\\Users\\crdik\\Downloads\\mpw\\dataset and backend\\Testing1"
    time_steps = 10
    image_shape = (224, 224, 3)

    all_tumor_areas = []

    for filename in os.listdir(training_folder):
        image_path = os.path.join(training_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = preprocess_image(image_path, target_size=image_shape[:2])
                _, tumor_areas = generate_synthetic_data_logistic(image, time_steps)
                all_tumor_areas.append(tumor_areas)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Flatten tumor areas and prepare LSTM training data
    all_tumor_areas = np.array(all_tumor_areas).flatten()
    X_train, y_train = prepare_lstm_data(all_tumor_areas)

    # Train LSTM model
    lstm_model = train_lstm_model(X_train, y_train)

    # Test on new data and plot predictions
    for filename in os.listdir(testing_folder):
        image_path = os.path.join(testing_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = preprocess_image(image_path, target_size=image_shape[:2])
                _, tumor_areas = generate_synthetic_data_logistic(image, time_steps)

                # Predict growth
                predicted_growth = lstm_model.predict(tumor_areas[:-1].reshape(-1, 1, 1))

                # Plot growth predictions
                plt.figure(figsize=(8, 6))
                plt.plot(range(time_steps-1), tumor_areas[:-1], label='True Growth', marker='o', color='b')
                plt.plot(range(time_steps-1), predicted_growth, label='Predicted Growth', linestyle='--', marker='x', color='r')
                plt.title(f"Tumor Growth Over Time for {filename}")
                plt.xlabel("Time Step")
                plt.ylabel("Tumor Area (pixels)")
                plt.legend()
                plt.grid(True)
                plt.show()
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
