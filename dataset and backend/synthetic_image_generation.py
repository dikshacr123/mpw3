import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Function to preprocess the image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or cannot be opened: {image_path}")

    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    normalized_image = resized_image / 255.0
    rgb_image = cv2.cvtColor((normalized_image * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)

    return rgb_image

# Define the generator model
def build_generator(latent_dim, image_shape=(224, 224, 3)):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(image_shape[0], image_shape[1], image_shape[2] + 1)))

    model.add(layers.Conv2DTranspose(256, kernel_size=3, strides=1, padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(128, kernel_size=3, strides=1, padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(64, kernel_size=3, strides=1, padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding='same'))
    model.add(layers.ReLU())
    model.add(layers.Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh'))
    return model

# Logistic Growth Function (for tumor progression)
def logistic_growth(V0, V_max, r, t):
    return V_max / (1 + ((V_max - V0) / V0) * np.exp(-r * t))

# Generate synthetic tumor data for an image with explicit tumor growth
def generate_synthetic_data(image, time_steps, generator, latent_dim, growth_scale=0.1):
    generated_images = []
    tumor_areas = []

    for t in range(time_steps):
        # Introduce increasing noise levels to simulate growth
        noise = np.random.normal(0, 1, size=(1, latent_dim)) * (1 + t * growth_scale)
        time_step_condition = np.full((1, image.shape[0], image.shape[1], 1), t)  # Time-step condition
        conditioned_input = np.concatenate([image[np.newaxis, :, :, :], time_step_condition], axis=-1)

        synthetic_image = generator.predict([conditioned_input, noise])
        synthetic_image = (synthetic_image + 1) / 2  # Rescale to [0, 1] range

        generated_images.append(synthetic_image[0])
        tumor_mask = synthetic_image[0, :, :, 0] > 0.5  # Thresholding
        tumor_area = np.sum(tumor_mask)
        tumor_areas.append(tumor_area)

    return np.array(tumor_areas), generated_images

# Prepare LSTM training data from synthetic tumor areas
def prepare_lstm_data(tumor_areas):
    X = tumor_areas[:-1].reshape(-1, 1, 1)  # Features (previous time steps)
    y = tumor_areas[1:].reshape(-1, 1)  # Target (next time step)
    return X, y

# Define and train the LSTM model
def train_lstm_model(X, y, epochs=300, batch_size=1):
    lstm_model = models.Sequential()
    lstm_model.add(layers.LSTM(50, activation='relu', input_shape=(1, 1)))
    lstm_model.add(layers.Dense(1))
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=epochs, batch_size=batch_size)
    return lstm_model

# Load dataset
training_folder = "training_images"
testing_folder = "testing_images"
time_steps = 30

# Predefined latent dimension and generator model
latent_dim = 100
image_shape = (224, 224, 3)
generator = build_generator(latent_dim, image_shape)

# Training data generation
all_tumor_areas = []

for filename in os.listdir(training_folder):
    image_path = os.path.join(training_folder, filename)
    image = preprocess_image(image_path)
    tumor_areas, _ = generate_synthetic_data(image, time_steps, generator, latent_dim, growth_scale=0.2)
    all_tumor_areas.append(tumor_areas)

# Stack all tumor areas for training
all_tumor_areas = np.vstack(all_tumor_areas)

# Prepare LSTM training data
X_train, y_train = prepare_lstm_data(all_tumor_areas.flatten())
lstm_model = train_lstm_model(X_train, y_train)

# Predict tumor growth on test images
for filename in os.listdir(testing_folder):
    image_path = os.path.join(testing_folder, filename)
    image = preprocess_image(image_path)
    tumor_areas, generated_images = generate_synthetic_data(image, time_steps, generator, latent_dim, growth_scale=0.2)

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

    # Visualize generated images
    fig, axes = plt.subplots(1, time_steps, figsize=(15, 15))
    for i, ax in enumerate(axes):
        ax.imshow(generated_images[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f"Time Step {i}")
    plt.show()