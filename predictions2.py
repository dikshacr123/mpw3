import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle

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

# Generate synthetic data in batches
def generate_synthetic_data_logistic(image, time_steps, V0=50, V_max=500, r=0.1):
    generated_images = []
    for t in range(time_steps):
        tumor_area = logistic_growth(V0, V_max, r, t)
        tumor_mask = np.zeros((image.shape[0], image.shape[1]))
        radius = int(np.sqrt(tumor_area / np.pi))
        center = (image.shape[1] // 2, image.shape[0] // 2)
        cv2.circle(tumor_mask, center, radius, 1, -1)
        synthetic_image = image.copy()
        synthetic_image[:, :, 0] += tumor_mask
        synthetic_image = np.clip(synthetic_image, 0, 1)
        synthetic_image = synthetic_image.reshape(image.shape[0], image.shape[1], 3)  
        generated_images.append(synthetic_image)
    return np.array(generated_images)

# Build ConvLSTM model
def build_conv_lstm(image_shape=(224, 224, 3), time_steps=10):
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

# Prepare data for ConvLSTM
def prepare_conv_lstm_data(images):
    X = images[:-1]
    y = images[1:]
    return X, y

# Main function with batch processing
def main():
    training_folder = r"C:\\Users\\crdik\\Downloads\\mpw\\dataset and backend\\Training\\glioma"
    testing_folder = r"C:\\Users\\crdik\\Downloads\\mpw\\dataset and backend\\Testing1"
    time_steps = 10
    image_shape = (224, 224, 3)
    batch_size = 20

    filenames = [f for f in os.listdir(training_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for i in range(0, len(filenames), batch_size):
        X_train_list, y_train_list = [], []
        batch_filenames = filenames[i:i+batch_size]

        for filename in batch_filenames:
            image_path = os.path.join(training_folder, filename)
            try:
                image = preprocess_image(image_path, target_size=image_shape[:2])
                synthetic_images = generate_synthetic_data_logistic(image, time_steps)
                X, y = prepare_conv_lstm_data(synthetic_images)
                X_train_list.append(X)
                y_train_list.append(y)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

        X_train = np.vstack(X_train_list)
        y_train = np.vstack(y_train_list)
        num_samples = X_train.shape[0] // time_steps  
        X_train = X_train[:num_samples * time_steps].reshape(num_samples, time_steps, *image_shape)
        y_train = y_train[:num_samples * time_steps].reshape(num_samples, time_steps, *image_shape)

        conv_lstm_model = build_conv_lstm(image_shape=image_shape, time_steps=time_steps)

        early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        conv_lstm_model.fit(X_train, y_train, epochs=20, batch_size=1, callbacks=[early_stopping])

    for filename in os.listdir(testing_folder):
        image_path = os.path.join(testing_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = preprocess_image(image_path, target_size=image_shape[:2])
                synthetic_images = generate_synthetic_data_logistic(image, time_steps)
                synthetic_images = synthetic_images.reshape(1, time_steps, *image_shape)
                predictions = conv_lstm_model.predict(synthetic_images)
                fig, ax = plt.subplots(1, 1, figsize=(15, 15))
                ax.imshow(image)
                ax.axis('off')
                ax.set_title("Actual Image")
                fig, axes = plt.subplots(1, time_steps, figsize=(15, 15))
                for i, ax in enumerate(axes):
                    ax.imshow(predictions[0, :, :, 0], cmap='gray')
                    ax.axis('off')
                    ax.set_xlabel(f'Time Step {i}')
                ax.set_title(f"Predicted MRI at {i} Time Steps ")
                plt.show()
                pickle.dump(conv_lstm_model, open('predictions.pkl', 'wb'))
                
                # Compute accuracy
                y_true = synthetic_images[:, -1, :, :, 0]
                y_pred = predictions[:, :, :, 0]
                accuracy = np.mean(np.isclose(y_true, y_pred, atol=0.1)) * 100
                print(f"Overall model accuracy: {accuracy:.2f}%")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    main()
