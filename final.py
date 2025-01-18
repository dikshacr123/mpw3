import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import ConvLSTM2D, Conv2D, BatchNormalization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, classification_report
from skimage.metrics import structural_similarity as ssim


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

        synthetic_image = synthetic_image.reshape(image.shape[0], image.shape[1], 3)  # Keep 3 channels

        generated_images.append(synthetic_image)
        tumor_areas.append(tumor_area)

    return np.array(generated_images), np.array(tumor_areas)


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
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    return model


def prepare_conv_lstm_data(images):
    X = images[:-1]
    y = images[1:]
    return X, y


def main():
    training_folder = r"C:\\Users\\crdik\\Downloads\\mpw\\dataset and backend\\Training1"
    testing_folder = r"C:\\Users\\crdik\\Downloads\\mpw\\dataset and backend\\Testing1"
    time_steps = 10
    image_shape = (224, 224, 3)

    X_train_list, y_train_list = [], []

    for filename in os.listdir(training_folder):
        image_path = os.path.join(training_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = preprocess_image(image_path, target_size=image_shape[:2])
                synthetic_images, _ = generate_synthetic_data_logistic(image, time_steps)

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
    conv_lstm_model.fit(X_train, y_train, epochs=1, batch_size=1)

    all_y_true = []
    all_y_pred = []

    for filename in os.listdir(testing_folder):
        image_path = os.path.join(testing_folder, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                image = preprocess_image(image_path, target_size=image_shape[:2])
                synthetic_images, _ = generate_synthetic_data_logistic(image, time_steps)

                synthetic_images = synthetic_images.reshape(1, time_steps, *image_shape)

                predictions = conv_lstm_model.predict(synthetic_images)
                y_true = synthetic_images[0, -1, :, :, 0]  # Ground truth
                y_pred = predictions[0, :, :, 0]  # Predicted

                all_y_true.append(y_true.flatten())
                all_y_pred.append(y_pred.flatten())

                mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
                ssim_score = ssim(y_true, y_pred)
                print(f"MSE: {mse:.4f}, SSIM: {ssim_score:.4f}")

                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(y_true, cmap='gray')
                ax[0].set_title("Ground Truth")
                ax[0].axis('off')

                ax[1].imshow(y_pred, cmap='gray')
                ax[1].set_title("Prediction")
                ax[1].axis('off')

                plt.show()

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    # Flatten and compute confusion matrix and accuracy
    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)
    binary_y_true = (all_y_true > 0.5).astype(int)
    binary_y_pred = (all_y_pred > 0.5).astype(int)

    cm = confusion_matrix(binary_y_true, binary_y_pred)
    acc = accuracy_score(binary_y_true, binary_y_pred)

    print("Confusion Matrix:\n", cm)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(binary_y_true, binary_y_pred))


if __name__ == "__main__":
    main()
