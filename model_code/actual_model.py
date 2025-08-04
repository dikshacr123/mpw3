import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, ConvLSTM2D, TimeDistributed, BatchNormalization
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def build_convlstm_model(input_shape=(8, 240, 240, 3), num_future=2):
    """
    ConvLSTM model that takes 8 RGB frames (240x240) and predicts the next 2 frames.
    
    input_shape: (T_in, H, W, C)
    num_future: number of future frames to predict
    """
    input_seq = Input(shape=input_shape)  # (batch, 8, 240, 240, 3)

    # Encoder block
    x = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(input_seq)
    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='relu'))(x)

    # ConvLSTM layer: now returns sequences
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True
    )(x)

    x = BatchNormalization()(x)

    # Repeat ConvLSTM to simulate future steps (optionally use more ConvLSTM layers)
    x = ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding='same',
        return_sequences=True
    )(tf.repeat(x[:, -1:, :, :, :], repeats=num_future, axis=1))  # repeat last state

    # Decoder block
    x = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='relu'))(x)
    x = TimeDistributed(Conv2D(3, (3, 3), padding='same', activation='sigmoid'))(x)  # RGB output

    model = Model(inputs=input_seq, outputs=x)
    return model

def load_image(path):
    """Loads and preprocesses an image."""
    img = cv2.imread(path)  # reads as BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB
    img = img.astype(np.float32) / 255.0  # normalize
    return img  # shape: (240, 240, 3)

def load_dataset(X_paths, Y_paths):
    X_data = []
    Y_data = []

    for patient_x, patient_y in zip(X_paths, Y_paths):
        # Load all 8 input frames
        x_seq = [load_image(p) for p in patient_x]
        X_data.append(x_seq)

        # Load 2 target frames
        y_seq = [load_image(p) for p in patient_y]
        Y_data.append(y_seq)

    X_data = np.array(X_data)  # shape: (N, 8, 240, 240, 3)
    Y_data = np.array(Y_data)  # shape: (N, 2, 240, 240, 3)

    return X_data, Y_data

model = build_convlstm_model(input_shape=(8, 240, 240, 3), num_future=2)
model.compile(optimizer='adam', loss='mse')
model.summary()


base_dir = "mpw3-master/synthetic images"
X = []
Y = []

# List all folders in the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Make sure it's a directory and matches BraTS20_Training_XXX
    if os.path.isdir(folder_path) and folder_name.startswith("BraTS20_Training_"):
        try:
            # Check if all 10 frames exist
            input_frames = [os.path.join(folder_path, f"frame_{i}.png") for i in range(8)]
            target_frames = [os.path.join(folder_path, f"frame_{i}.png") for i in range(8, 10)]

            # Ensure all files exist
            if all(os.path.exists(f) for f in input_frames + target_frames):
                X.append(input_frames)
                Y.append(target_frames)
        except Exception as e:
            print(f"Error in {folder_name}: {e}")

base_test_dir = "mpw3-master/Testing_images"
X_test = []
Y_actual = []
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Make sure it's a directory and matches BraTS20_Training_XXX
    if os.path.isdir(folder_path) and folder_name.startswith("BraTS20_Training_"):
        try:
            # Check if all 10 frames exist
            input_frames = [os.path.join(folder_path, f"frame_{i}.png") for i in range(8)]
            target_frames = [os.path.join(folder_path, f"frame_{i}.png") for i in range(8, 10)]

            # Ensure all files exist
            if all(os.path.exists(f) for f in input_frames + target_frames):
                X_test.append(input_frames)
                Y_actual.append(target_frames)
        except Exception as e:
            print(f"Error in {folder_name}: {e}")

X_data, Y_data = load_dataset(X, Y)
print(X_data.shape)

model.fit(X_data, Y_data, batch_size=1, epochs=10)
Y_predict = model.predict(X_test)
output_folder = "predicted_frames"
os.makedirs(output_folder, exist_ok=True)
import matplotlib.pyplot as plt

for patient_idx in range(Y_predict.shape[0]):
    for frame_idx in range(Y_predict.shape[1]):
        frame = Y_predict[patient_idx, frame_idx]  # shape: (240, 240, 3)
        
        # Create a filename like: patient_000_frame_8.png
        filename = f"patient_{patient_idx:03d}_frame_{frame_idx + 8}.png"
        filepath = os.path.join(output_folder, filename)
        
        # Save image (auto-scales pixel values in [0, 1])
        plt.imsave(filepath, frame)

# Convert tensors to numpy if needed
Y_pred = Y_predict.astype(np.float32)
Y_actual = Y_actual.astype(np.float32)

# MSE
mse = np.mean((Y_pred - Y_actual) ** 2)

# MAE
mae = np.mean(np.abs(Y_pred - Y_actual))

# PSNR
psnr_score = psnr(Y_actual, Y_pred, data_range=1.0)

# SSIM for one image pair
ssim_score = ssim(Y_actual[0,0], Y_pred[0,0], multichannel=True, data_range=1.0)

print("MSE:", mse)
print("MAE:", mae)
print("PSNR:", psnr_score)
print("SSIM:", ssim_score)

#plt.figure(figsize=(10, 4))
#for t in range(Y_predict.shape[1]):
#    plt.subplot(1, 2, t+1)
#    plt.imshow(Y_predict[0, t])
#    plt.title(f"Predicted t+{t+1}")
#    plt.axis("off")
#plt.tight_layout()
#plt.show()