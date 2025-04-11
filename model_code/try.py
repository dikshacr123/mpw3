import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import cv2

# Load the saved model from pkl
with open("conv_lstm_tumor_growth_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load and preprocess one flair image from the testing folder
test_image_path = os.path.join("synthetic images", "testing","BraTS20_Training_003", "flair_middle.png")
test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)

# Resize and normalize
test_img = cv2.resize(test_img, (224, 224))
test_img = test_img.astype('float32') / 255.0

# Expand dims to create a batch of one with repeated frames
# Shape needed: (1, time_steps, height, width, channels)
time_steps = 10
test_sequence = np.repeat(test_img[np.newaxis, :, :, np.newaxis], time_steps, axis=0)
test_sequence = np.expand_dims(test_sequence, axis=0)

# Predict
predicted_growth = model.predict(test_sequence)[0]

# Show prediction
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(test_img, cmap='gray')
plt.title("Input Flair Image")

plt.subplot(1, 2, 2)
plt.imshow(predicted_growth[:, :, 0], cmap='gray')
plt.title("Predicted Tumor Growth")
plt.show()
