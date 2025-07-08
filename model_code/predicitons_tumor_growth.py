import os, glob, pickle
import numpy as np
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import ConvLSTM2D, BatchNormalization, Conv3D
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Load Synthetic Sequences
# ----------------------------
def load_sequences(root_dir, n_frames=11, img_size=(128,128)):
    seqs = []
    for pat in os.listdir(root_dir):
        folder = os.path.join(root_dir, pat)
        frames = []
        for i in range(n_frames):
            p = os.path.join(folder, f"frame_{i}.png")
            if not os.path.exists(p): break
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)/255.0
            frames.append(img)
        if len(frames)==n_frames:
            seqs.append(np.stack(frames, axis=0))
    return np.array(seqs)[..., np.newaxis]

#building model
def build_conv_lstm(input_shape):
    model = Sequential([
        ConvLSTM2D(16, (3,3), padding='same', return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        ConvLSTM2D(8, (3,3), padding='same', return_sequences=True),
        BatchNormalization(),
        Conv3D(1, (3,3,3), activation='sigmoid', padding='same')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

def train_and_save(seqs, model_path_h5, pickle_path):
    X = seqs[:, :5]
    y = seqs[:, 5:10]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    model = build_conv_lstm(X_train.shape[1:])
    model.fit(X_train, y_train, epochs=10, batch_size=2, validation_data=(X_val, y_val))
    model.save(model_path_h5)
    with open(pickle_path, 'wb') as f:
        pickle.dump(model_path_h5, f)
    return model

#predict the images
def predict_from_flair(model, flair_img, pred_steps=10, img_size=(128,128)):
    seq = np.repeat(flair_img[np.newaxis,...], 5, axis=0)[..., np.newaxis]
    seq_list = list(seq)
    for _ in range(pred_steps):
        inp = np.expand_dims(np.stack(seq_list[-5:]), axis=0)
        out = model.predict(inp)[0, -1]
        seq_list.append(out)
    return seq_list

#generate gif

import imageio
def save_predictions(seq_list, out_dir="results", make_gif=True):
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(seq_list):
        path = os.path.join(out_dir, f"pred_{i}.png")
        imageio.imsave(path, (f.squeeze()*255).astype('uint8'))
    if make_gif:
        frames = [imageio.imread(os.path.join(out_dir, f)) for f in sorted(os.listdir(out_dir))]
        imageio.mimsave(os.path.join(out_dir, "pred.gif"), frames, fps=1)

if __name__=="__main__":
    sequences = load_sequences(r"C:\Users\Win 10\Downloads\mpw3-master (1)\mpw3-master\synthetic images\training")
    model = train_and_save(sequences, "conv_lstm_growth.h5", "model_path.pkl")

    flair = cv2.imread(r"synthetic images\training\BraTS20_Training_001\flair_middle.png", cv2.IMREAD_GRAYSCALE)
    flair = cv2.resize(flair, (128,128))/255.0
    seq_list = predict_from_flair(model, flair)
    save_predictions(seq_list, out_dir="results")
    print("✅ Done. Check the 'results' folder.")
