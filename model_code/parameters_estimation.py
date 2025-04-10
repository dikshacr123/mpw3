from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import csv

# Load data
df = pd.read_csv(r"FE\tumor_features.csv")
bbox = df["Bounding_Box_Dimensions"].str.extract(r'(\d+\.?\d*) x (\d+\.?\d*) x (\d+\.?\d*)').astype(float)
bbox.columns = ['Width_mm', 'Height_mm', 'Depth_mm']
bounding_box_volume = bbox.prod(axis=1)
V0_all = df["Tumor_Volume_mm3"].values
t0_all = np.zeros(len(V0_all))

# Define Richards model with clamp
def safe_richards(t, K, B, Q, M, nu):
    z = 1 + Q * np.exp(np.clip(-B * (t - M), -50, 50))  # clip to avoid overflow
    z = np.clip(z, 1e-6, 1e6)  # clip base to avoid invalid powers
    return K * z ** (-1 / nu)

# Initial guess
initial = [np.median(1.5 * bounding_box_volume), 0.1, 10, 5, 1.0]

# Reasonable parameter bounds
bounds = (
    [1e3,  1e-3,  0.1, 0.0, 0.1],    # Lower bounds
    [1e7,  1.0,  100.0, 10.0, 5.0]    # Upper bounds
)

# Fit the model
popt, _ = curve_fit(safe_richards, t0_all, V0_all, p0=initial, bounds=bounds, maxfev=10000)

# Output
param_names = ["K", "B", "Q", "M", "nu"]
global_params = dict(zip(param_names, popt))
print(global_params)

with open(r"model_code\global_richards_params.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Parameter", "Value"])
    for param, value in global_params.items():
        writer.writerow([param, value])

