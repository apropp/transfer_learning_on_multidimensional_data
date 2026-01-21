'''
UQ_experiment.py: Sample code to run UQ on surrogate model saved at 'MODEL_PATH'.
                    - Compares 2000 forward passes of surrogate model to standard
                      Monte Carlo with 6, 12, 24, and 48 hours' worth of data
                    - Saves pdf and cdf of surrogate model and each version of MC
                    - Saves several metrics of error for each approach to UQ, including:
                        - KL divergence
                        - Wasserstein distance
                        - RMSE
                        - MSE
                        - MAE

Title:  TRANSFER LEARNING ON MULTI-DIMENSIONAL DATA: 
        A NOVEL APPROACH TO NEURAL NETWORK-BASED SURROGATE MODELING
        DOI: 10.1615/JMachLearnModelComput.2024057138

Findings: Our main finding is that the CNN surrogate model performs
          comparably to MC with 24 or 48 hours' worth of data.
                - This result is robust to all metrics of error tested.
                - In our manuscript, we perform UQ with the CNN surrogate trained on
                  270 (low frequency) 1D solutions and 45 2D solutions.
                - However, results are robust to other splits of the
                  training data between 1D and 2D.

author: A.M. Propp
email: propp@stanford.edu
GitHub: apropp
Updated: May 2024
'''

import csv
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error
import h5py
import time

# Local imports (Assuming these are custom modules you've written)
from utils_UQ import bt_cdf_pdf_MC, bt_cdf_pdf_TL
from dense_ed_multidimensional import DenseED
from utils_surrogate import *

# Constants
MODEL_PATH = '../models/pretrained_surrogate_example.pth'
DATA_PATH = '../data/'
RESULTS_PATH = '../'
UQ_ERRORS_FILENAME = RESULTS_PATH + 'UQ_errors.txt'
N_TRIALS = 50
N_datafiles = 7
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sat_th, bt_dist, max_dist = 0.15, 100, 150

def load_hdf5_data(filename):
    with h5py.File(filename, 'r') as f:
        ks = np.transpose(f['ks'], (3,2,0,1)) * (10**14)
        ps = np.transpose(f['ps'], (3,2,0,1))
        ss = np.transpose(f['ss'], (3,2,0,1))
    return ks, ps, ss

def load_and_combine_data(filenames):
    ks_full, ps_full, ss_full = [], [], []
    for filename in filenames:
        ks, ps, ss = load_hdf5_data(DATA_PATH + filename)
        ks_full.append(ks)
        ps_full.append(ps)
        ss_full.append(ss)
    return {
        'ks_hfs': np.concatenate(ks_full, axis=0),
        'ps_hfs': np.concatenate(ps_full, axis=0),
        'ss_hfs': np.concatenate(ss_full, axis=0)
    }


def setup_model():
    model = DenseED(1, 16, blocks=(7,12,7), growth_rate=40, drop_rate=0, bn_size=8, num_init_features=128, bottleneck=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def perform_model_inference(model, data):
    results = np.zeros((data.shape[0], 16, 128, 128))
    for i, datum in enumerate(data):
        tensor = torch.tensor(datum, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            output = model(tensor)
        results[i] = output.cpu().numpy()
    return results

def compute_kl_divergence(p, q, sigma=0):
    if sigma != 0:
        p = gaussian_filter(p, sigma)
        q = gaussian_filter(q, sigma)
    p = np.clip(p / p.sum(), 1e-10, None)
    q = np.clip(q / q.sum(), 1e-10, None)
    return np.sum(p * np.log(p / q))

print("Loading data...")
filenames = [f'DATA_2D_v{i+1}.hdf5' for i in range(N_datafiles)]
data = load_and_combine_data(filenames)

print("Loading surrogate model...")
model = setup_model()
model_output = perform_model_inference(model, data['ks_hfs'])


# Indicate how many simulation runs we can use based on budget
dps_values = {
    "full": None,
    "6hMC": 70,
    "12hMC": 139,
    "24hMC": 278,
    "48hMC": 556,
    "model": 2000
}


# Initialize result dictionaries
cdf_results = {key: [] for key in dps_values}
pdf_results = {key: [] for key in dps_values}
errors = {key: {"KL": [], "WD": [], "RMSE": [], "MSE": [], "MAE": []} for key in dps_values if key != "full"}

# Run UQ and collect results
print("Running UQ...")
for _ in range(N_TRIALS):
    for label, dps in dps_values.items():
        if label == 'model':
            cdf, pdf = bt_cdf_pdf_TL(model_out=model_output, dps=dps, threshold=sat_th, dist_breakthru=bt_dist)
        if label == 'full':
            cdf, pdf = bt_cdf_pdf_MC(dat=data, qoi='ss_hfs', dps=False, threshold=sat_th, dist_breakthru=bt_dist, dist_max=max_dist)
        else:
            cdf, pdf = bt_cdf_pdf_MC(dat=data, qoi='ss_hfs', dps=dps, threshold=sat_th, dist_breakthru=bt_dist, dist_max=max_dist)
        cdf_results[label].append(cdf)
        pdf_results[label].append(pdf)

# Calculate errors
print("Calculating errors...")
for label in dps_values:
    if label == "full":
        continue
    ground_truth_pdf = pdf_results["full"]
    model_pdf = pdf_results[label]
    for gt, model in zip(ground_truth_pdf, model_pdf):
        errors[label]["KL"].append(compute_kl_divergence(gt, model))
        errors[label]["WD"].append(wasserstein_distance(gt, model))
        errors[label]["RMSE"].append(mean_squared_error(gt, model, squared=False))
        errors[label]["MSE"].append(mean_squared_error(gt, model))
        errors[label]["MAE"].append(mean_absolute_error(gt, model))

# Prepare and save detailed results
print("Saving...")
metrics = ["KL", "WD", "RMSE", "MSE", "MAE"]
df_results = pd.DataFrame(index=dps_values.keys(), columns=[f"{metric}_mean" for metric in metrics] + [f"{metric}_std" for metric in metrics])

for label in dps_values:
    if label == "full":
        continue
    for metric in metrics:
        df_results.loc[label, f"{metric}_mean"] = np.mean(errors[label][metric])
        df_results.loc[label, f"{metric}_std"] = np.std(errors[label][metric])

# Save errors to csv
df_results.to_csv(UQ_ERRORS_FILENAME)

# Save CDFs and PDFs to CSV
for label in dps_values:
    np.savetxt((RESULTS_PATH + f"cdf_{label}.csv"), cdf_results[label], delimiter=",")
    np.savetxt((RESULTS_PATH + f"pdf_{label}.csv"), pdf_results[label], delimiter=",")