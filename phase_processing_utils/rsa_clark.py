import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform
import pandas as pd
import time
import sys

# === Select frequency band: 'theta', 'alpha', or 'beta' ===
band = 'beta'  # options: 'theta', 'alpha', 'beta'

# === Load brain RDMs and model RDM ===
brain_rdms = np.load('/Users/vant7e/Documents/RRI/rsa_analysis/subject/sub-20284/rdms_brain_phase/brain_rdms_overt.npy')  # shape: (n_freqs, n_times, n_trials, n_trials)
model_rdm = np.load('/Users/vant7e/Documents/RRI/rsa_analysis/output_rdm/beginMatch/rdms/clip/rdm_clip_overt.npy')  # shape: (n_trials, n_trials)

# === Set frequency indices and labels based on band ===
if band == 'theta':
    freqs = np.arange(3, 8, 1)  # 3–7 Hz
    brain_rdms = brain_rdms[0:5]  # indices 0,1,2,3,4 (3–7 Hz)
elif band == 'alpha':
    freqs = np.arange(8, 14, 1)  # 8–13 Hz
    brain_rdms = brain_rdms[5:11]  # indices 5,6,7,8,9,10 (8–13 Hz)
elif band == 'beta':
    freqs = np.arange(14, 31, 1)  # 14–30 Hz
    brain_rdms = brain_rdms[11:28]  # indices 11–27 (14–30 Hz)
else:
    raise ValueError("band must be 'theta', 'alpha', or 'beta'")

n_freqs, n_times, n_trials, _ = brain_rdms.shape
print(f"Analyzing {band} band: {freqs} Hz (shape: {brain_rdms.shape})")

# === Check for dimension match ===
if model_rdm.shape[0] != n_trials or model_rdm.shape[1] != n_trials:
    print(f"ERROR: Model RDM shape {model_rdm.shape} does not match brain RDM trials {n_trials}.")
    print("You may need to remove extra labels or ensure both RDMs are for the same set of trials.")
    sys.exit(1)

# === Prepare for RSA computation ===
rsa_map = np.zeros((n_freqs, n_times))
p_map = np.ones((n_freqs, n_times))
n_perm = 1000  # Number of permutations

# Get upper triangle indices
iu = np.triu_indices(n_trials, k=1)
model_rdm_flat = model_rdm[iu]

start_time = time.time()
print(f"Starting RSA computation for {n_freqs} frequencies and {n_times} time points...")

for f in range(n_freqs):
    freq_start = time.time()
    for t in range(n_times):
        brain_rdm_flat = brain_rdms[f, t][iu]
        # Compute Spearman correlation
        rsa_score, _ = spearmanr(brain_rdm_flat, model_rdm_flat)
        rsa_map[f, t] = rsa_score

        # Permutation test
        null_dist = []
        for _ in range(n_perm):
            perm = np.random.permutation(n_trials)
            perm_model_rdm_flat = model_rdm[perm][:, perm][iu]
            null_score, _ = spearmanr(brain_rdm_flat, perm_model_rdm_flat)
            null_dist.append(null_score)
        null_dist = np.array(null_dist)
        # Two-tailed p-value
        p = (np.sum(np.abs(null_dist) >= np.abs(rsa_score)) + 1) / (n_perm + 1)
        p_map[f, t] = p
    freq_time = time.time() - freq_start
    print(f"  Done frequency {freqs[f]:.1f} Hz ({f+1}/{n_freqs}) in {freq_time:.2f} seconds.")

total_time = time.time() - start_time
print(f"All RSA computations finished in {total_time/60:.2f} minutes.")

# === Save RSA map, p-values, and frequency axis ===
np.savetxt(f'rsa_map_{band}.txt', rsa_map)
np.savetxt(f'rsa_pvalues_{band}.txt', p_map)
np.savetxt(f'rsa_freqs_{band}.txt', freqs)

# === Visualization ===
plt.figure(figsize=(12, 6))
plt.imshow(rsa_map, aspect='auto', origin='lower', cmap='RdBu_r',
           extent=[0, n_times-1, freqs[0], freqs[-1]])
plt.colorbar(label='RSA (Spearman r)')
plt.xlabel('Time Index')
plt.ylabel('Frequency (Hz)')
plt.title(f'RSA (Overt vs. CLIP) - {band} band')
plt.yticks(freqs)

# Mark significant results (e.g., p < 0.05, FDR correction can be added)
sig = p_map < 0.05
y, x = np.where(sig)
plt.scatter(x, freqs[y], color='k', s=5, label='p < 0.05')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f'rsa_map_{band}.png', dpi=300)
plt.show()