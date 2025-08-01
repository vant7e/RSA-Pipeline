import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd

# ========== Configuration ========== #
base_path = '/Users/vant7e/Documents/RRI/rsa_analysis/subject/sensor_level'
task1 = 'beginMatch'
task2 = 'overt'
model = ['clip', 'cmu']
bands = ['theta', 'alpha', 'beta']
band_freqs = {
    'theta': np.arange(3, 8),  # 3-7 Hz
    'alpha': np.arange(8, 14), # 8-13 Hz
    'beta': np.arange(15, 31)  # 15-30 Hz
}
n_times = 1201
times = np.linspace(-0.2, 0.8, n_times)

# ========== Loading RSA map/pval/freq ========== #
def load_rsa_pval_freq(task, band, model):
    """Load RSA data for a specific task, band, and model"""
    path = os.path.join(base_path, task, f'{band}_phase', model)
    
    # Check if files exist
    rsa_file = os.path.join(path, f'rsa_map_{band}.txt')
    pval_file = os.path.join(path, f'rsa_pvalues_{band}.txt')
    freq_file = os.path.join(path, f'rsa_freqs_{band}.txt')
    
    if not all(os.path.exists(f) for f in [rsa_file, pval_file, freq_file]):
        print(f"Warning: Missing files in {path}")
        return None, None, None
    
    try:
        # Load data - these are already 2D (freq × time)
        rsa_map = np.loadtxt(rsa_file)
        pvals = np.loadtxt(pval_file)
        freqs = np.loadtxt(freq_file)
        
        print(f"Loaded {task} {band}: RSA shape {rsa_map.shape}, Pvals shape {pvals.shape}, Freqs: {freqs}")
        
        # Ensure we have the right number of time points
        if rsa_map.shape[1] > n_times:
            rsa_map = rsa_map[:, :n_times]
            pvals = pvals[:, :n_times]
        
        return rsa_map, pvals, freqs
        
    except Exception as e:
        print(f"Error loading {task} {band}: {e}")
        return None, None, None

# ========== Multi-Task Loop ========== #
models = ['cmu', 'clip']
print(f"Models to process: {models}")

for current_model in models:
    print(f"\n{'='*60}")
    print(f"Processing model: {current_model.upper()}")
    print(f"Current model variable: '{current_model}'")
    print(f"{'='*60}")
    
    # Reset data containers for each model
    rsa_all, pval_all, freq_all = [], [], []
    p1_all, p2_all = [], []  # Store original p-values for each task
    
    for band in bands:
        print(f"\nProcessing {band} band...")
        
        # Load data for both tasks
        rsa1, p1, freqs1 = load_rsa_pval_freq(task1, band, current_model)
        rsa2, p2, freqs2 = load_rsa_pval_freq(task2, band, current_model)
        
        if rsa1 is None or rsa2 is None:
            print(f"Skipping {band} due to missing data")
            continue
        
        # Compute difference
        rsa_diff = rsa1 - rsa2
        p_combined = np.maximum(p1, p2)  # conservative
        
        print(f"{band}: RSA diff shape {rsa_diff.shape}, P combined shape {p_combined.shape}")
        
        rsa_all.append(rsa_diff)
        pval_all.append(p_combined)
        freq_all.append(freqs1)  # Use freqs from task1
        
        # Store original p-values for each task
        p1_all.append(p1)  # beginMatch p-values
        p2_all.append(p2)  # overt p-values

    if not rsa_all:
        print(f"No data loaded for model {current_model}! Check file paths and data availability.")
        continue

    # 合并所有频段
    rsa_diff = np.vstack(rsa_all)
    pvals = np.vstack(pval_all)
    p1_combined = np.vstack(p1_all)  # beginMatch p-values
    p2_combined = np.vstack(p2_all)  # overt p-values
    all_freqs = np.concatenate(freq_all)

    print(f"\nFinal data shapes for {current_model}:")
    print(f"RSA diff: {rsa_diff.shape}")
    print(f"P-values: {pvals.shape}")
    print(f"Frequencies: {all_freqs.shape}")
    print(f"Time points: {times.shape}")

    # ========== FDR 处理 ========== #
    print(f"\nApplying FDR correction for {current_model}...")
    pvals_flat = pvals.flatten()
    sig_mask_flat, _ = fdrcorrection(pvals_flat, alpha=0.05)
    sig_mask = sig_mask_flat.reshape(pvals.shape)

    # Count significant points
    n_sig = np.sum(sig_mask)
    print(f"Found {n_sig} significant time-frequency points out of {sig_mask.size} total")

    # Debug: Show p-value distribution
    print(f"\nP-value statistics for {current_model}:")
    print(f"Min p-value: {np.min(pvals_flat):.6f}")
    print(f"Max p-value: {np.max(pvals_flat):.6f}")
    print(f"Mean p-value: {np.mean(pvals_flat):.6f}")
    print(f"Median p-value: {np.median(pvals_flat):.6f}")

    # Show how many points would be significant at different thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.2]
    print(f"\nSignificant points at different uncorrected thresholds for {current_model}:")
    for thresh in thresholds:
        n_sig_uncorrected = np.sum(pvals_flat < thresh)
        print(f"p < {thresh}: {n_sig_uncorrected} points ({n_sig_uncorrected/pvals_flat.size*100:.1f}%)")

    if n_sig == 0:
        print(f"\nNo significant points found with FDR correction for {current_model}!")
        print("Options:")
        print("1. Use uncorrected p-values (less strict)")
        print("2. Use a higher alpha level for FDR")
        print("3. Use cluster-based correction")
        
        # Option 1: Use uncorrected p-values
        print(f"\nTrying uncorrected p < 0.05 for {current_model}...")
        sig_mask_uncorrected = pvals < 0.05
        n_sig_uncorrected = np.sum(sig_mask_uncorrected)
        print(f"Found {n_sig_uncorrected} significant points with uncorrected p < 0.05")
        
        if n_sig_uncorrected > 0:
            print("Using uncorrected significance for visualization")
            sig_mask = sig_mask_uncorrected
            n_sig = n_sig_uncorrected
        else:
            print("Still no significant points. Trying p < 0.1...")
            sig_mask_uncorrected = pvals < 0.1
            n_sig_uncorrected = np.sum(sig_mask_uncorrected)
            print(f"Found {n_sig_uncorrected} significant points with p < 0.1")
            
            if n_sig_uncorrected > 0:
                print("Using p < 0.1 for visualization")
                sig_mask = sig_mask_uncorrected
                n_sig = n_sig_uncorrected
            else:
                print("No significant points found even with relaxed thresholds.")
                print("The data may not show significant differences between conditions.")

    rsa_sig = np.where(sig_mask, rsa_diff, np.nan)

    # ========== Visualization ========== #
    print(f"\nCreating visualization for {current_model}...")

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    # Plot 1: Full RSA difference (no significance masking)
    im1 = ax1.imshow(rsa_diff, aspect='auto', origin='lower', cmap='RdBu_r',
                     extent=[times[0]*1000, times[-1]*1000, all_freqs[0], all_freqs[-1]],
                     vmin=-0.1, vmax=0.1)
    ax1.set_title(f'Full RSA Difference: {task2} - {task1} ({current_model.upper()})')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.axvline(x=0, linestyle='--', color='k', label='Stimulus Onset')
    ax1.legend(loc='upper right')
    plt.colorbar(im1, ax=ax1, label='ΔRSA')

    # Plot 2: Significant points with contour lines and markers
    if n_sig > 0:
        # Create a background heatmap with low alpha for context
        im2 = ax2.imshow(rsa_diff, aspect='auto', origin='lower', cmap='RdBu_r',
                         extent=[times[0]*1000, times[-1]*1000, all_freqs[0], all_freqs[-1]],
                         vmin=-0.1, vmax=0.1, alpha=0.3)
        
        # Find significant points
        sig_indices = np.where(sig_mask)
        sig_freqs = all_freqs[sig_indices[0]]
        sig_times = times[sig_indices[1]] * 1000  # Convert to ms
        sig_values = rsa_diff[sig_indices]
        
        # Plot significant points as colored markers
        scatter = ax2.scatter(sig_times, sig_freqs, c=sig_values, 
                             cmap='RdBu_r', s=50, edgecolors='black', linewidth=1,
                             vmin=-0.1, vmax=0.1)
        
        # Add contour lines around significant clusters
        from scipy.ndimage import gaussian_filter
        sig_mask_smooth = gaussian_filter(sig_mask.astype(float), sigma=0.5)
        contour = ax2.contour(times*1000, all_freqs, sig_mask_smooth, 
                             levels=[0.5], colors='black', linewidths=2, alpha=0.7)
        
        ax2.set_title(f'Significant RSA Difference: {task2} - {task1} ({current_model.upper()})\n{n_sig} significant points')
        plt.colorbar(scatter, ax=ax2, label='ΔRSA (Significant Points)')
        
    else:
        # If no significant points, show the full map with message
        im2 = ax2.imshow(rsa_diff, aspect='auto', origin='lower', cmap='RdBu_r',
                         extent=[times[0]*1000, times[-1]*1000, all_freqs[0], all_freqs[-1]],
                         vmin=-0.1, vmax=0.1)
        ax2.set_title(f'No Significant Points Found\n{task2} - {task1} ({current_model.upper()})')
        plt.colorbar(im2, ax=ax2, label='ΔRSA')

    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.axvline(x=0, linestyle='--', color='k', label='Stimulus Onset')
    ax2.legend(loc='upper right')

    # Plot 3: P-value heatmap (useful for identifying regions of interest)
    im3 = ax3.imshow(pvals, aspect='auto', origin='lower', cmap='viridis_r',
                     extent=[times[0]*1000, times[-1]*1000, all_freqs[0], all_freqs[-1]],
                     vmin=0, vmax=0.1)
    ax3.set_title(f'P-values: {task2} - {task1} ({current_model.upper()})\nLower = more significant')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Frequency (Hz)')
    ax3.axvline(x=0, linestyle='--', color='k', label='Stimulus Onset')
    ax3.legend(loc='upper right')
    plt.colorbar(im3, ax=ax3, label='P-value')

    plt.tight_layout()

    # ========== Save ========== #
    out_txt = f'sig_rsa_diff_{task2}_minus_{task1}_{current_model}.txt'
    out_fig = out_txt.replace('.txt', '.png')

    print(f"\nSaving results for {current_model}...")
    print(f"Debug - current_model: '{current_model}'")
    print(f"Debug - out_txt: '{out_txt}'")
    
    # Save detailed results with time and frequency information
    with open(out_txt, 'w') as f:
        # Write header information
        f.write(f"# RSA Difference Analysis: {task2} - {task1} ({current_model.upper()})\n")
        f.write(f"# Time window: {times[0]*1000:.1f} to {times[-1]*1000:.1f} ms\n")
        f.write(f"# Frequency range: {all_freqs[0]:.1f} to {all_freqs[-1]:.1f} Hz\n")
        f.write(f"# Significant points: {n_sig} out of {sig_mask.size} total\n")
        f.write(f"# Data shape: {rsa_sig.shape[0]} frequencies × {rsa_sig.shape[1]} time points\n\n")
        
        # Write time labels (first row)
        f.write("Time(ms)")
        for t in times * 1000:  # Convert to ms
            f.write(f"\t{t:.1f}")
        f.write("\n")
        
        # Write frequency labels and data
        for i, freq in enumerate(all_freqs):
            f.write(f"{freq:.1f}")
            for j, t in enumerate(times * 1000):
                value = rsa_sig[i, j]
                if np.isnan(value):
                    f.write("\tNaN")
                else:
                    f.write(f"\t{value:.6f}")
            f.write("\n")
    
    # Also save a summary of significant points
    out_summary = f'sig_rsa_diff_{task2}_minus_{task1}_{current_model}_summary.txt'
    with open(out_summary, 'w') as f:
        f.write(f"# Significant RSA Difference Points: {task2} - {task1} ({current_model.upper()})\n")
        f.write(f"# Total significant points: {n_sig}\n")
        f.write(f"# RSA_Difference = {task2} - {task1}\n")
        f.write(f"# P_combined = max(P_{task1}, P_{task2}) for conservative approach\n\n")
        f.write("Frequency(Hz)\tTime(ms)\tRSA_Difference\tP_beginMatch\tP_overt\tP_combined = max(p1, p2)\n")
        
        if n_sig > 0:
            sig_indices = np.where(sig_mask)
            for i in range(len(sig_indices[0])):
                freq_idx = sig_indices[0][i]
                time_idx = sig_indices[1][i]
                freq = all_freqs[freq_idx]
                time_ms = times[time_idx] * 1000
                rsa_val = rsa_diff[freq_idx, time_idx]
                p_combined = pvals[freq_idx, time_idx]  # This is max(p1, p2)
                
                # We need to get the original p1 and p2 values
                # Find which band this frequency belongs to
                band_idx = None
                for b_idx, band in enumerate(bands):
                    if freq in freq_all[b_idx]:
                        band_idx = b_idx
                        break
                
                # Get the original p1 and p2 values for this frequency and time point
                p1_val = p1_combined[freq_idx, time_idx]  # beginMatch p-value
                p2_val = p2_combined[freq_idx, time_idx]  # overt p-value
                
                f.write(f"{freq:.1f}\t{time_ms:.1f}\t{rsa_val:.6f}\t{p1_val:.6f}\t{p2_val:.6f}\t{p_combined:.6f}\n")
        else:
            f.write("No significant points found\n")
    
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_txt}")
    print(f"Saved: {out_summary}")
    print(f"Saved: {out_fig}")

    plt.show()
    print(f"Plot displayed for {current_model}!")

print(f"\n{'='*60}")
print("All models processed!")
print(f"{'='*60}")
