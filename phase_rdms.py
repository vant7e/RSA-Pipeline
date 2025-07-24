import mne
import numpy as np
from scipy.spatial.distance import pdist, squareform
from mne.time_frequency import tfr_morlet
import os
import sys
import glob
from tqdm import tqdm
import logging

# === User frequency selection ===
# Set frequencies to a numpy array (e.g., np.arange(4, 31, 1)) to use custom frequencies,
# or set to None to use all available frequencies (default: 3–30 Hz, 1 Hz step)
frequencies = np.arange(4, 31, 1)  # e.g., np.arange(4, 31, 1) or None for auto

# === Subject list ===
subjects = []  # Leave empty to auto-detect all sub-* folders
subjects_dir = '/RRI/rsa_analysis/subject'

# === Auto-detect subjects if not defined ===
if not subjects:
    subjects = [d for d in os.listdir(subjects_dir)
                if os.path.isdir(os.path.join(subjects_dir, d)) and d.startswith('sub-')]
    print(f"Auto-detected subjects: {subjects}")

# === Device selection (placeholder, as MNE is CPU-based) ===
use_gpu = False  # Set to True if you want to use GPU manually
if use_gpu:
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA GPU for computation.")
        else:
            device = torch.device("cpu")
            print("CUDA GPU not available. Falling back to CPU.")
    except ImportError:
        device = "cpu"
        print("PyTorch not installed. Using CPU.")
else:
    device = "cpu"
    print("Using CPU for computation.")

# === Time window selection ===
# Set time_window to None to use the full epoch time range
time_window = None  # e.g., (-0.2, 0.8) or None for full window

# === Chunking parameters ===
chunk_size = 5  # Number of frequencies per chunk

for subject in subjects:
    try:
        print(f"\n=== Processing {subject} ===")
        epochs_path = os.path.join(subjects_dir, subject, f'{subject}_{task}_epo.fif') #set correct epoch path in here
        output_dir = os.path.join(subjects_dir, subject, "rdms_brain_phase")
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, "phase_processing.log")
        # Set up logging for this subject
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(log_path, mode='w'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger()
        logger.info(f"=== Processing {subject} ===")

        # === Load epochs ===
        epochs = mne.read_epochs(epochs_path, preload=True)

        # === Frequency definition ===
        if frequencies is None:
            sfreq = epochs.info['sfreq']
            fmax = int(sfreq // 2 - 1)  # Nyquist
            frequencies_subj = np.arange(3, fmax + 1, 1)
            logger.info(f"No frequencies defined by user. Using default: {frequencies_subj}")
        else:
            frequencies_subj = frequencies
            logger.info(f"User-defined frequencies: {frequencies_subj}")
        n_cycles = frequencies_subj / 2.

        # === Process in chunks ===
        n_chunks = int(np.ceil(len(frequencies_subj) / chunk_size))
        logger.info(f"Processing {len(frequencies_subj)} frequencies in {n_chunks} chunks (chunk size: {chunk_size})")

        for chunk_idx in tqdm(range(n_chunks), desc=f"Chunks ({subject})", position=0):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, len(frequencies_subj))
            freqs_chunk = frequencies_subj[start:end]
            n_cycles_chunk = n_cycles[start:end]
            logger.info(f"Processing frequencies: {freqs_chunk}")

            # Compute TFR for this chunk
            tfr = tfr_morlet(
                epochs, freqs=freqs_chunk, n_cycles=n_cycles_chunk,
                return_itc=False, output='complex', use_fft=True, decim=1, n_jobs=1, average=False
            )
            complex_data = tfr.data  # (n_epochs, n_channels, n_freqs_chunk, n_times)
            phase_data = np.angle(complex_data)

            # Time window selection
            times = tfr.times
            if time_window is not None:
                t_min, t_max = times[0], times[-1]
                if time_window[0] < t_min or time_window[1] > t_max:
                    logger.warning(f"⚠️ Specified time_window {time_window} exceeds bounds ({t_min:.2f}, {t_max:.2f}). Clipping.")
                    time_window = (max(time_window[0], t_min), min(time_window[1], t_max))
                time_mask = (times >= time_window[0]) & (times <= time_window[1])
                phase_data = phase_data[:, :, :, time_mask]
                times = times[time_mask]
                logger.info(f"Using time window: {time_window}")
            else:
                logger.info(f"Using full time window: {times[0]:.3f} to {times[-1]:.3f} s")

            n_epochs, n_channels, n_freqs_chunk, n_times = phase_data.shape
            brain_rdms = np.zeros((n_freqs_chunk, n_times, n_epochs, n_epochs), dtype=np.float32)

            for f_idx in tqdm(range(n_freqs_chunk), desc=f"Freqs {freqs_chunk[0]}-{freqs_chunk[-1]} Hz", position=1, leave=False):
                for t_idx in range(n_times):
                    phase_vecs = phase_data[:, :, f_idx, t_idx]  # (n_epochs, n_channels)
                    phase_vecs = np.nan_to_num(phase_vecs, nan=0.0, posinf=0.0, neginf=0.0)
                    brain_rdm = squareform(pdist(phase_vecs, metric='correlation'))
                    brain_rdms[f_idx, t_idx] = brain_rdm
                logger.info(f"  Done frequency {freqs_chunk[f_idx]:.1f} Hz ({f_idx+1}/{n_freqs_chunk})")

            # Save this chunk
            chunk_fname = f"brain_rdms_{freqs_chunk[0]:.0f}-{freqs_chunk[-1]:.0f}Hz.npy"
            np.save(os.path.join(output_dir, chunk_fname), brain_rdms)
            logger.info(f"Saved chunk to {os.path.join(output_dir, chunk_fname)}")

        # === Merge all frequency chunks into a single brain_rdms.npy ===
        chunk_files = sorted(glob.glob(os.path.join(output_dir, "brain_rdms_*Hz.npy")))
        logger.info(f"Merging {len(chunk_files)} chunk files into a single brain_rdms.npy...")
        all_rdms = []
        for fpath in chunk_files:
            logger.info(f"  Loading {fpath}")
            rdm = np.load(fpath)
            all_rdms.append(rdm)
        # Safety check
        for i, rdm in enumerate(all_rdms):
            assert rdm.shape[2:] == all_rdms[0].shape[2:], f"Incompatible RDM shape in chunk {chunk_files[i]}"
        brain_rdms_all = np.concatenate(all_rdms, axis=0)
        combined_path = os.path.join(output_dir, "brain_rdms.npy")
        np.save(combined_path, brain_rdms_all)
        logger.info(f"Saved combined RDMs to {combined_path} (shape: {brain_rdms_all.shape})")
        print(f"✅ Finished {subject}")
    except Exception as e:
        print(f"Error processing {subject}: {e}")
        continue 
