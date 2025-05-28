# Preprocessing Pipeline for MNE (Raw ➔ Epochs ➔ Evoked ➔ Trans ➔ Forward ➔ Inverse ➔ Source)
# =============================================================
# How to run (from terminal): 
# >>> python Preprocessing_Pipeline.py --subject sub-20284 --task overt
# >>> python batch_processing.py --subjects sub-20284 sub-20285 --tasks overt beginMatch
# =============================================================
#--raw     Convert CTF to raw.fif
#--trans   Create transformation matrix
#--epochs  Create epochs from events
#--evoked  Compute evoked response
#--fwd     Compute forward model
#--inv     Compute inverse operator
#--stc     Apply inverse to epochs → STCs
#--None    Appky all functions
# =============================================================
# Please making sure MNE and other dependencies are installed in your environment (see requirnment.txt)
# /your_project/
#  └── subject/
#      └── sub-200001/
#          ├── sub-20284.ds
#          ├── bem/
#          │   ├── sub-20284-oct-6-src.fif
#          │   └── sub-20284-bem-sol.fif
#          └── (outputs will be generated here)
# =============================================================
import os
import sys
import mne
import argparse
import numpy as np
import logging
import shutil
from datetime import datetime
from mne.viz import plot_sensors

class StreamToLogger(object):
    """
    Redirects writes to logger instance.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):  # Needed for compatibility
        pass
    
# Setup logging
def setup_logging(log_path):
    """
    Sets up logging to both a file and the console.
    """
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# Convert .ds file to raw.fif
def run_raw(subject, data_dir, task):
    """
    Converts CTF (.ds) file to MNE Raw (.fif)
    Sets montage and optionally applies filtering.
    """
    ds_path = os.path.join(data_dir, f"{task}.ds")  # Use task-specific .ds folder
    raw_out = os.path.join(data_dir, f"{subject}_{task}_raw.fif")

    if not os.path.exists(ds_path):
        raise FileNotFoundError(f"[ERROR] .ds folder not found: {ds_path}")
    
    # Customize filters below if needed
    # raw = mne.io.read_raw_ctf(ds_path, preload=True).filter(l_freq=1.0, h_freq=40.0)
    # l_freq = low frequency; h_freq = high frequency
    logging.info(f"[RAW] Loading: {ds_path}")
    raw = mne.io.read_raw_ctf(ds_path, preload=True)
    
    raw.set_montage("standard_1005", match_case=False) # Set standard 10-05 montage
    
    # Save as .fif for further preprocessing
    raw.save(raw_out, overwrite=True)

    # Sensor visualization
    try:
        raw_mag = raw.copy().pick_types(meg='mag')  # Optional: reduce clutter
        raw_mag.plot_sensors(kind='3d', show_names=False)  # Use picked copy
    except Exception as e:
        logging.warning(f"[RAW] Could not plot sensors in 3D: {e}")

    # Plot topomap view (without names to avoid clutter)
    try:
        raw.plot_sensors(kind='topomap', show_names=False)
    except Exception as e:
        logging.warning(f"[RAW] Could not plot sensors in topomap view: {e}")
        
def run_trans(subject, data_dir, task, trans_method="mri-fiducials", use_gui=False):
    """
    Computes MRI-to-head transformation (.trans.fif) using fiducials or MNE GUI.

    Parameters:
    - trans_method: 'mri-fiducials' (default), 'digitals', or 'manual'
    - use_gui: If True, launches the MNE GUI for manual coregistration
    """
    raw_fif = os.path.join(data_dir, f"{subject}_{task}_raw.fif")
    trans_out = os.path.join(data_dir, f"{subject}_{task}_trans.fif")
    raw = mne.io.read_raw_fif(raw_fif, preload=False)

    # Dynamically determine subject_name based on actual directory structure
    subjects_dir = os.getenv("SUBJECTS_DIR")
    if subjects_dir is None:
        raise EnvironmentError("Please set SUBJECTS_DIR before running trans computation.")

    # Support both 'sub-20284' and '20284' folder naming
    possible_names = [subject.replace("sub-", ""), subject]
    for name in possible_names:
        check_path = os.path.join(subjects_dir, name, "mri", "T1.mgz")
        if os.path.exists(check_path):
            subject_name = name
            break
    else:
        raise FileNotFoundError(
            f"Neither {possible_names[0]} nor {possible_names[1]} found under {subjects_dir}. "
            f"Expected mri/T1.mgz inside subject folder."
        )

    logging.info(f"[TRANS] Using SUBJECT folder: {subject_name}, {subjects_dir}")

    # Backup existing .trans file if present
    if os.path.exists(trans_out):
        backup_path = trans_out.replace(".fif", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fif")
        shutil.copy2(trans_out, backup_path)
        logging.info(f"[TRANS] Existing .trans file backed up to: {backup_path}")

    if use_gui:
        logging.info("[TRANS] Launching MNE Coreg GUI...")
        mne.gui.coregistration(subject=subject_name, inst=raw.info, subjects_dir=subjects_dir)
        logging.info("[TRANS] Please save the .trans file manually via the GUI.")
        if os.path.exists(trans_out):
            logging.info(f"[TRANS] Confirmed saved: {trans_out}")
        else:
            logging.warning(f"[TRANS] No .trans file found at expected path: {trans_out}")
        return

    logging.info("[TRANS] Running auto coregistration using fiducials...")
    coreg = mne.coreg.Coregistration(subject=subject_name, subjects_dir=subjects_dir, info=raw.info)
    coreg.fit_fiducials(verbose=True)
    trans = coreg.trans

    mne.write_trans(trans_out, trans, overwrite=True)
    logging.info(f"[TRANS] Saved to: {trans_out}")
    return trans

def run_epochs(subject, data_dir, task):
    """
    Extract epochs around stimulus events.

    This step:
    - Loads raw .fif data
    - Loads bad channel list (if available)
    - Extracts events from annotations
    - Defines stimulus-locked epochs (-200ms to +800ms)
    - Optionally drops bad epochs
    - Saves .fif and shows interactive preview

    Parameters to modify:
    - tmin / tmax (epoch window)
    - baseline (e.g., (None, 0) means full pre-stimulus baseline)
    - event_dict (adjust if your markers differ)
    - rejection thresholds (for MEG/EEG/iEEG)
    """

    raw_fif = os.path.join(data_dir, f"{subject}_{task}_raw.fif")
    epo_out = os.path.join(data_dir, f"{subject}_{task}_epo.fif")
    raw = mne.io.read_raw_fif(raw_fif, preload=True)

    # Load optional bad channels 
    bad_path = os.path.join(data_dir, "BadChannels.txt")
    if os.path.exists(bad_path):
        with open(bad_path, 'r') as f:
            bads = [line.strip() for line in f.readlines()]
        raw.info['bads'] = bads
        logging.info(f"[EPOCHS] Loaded bad channels: {bads}")
    else:
        logging.info("[EPOCHS] No BadChannels.txt found.")

    # Event extraction
    events, event_id = mne.events_from_annotations(raw)
    if not event_id:
        raise RuntimeError("No events found! Check if annotations exist in your raw file.")

    # Customize this mapping if your triggers are different, e.g.,{'stimulus': 1, 'response': 2})
    event_dict = {"stimulus": list(event_id.values())[0]}
    logging.info(f"[EPOCHS] Event ID: {event_dict}")

    # Epoch configuration: Adjust tmin, tmax, and baseline for your design
    epochs = mne.Epochs(
        raw, events, event_id=event_dict,
        tmin=-0.2, tmax=0.8,
        baseline=(None, 0),
        preload=True
    )

    # baseline=(None, 0) means: Use the entire pre-stimulus interval (e.g., -0.2s to 0s) to remove DC offset (baseline correction)
    # preload=True means: Load all data into memory now, enabling faster access, ICA, rejection, and saving
    # Optional epoch rejection
    # reject_criteria = dict(mag=4e-12)  # Example for MEG
    # epochs.drop_bad(reject=reject_criteria)

    logging.info(f"[EPOCHS] Number of good epochs: {len(epochs)}")
    epochs.save(epo_out, overwrite=True)
    logging.info(f"[EPOCHS] Saved to: {epo_out}")
    epochs.plot(n_epochs=10, events=True, block=True) # visualization options for automatically saving the plots, n_epoch = n channels canbe making adjustable based on needs.

    return epochs

# Evoked (Averaging)
def run_evoked(subject, data_dir, task):
    """
    Computes average evoked response from epochs.
    """
    epo_fif = os.path.join(data_dir, f"{subject}_{task}_epo.fif")
    ave_out = os.path.join(data_dir, f"{subject}_{task}_ave.fif")
    epochs = mne.read_epochs(epo_fif, preload=True)
   
    evoked = epochs.average() # Average all epochs
    # Baseline correction has been applied. This happened automatically during creation of the epochs object
    # "Baseline correction" may also be initiated (or disabled) manually
    # The information about the baseline period of Epochs is transferred to derived Evoked objects to maintain provenance as you process your data
    
    print(f"Epochs baseline: {epochs.baseline}") 
    print(f"Evoked baseline: {evoked.baseline}")
    
    evoked.save(ave_out, overwrite=True)
    logging.info(f"[EVOKED] Saved to: {ave_out}")
    return evoked

# Forward Model
def run_forward(subject, data_dir, task):
    """
    Computes forward model from head model and sensor config.
    Requires .trans, source space (.src), and BEM model (.bem-sol).
    """
    raw_fif = os.path.join(data_dir, f"{subject}_{task}_raw.fif")
    trans = os.path.join(data_dir, f"{subject}_{task}_trans.fif")
    src = os.path.join(data_dir, "bem", f"{subject}-oct-6-src.fif")
    bem = os.path.join(data_dir, "bem", f"{subject}-bem-sol.fif")
    fwd_out = os.path.join(data_dir, f"{subject}_{task}_fwd.fif")
    raw = mne.io.read_raw_fif(raw_fif, preload=False)

    fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=1, verbose=None)
        
    #n_jobs = the number of jobs to run in parallel. If n_jobs = -1, it is set to the number of CPU cores. Requires the joblib package.
    #n_jobs = None is a marker for ‘unset’ that will be interpreted as n_jobs=1 (sequential execution)
    
    #mindist = 5.0 mm -> Minimum distance of sources from inner skull surface (in mm). 
    #eeg = False, If True (default), include EEG computations.
    #meg = True, If True (default), include MEG computations.
    #verbose = control verbosity of the logging output. If None, use the default verbosity level.
    #More Info: https://mne.tools/stable/generated/mne.make_forward_solution.html
    
    mne.write_forward_solution(fwd_out, fwd, overwrite=True)
    logging.info(f"[FORWARD] Saved to: {fwd_out}")
    return fwd

# Inverse Operator
def run_inverse(subject, data_dir, task):
    """
    Computes inverse operator from fwd, epochs, and noise covariance.
    "loose=0.2" means partially free orientation; "depth=0.8" applies depth weighting.
    """
    epo_fif = os.path.join(data_dir, f"{subject}_{task}_epo.fif")
    fwd_fif = os.path.join(data_dir, f"{subject}_{task}_fwd.fif")
    inv_out = os.path.join(data_dir, f"{subject}_{task}_inv.fif")
    cov_out = os.path.join(data_dir, "source_outputs", "overt", "noise-cov.fif")
    epochs = mne.read_epochs(epo_fif, preload=True)
    fwd = mne.read_forward_solution(fwd_fif)
    cov = mne.compute_covariance(epochs, tmax=0, method="empirical")
    mne.write_cov(cov_out, cov, overwrite=True)
   
    inv = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, cov, loose=0.2, depth=0.8)
    
    #loose = 'auto' (default) Uses 0.2 for surface source spaces (unless fixed is True) and 1.0 for other source spaces (volume or mixed).
    #depth = 0.8 (default) will applies moderate depth compensation
    #More info: https://mne.tools/stable/generated/mne.minimum_norm.make_inverse_operator.html 
    
    mne.minimum_norm.write_inverse_operator(inv_out, inv, overwrite=True)
    logging.info(f"[INVERSE] Saved to: {inv_out}")
    return inv

#  Source Time Courses (STCs)
def run_stcs(subject, data_dir, task):
    """
    Applies inverse solution to all epochs to estimate source-level activity.
    STC = Source Time Course (saved as .stc files per trial)
    """
    epo_fif = os.path.join(data_dir, f"{subject}_{task}_epo.fif")
    inv_fif = os.path.join(data_dir, f"{subject}_{task}_inv.fif")
    output_dir = os.path.join(data_dir, "source_outputs", "overt", "stcs")
    os.makedirs(output_dir, exist_ok=True)
    epochs = mne.read_epochs(epo_fif, preload=True)
    inv = mne.minimum_norm.read_inverse_operator(inv_fif)
    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=1. / 9., method='dSPM')
    for i, stc in enumerate(stcs):
        stc.save(os.path.join(output_dir, f"stc_{i:03d}"), overwrite=True)
    logging.info(f"[STCS] Saved {len(stcs)} STCs to: {output_dir}")

# === CLI ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MEG/EEG Preprocessing")
    parser.add_argument("--tasks", nargs="+", required=True, help="Task name (matches .ds folder name), e.g., overt, beginMatch")
    parser.add_argument("--subjects", nargs='+', required=True, help="Subject ID(s), e.g., sub-20284 sub-20285")
    parser.add_argument("--raw", action="store_true", help="Run raw conversion")
    parser.add_argument("--trans", action="store_true", help="Run transformation matrix")
    parser.add_argument("--epochs", action="store_true", help="Run epoch creation")
    parser.add_argument("--evoked", action="store_true", help="Run evoked computation")
    parser.add_argument("--fwd", action="store_true", help="Run forward model")
    parser.add_argument("--inv", action="store_true", help="Run inverse operator")
    parser.add_argument("--stc", action="store_true", help="Run source estimates")
    args = parser.parse_args()

    for subject in args.subjects:
        for task in args.tasks:
            # base_dir = os.path.join("subject", subject) # Default Path
            base_dir = os.path.join("/Users/vant7e/Documents/RRI/rsa_analysis/subject", subject) #or you can change the BASE PATHWAY IN HERE
            os.makedirs(base_dir, exist_ok=True)

            log_dir = os.path.join(os.getcwd(), subject, "logs")  # <-- FIXED here
            os.makedirs(log_dir, exist_ok=True)

            log_path = os.path.join(log_dir, f"{subject}_{task}_log.txt")
            setup_logging(log_path)
            
            if not any([args.raw, args.trans, args.epochs, args.evoked, args.fwd, args.inv, args.stc]):
                logging.info("[INFO] No specific flags given. Running full pipeline...")
                run_raw(subject, base_dir, task)
                run_trans(subject, base_dir, task)
                run_epochs(subject, base_dir, task)
                run_evoked(subject, base_dir, task)
                run_forward(subject, base_dir, task)
                run_inverse(subject, base_dir, task)
                run_stcs(subject, base_dir, task)
            else:
                if args.raw: run_raw(subject, base_dir, task)
                if args.trans: run_trans(subject, base_dir, task)
                if args.epochs: run_epochs(subject, base_dir, task)
                if args.evoked: run_evoked(subject, base_dir, task)
                if args.fwd: run_forward(subject, base_dir, task)
                if args.inv: run_inverse(subject, base_dir, task)
                if args.stc: run_stcs(subject, base_dir, task)
            logging.info(f"[DONE] {subject} {task} done.")
            logging.info("[DONE] All requested steps completed.")
