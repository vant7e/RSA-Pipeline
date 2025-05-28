# MNE-Python Preprocessing Pipeline

## Directory Structure

```
rsa_analysis/
│
├── Preprocessing_Pipeline.py       # Main preprocessing script
├── requirements.txt                # Pip dependencies
├── environment.yml                 # Full Conda environment
├── outputs/                        # (Optional) Generated analysis outputs (e.g., RDMs, surface plots)
├── subject/                        # Contains subject-specific data and outputs
│   ├── sub-20284/
│   │   ├── PIC_oneset_overt.ds/               # Raw CTF data
│   │   ├── pic_onset_beginMatch.ds            # Raw CTF data
│   │   ├── sub-20284_overt_raw.fif
│   │   ├── sub-20284_overt_trans.fif
│   │   ├── sub-20284_overt_epo.fif
│   │   ├── sub-20284_overt_ave.fif
│   │   ├── sub-20284_overt_fwd.fif
│   │   ├── sub-20284_overt_inv.fif
│   │   ├── logs/                   # Automatically generated logs for each step
│   │   └── source_outputs/
│   │       └── overt/
│   │           ├── noise-cov.fif
│   │           └── stcs/
│   │               └── stc_000.stc
│   └── bem/                        # BEM and source space files
│       ├── sub-20284-oct-6-src.fif # Required MRI file
│       └── sub-20284-bem-sol.fif   # Required MRI file
```

## Features

- Convert .ds (CTF) files to .fif
- Coregistration (automatic or manual with GUI)
- Epoch extraction from annotated events
- Evoked averaging
- Forward model creation
- Inverse operator computation
- Source estimate (STC) generation
- Logging to both console and file

## Requirements

Full list in requirements.txt or environment.yml.

## Setup

- Clone this repository 

```bash
git clone https://github.com/vant7e/RSA-Pipeline.git
cd rsa_analysis
```

- Ensure `SUBJECTS_DIR` is set properly to point to your Freesurfer subjects directory:

```bash 
export SUBJECTS_DIR=/path/to/your/subjects/folder
```

- Create environment via conda (recommended)

```bash
conda env create -f environment.yml
conda activate rsapipeline
```
- Or manually via pip

```bash
pip install -r requirements.txt
```

## Usage

Run Pipeline for One Subject & One or More Tasks:

```bash
python Preprocessing_Pipeline.py --subjects sub-20284 --tasks pic_onset_overt pic_onset_beginMatch --raw --trans --epochs --evoked --fwd --inv --stc
```

# Optional Processing Flags: 
Use these flags to control which steps of the pipeline are executed:

```
--raw   → Convert .ds CTF file to .fif format
--trans  → Compute MRI-to-head transformation (.trans.fif)
--epochs → Extract epochs from annotated events
--evoked → Compute averaged evoked response
--fwd   → Create forward solution (.fwd.fif)
--inv   → Compute inverse operator (.inv.fif)
--stc   → Apply inverse to generate source estimates (STCs)
```

Run Full Pipeline for Multiple Subjects & Tasks:

```bash
python Preprocessing_Pipeline.py --subjects sub-20284 sub-20285 --tasks pic_onset_overt pic_onset_beginMatch
```

This will:
- Run all processing steps
- For both sub-20284 and sub-20285
- For each task: overt, beginMatch


To launch the manual coregistration GUI:
```bash
python Preprocessing_Pipeline.py --subjects sub-20284 --tasks overt --trans_gui
```

## Notes

- Ensure `SUBJECTS_DIR` is set correctly (e.g., to your `freesurfer` directory).
- This pipeline assumes preprocessed anatomical data (e.g., `recon-all`) is already available.
- Multi-task support (e.g., `--tasks overt covert`) is enabled.
- All intermediate files will be overwritten if they already exist.

## Logging

All outputs are logged in real-time and saved to file:

```bash
subjects/sub-[20284]/logs/preprocessing_sub-20284.log
```

This includes:
- MNE internal info
- Pipeline status messages
- Errors and warnings

You can customize the logging format and verbosity in the `setup_logging()` function in `Preprocessing_Pipeline.py`.

