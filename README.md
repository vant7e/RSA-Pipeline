# MNE-Python Preprocessing Pipeline

This repository contains a modular, CLI-accessible preprocessing pipeline for MEG/EEG data using MNE-Python. The pipeline handles key steps from raw data conversion to source reconstruction and logging.

## Directory Structure

rsa_analysis/
│
├── Preprocessing_Pipeline.py     # Main preprocessing script
├── requirements.txt              # Pip dependencies
├── environment.yml               # Full conda environment
├── outputs/                      # Generated outputs (e.g., RDMs, projections)
├── subject/                      # Participant-specific MEG/EEG data and trans files
    └── logs/                     # Automatically created log files for each session
    └── bem/
