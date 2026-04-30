# SSVEP-BCI-Speller-Final-Year-Project

This repository contains the software developed for my third-year individual project: **An SSVEP-Based BCI Speller with Offline CCA and TRCA Evaluation**.

The project includes a Pygame-based visual stimulation interface, an 18-target offline evaluation protocol, and CCA/TRCA-based SSVEP frequency recognition scripts.

## Project overview

The system was designed as a proof-of-concept SSVEP-based BCI speller. The visual interface contains a 40-target keyboard-style layout, including letters, digits, and auxiliary text-entry keys. However, the dissertation evaluation was conducted on an 18-target subset rather than the full 40-target interface.

The offline evaluation compared:

- Canonical Correlation Analysis (CCA)
- Standard Task-Related Component Analysis (TRCA)

The evaluation used four EEG analysis windows:

- 1.0 s
- 2.0 s
- 3.0 s
- 5.0 s

## Main files

| File | Description |
|---|---|
| `main_gui_integrated_18target.py` | Main Pygame-based SSVEP keyboard interface and block-based experiment control. |
| `offline_eval.py` | Offline CCA/TRCA evaluation script used for the dissertation results. |
| `realtime_ssvep_recognition.py` | Real-time SSVEP recognition utilities prepared for future online integration. |
| `data_saver.py` | Data saving module for EEG recordings, trial data, and experiment metadata. |
| `eeg_device_interface.py` | EEG device interface module, including simulated EEG and device connection structure. |
| `sound_feedback.py` | Sound feedback module for experiment cues and recognition feedback. |

## Offline evaluation protocol

The reported dissertation results are based on the 18-target offline evaluation protocol.

Selected target sequence:

```text
13579qetuoadgjlzcb
