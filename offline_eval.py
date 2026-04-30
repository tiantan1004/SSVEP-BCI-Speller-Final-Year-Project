"""
offline_eval.py
===============

Offline CCA / TRCA evaluation for the SSVEP-based BCI speller framework
described in the dissertation.

Experimental protocol (matches main_gui_integrated_18target.py):
    - Visual interface: 40-target Pygame keyboard prototype
    - Offline evaluation subset (18 targets):
          EVAL_TARGET_SEQUENCE = "13579qetuoadgjlzcb"
    - 6 blocks per session, each block presents the 18 targets once.
        * Blocks 0..4  -> calibration  (5 blocks * 18 targets = 90 trials)
        * Block  5     -> held-out test (18 trials)
    - Sampling rate fs = 256 Hz
    - Selected channels: Oz, O1, O2, POz, PO3, PO4, PO7, PO8  (8 channels)
    - EEG analysis windows: 1.0 s, 2.0 s, 3.0 s, 5.0 s
    - Visual response delay (gaze-onset to cortical SSVEP onset): 0.13 s
    - Candidate classes for evaluation = the 18 selected targets
      (NOT the full 40), so ITR uses N = 18.

Outputs (written to --out):
    results_summary.csv     numeric summary table for the dissertation
    accuracy_plot.png       CCA vs TRCA accuracy across window lengths
    itr_plot.png            CCA vs TRCA ITR across window lengths
    confusion_cca_W.png     18x18 confusion matrix per window (optional)
    confusion_trca_W.png    18x18 confusion matrix per window (optional)

This script is intentionally self-contained: it does NOT touch the GUI.
Run it after a recording session is complete:

    python offline_eval.py --data-dir experiment_data --out offline_results

"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Protocol constants. Adjust here if the experiment configuration changes,
# but keep them in sync with main_gui_integrated_18target.py / BCISettings.
# ---------------------------------------------------------------------------

EVAL_TARGET_SEQUENCE: str = "13579qetuoadgjlzcb"   # 18 unique characters

DEFAULT_FS: int = 256
DEFAULT_CALIBRATION_BLOCKS: int = 5
DEFAULT_TEST_BLOCKS: int = 1

# Posterior channels selected in Section 3.2.3 of the dissertation.
# Order must match how channels are stored in eeg_data; the loader will
# attempt to use a saved channel-name list if available, else assume this
# order with all channels included.
DEFAULT_CHANNEL_NAMES: List[str] = ["Oz", "O1", "O2", "POz",
                                   "PO3", "PO4", "PO7", "PO8"]

DEFAULT_WINDOW_LENGTHS: List[float] = [1.0, 2.0, 3.0, 5.0]
DEFAULT_VISUAL_DELAY_S: float = 0.13       # cortical SSVEP onset latency
DEFAULT_HARMONICS: int = 3                 # CCA reference harmonics
DEFAULT_ITR_GAP_S: float = 0.5             # gaze-shift / inter-trial gap
DEFAULT_BANDPASS: Tuple[float, float] = (6.0, 50.0)
DEFAULT_NOTCH_HZ: float = 50.0


# ---------------------------------------------------------------------------
# Frequency assignment for the 40-target keyboard.
# Mirrors main_gui_integrated_18target.RealTimeBCISystem._assign_frequencies:
# 8.0 Hz base, 0.2 Hz step, in row-major keyboard order.
# ---------------------------------------------------------------------------

_KEYBOARD_LAYOUT: List[List[str]] = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '<'],
    ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
    ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l'],
    ['z', 'x', 'c', 'v', 'b', 'n', 'm', '.'],
    ['_', ','],
]


def build_default_char_to_freq(base: float = 8.0, step: float = 0.2
                               ) -> Dict[str, float]:
    """Reproduce the GUI's char->frequency mapping when it is not stored."""
    mapping: Dict[str, float] = {}
    i = 0
    for row in _KEYBOARD_LAYOUT:
        for ch in row:
            mapping[ch] = round(base + i * step, 2)
            i += 1
    return mapping


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trial:
    """One presented trial (one cued character)."""
    block_idx: int
    trial_idx: int
    target_char: str
    target_freq: float           # Hz; resolved from saved data or default map
    block_role: str              # 'calibration' or 'test'
    eeg: np.ndarray              # shape (n_channels, n_samples), raw / unfiltered
    fs: int
    channel_names: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Loader
#
# The format used by DataSaver is not fully fixed in the project, so this
# loader is permissive: it walks --data-dir, picks up any .npz / .pkl /
# .pickle / .json+.npy combinations, and yields Trial objects.
#
# Supported per-trial layouts (any one of these is fine):
#
#   (A) one big file (.npz / .pkl) containing a 'trials' list, where each
#       element is a dict with keys:
#           eeg_data, target_char, block_idx, trial_idx, fs, n_channels,
#           target_frequency (optional), block_role (optional),
#           channel_names (optional), eval_target_sequence (optional)
#
#   (B) one file per trial named like trial_b{block}_t{trial}.{npz|pkl},
#       each containing the same keys as in (A).
#
# If `target_frequency` or `block_role` is missing from a trial, this
# script reconstructs them from `target_char` and `block_idx` using the
# defaults at the top of the file.
#
# >>> ADAPT THIS FUNCTION if your DataSaver writes a different layout. <<<
# ---------------------------------------------------------------------------

def _coerce_trial_dict(d: Dict[str, Any],
                       calibration_blocks: int,
                       char_to_freq: Dict[str, float]) -> Optional[Trial]:
    """Convert a saved-dict into a Trial. Returns None if essential fields missing."""
    eeg = d.get('eeg_data', d.get('eeg'))
    if eeg is None:
        return None
    eeg = np.asarray(eeg, dtype=np.float64)
    if eeg.ndim != 2:
        return None
    # Some savers store as (samples, channels). We want (channels, samples).
    # Heuristic: if rows >> cols, transpose.
    if eeg.shape[0] > eeg.shape[1] and eeg.shape[1] <= 64:
        eeg = eeg.T

    target_char = d.get('target_char', '')
    if not target_char:
        return None

    block_idx = int(d.get('block_idx', -1))
    trial_idx = int(d.get('trial_idx', -1))

    block_role = d.get('block_role')
    if block_role is None:
        block_role = ('calibration' if block_idx < calibration_blocks
                      else 'test')

    target_freq = d.get('target_frequency', d.get('frequency'))
    if not target_freq:
        target_freq = char_to_freq.get(target_char, 0.0)
    target_freq = float(target_freq)

    fs = int(d.get('fs', DEFAULT_FS))
    channel_names = d.get('channel_names')

    return Trial(
        block_idx=block_idx,
        trial_idx=trial_idx,
        target_char=target_char,
        target_freq=target_freq,
        block_role=str(block_role),
        eeg=eeg,
        fs=fs,
        channel_names=list(channel_names) if channel_names is not None else None,
    )


def load_trials(data_dir: Path,
                calibration_blocks: int = DEFAULT_CALIBRATION_BLOCKS,
                char_to_freq: Optional[Dict[str, float]] = None,
                ) -> List[Trial]:
    """Walk data_dir and return a list of Trial objects."""
    if char_to_freq is None:
        char_to_freq = build_default_char_to_freq()

    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")

    trials: List[Trial] = []

    # Walk recursively so per-session subfolders work too.
    for path in sorted(data_dir.rglob('*')):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        try:
            if suffix == '.npz':
                with np.load(path, allow_pickle=True) as z:
                    keys = list(z.keys())
                    if 'trials' in keys:
                        trial_list = z['trials'].tolist()
                        if isinstance(trial_list, dict):
                            trial_list = trial_list.get('trials', [])
                        for d in trial_list:
                            t = _coerce_trial_dict(d, calibration_blocks, char_to_freq)
                            if t is not None:
                                trials.append(t)
                    else:
                        # Treat the whole npz as one trial dict
                        d = {k: z[k].item() if z[k].dtype == object and z[k].shape == ()
                             else np.asarray(z[k]) for k in keys}
                        # unwrap object scalars
                        d2 = {}
                        for k, v in d.items():
                            if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
                                d2[k] = v.item()
                            else:
                                d2[k] = v
                        t = _coerce_trial_dict(d2, calibration_blocks, char_to_freq)
                        if t is not None:
                            trials.append(t)
            elif suffix in ('.pkl', '.pickle'):
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                if isinstance(obj, dict) and 'trials' in obj:
                    for d in obj['trials']:
                        t = _coerce_trial_dict(d, calibration_blocks, char_to_freq)
                        if t is not None:
                            trials.append(t)
                elif isinstance(obj, list):
                    for d in obj:
                        if isinstance(d, dict):
                            t = _coerce_trial_dict(d, calibration_blocks, char_to_freq)
                            if t is not None:
                                trials.append(t)
                elif isinstance(obj, dict):
                    t = _coerce_trial_dict(obj, calibration_blocks, char_to_freq)
                    if t is not None:
                        trials.append(t)
            elif suffix == '.json':
                # Sidecar metadata file; trial EEG must come from a
                # neighbouring .npy. Convention assumed:
                #   trial_b{}_t{}.json  +  trial_b{}_t{}.npy
                meta = json.loads(path.read_text())
                npy = path.with_suffix('.npy')
                if npy.exists():
                    meta['eeg_data'] = np.load(npy)
                    t = _coerce_trial_dict(meta, calibration_blocks, char_to_freq)
                    if t is not None:
                        trials.append(t)
            # Plain .npy / .csv files alone don't carry enough metadata to
            # reconstruct a Trial; they must be paired with a sidecar JSON
            # or be inside one of the container formats above.
        except Exception as exc:
            print(f"[loader] skipping {path.name}: {exc}", file=sys.stderr)
            continue

    return trials


# ---------------------------------------------------------------------------
# Channel selection
# ---------------------------------------------------------------------------

def select_channels(eeg: np.ndarray,
                    saved_names: Optional[List[str]],
                    desired: List[str]) -> np.ndarray:
    """Return EEG with rows reordered/subset to match `desired` channel names.

    If saved_names is None we cannot resolve channel identity, so we
    keep all channels (assuming the GUI was already configured with the
    posterior set). The TRCA/CCA pipeline handles arbitrary channel
    counts gracefully.
    """
    if saved_names is None:
        return eeg
    name_to_idx = {n: i for i, n in enumerate(saved_names)}
    rows = [name_to_idx[n] for n in desired if n in name_to_idx]
    if not rows:
        return eeg
    return eeg[rows, :]


# ---------------------------------------------------------------------------
# Preprocessing  (bandpass + notch + per-channel mean removal)
# ---------------------------------------------------------------------------

def _make_butter_bandpass(low: float, high: float, fs: float, order: int = 4):
    from scipy.signal import butter
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype='band')


def _make_iir_notch(f0: float, fs: float, q: float = 30.0):
    from scipy.signal import iirnotch
    return iirnotch(f0 / (0.5 * fs), q)


def preprocess(eeg: np.ndarray,
               fs: int,
               bandpass: Tuple[float, float] = DEFAULT_BANDPASS,
               notch_hz: float = DEFAULT_NOTCH_HZ) -> np.ndarray:
    """Zero-phase bandpass + notch + DC removal per channel."""
    from scipy.signal import filtfilt
    b_bp, a_bp = _make_butter_bandpass(bandpass[0], bandpass[1], fs)
    b_n, a_n = _make_iir_notch(notch_hz, fs)
    out = eeg.astype(np.float64, copy=True)
    out -= out.mean(axis=1, keepdims=True)
    out = filtfilt(b_bp, a_bp, out, axis=1)
    out = filtfilt(b_n, a_n, out, axis=1)
    return out


def extract_window(eeg: np.ndarray, fs: int,
                   window_s: float, delay_s: float) -> Optional[np.ndarray]:
    """Extract a window of length window_s seconds starting at delay_s."""
    start = int(round(delay_s * fs))
    n_samp = int(round(window_s * fs))
    end = start + n_samp
    if end > eeg.shape[1]:
        return None
    return eeg[:, start:end]


# ---------------------------------------------------------------------------
# CCA
# ---------------------------------------------------------------------------

def _reference_signals(freq: float, fs: int, n_samples: int,
                       n_harmonics: int = DEFAULT_HARMONICS) -> np.ndarray:
    """Return reference signal Y_f of shape (2*n_harmonics, n_samples)."""
    t = np.arange(n_samples) / fs
    rows = []
    for h in range(1, n_harmonics + 1):
        rows.append(np.sin(2 * np.pi * h * freq * t))
        rows.append(np.cos(2 * np.pi * h * freq * t))
    return np.asarray(rows)


def _cca_max_corr(X: np.ndarray, Y: np.ndarray) -> float:
    """Return the largest canonical correlation between X (C×T) and Y (R×T)."""
    from sklearn.cross_decomposition import CCA
    n_comp = min(X.shape[0], Y.shape[0])
    cca = CCA(n_components=1, max_iter=500)
    try:
        Xc, Yc = cca.fit_transform(X.T, Y.T)
    except Exception:
        return 0.0
    if Xc.shape[1] == 0:
        return 0.0
    # Correlation of first canonical components
    a = Xc[:, 0]
    b = Yc[:, 0]
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a @ a) * (b @ b))
    return float(abs(a @ b) / denom) if denom > 0 else 0.0


def cca_predict(X: np.ndarray,
                candidate_freqs: List[float],
                fs: int,
                n_harmonics: int = DEFAULT_HARMONICS) -> Tuple[int, np.ndarray]:
    """Return (predicted index into candidate_freqs, vector of correlations)."""
    corrs = np.zeros(len(candidate_freqs))
    n_samples = X.shape[1]
    for i, f in enumerate(candidate_freqs):
        Y = _reference_signals(f, fs, n_samples, n_harmonics)
        corrs[i] = _cca_max_corr(X, Y)
    return int(np.argmax(corrs)), corrs


# ---------------------------------------------------------------------------
# TRCA  (standard, class-specific spatial filters; no ensemble, no FBCCA)
#
# For each class i with K_i calibration trials X_{i,k} (C x T):
#   S_i = (1/(K(K-1))) sum_{k != l} X_{i,k} X_{i,l}^T      (inter-trial cov)
#   Q_i = (1/K) sum_k X_{i,k} X_{i,k}^T                    (auto cov)
#   w_i = leading generalised eigenvector of (S_i, Q_i)
# Class template: T_i = mean_k X_{i,k}
# Test: for each class, score = corr( w_i^T X_test , w_i^T T_i )
# Predicted class = argmax_i score
# ---------------------------------------------------------------------------

def _trca_spatial_filter(trials: np.ndarray) -> np.ndarray:
    """trials: (K, C, T) -> spatial filter w of length C."""
    from scipy.linalg import eigh
    K, C, T = trials.shape
    if K < 2:
        raise ValueError("TRCA requires at least 2 trials per class.")
    # Center each trial
    centered = trials - trials.mean(axis=2, keepdims=True)
    # S = sum_{k != l} X_k X_l^T
    sum_X = centered.sum(axis=0)              # (C, T)
    S = sum_X @ sum_X.T                       # (C, C)
    # subtract diagonal terms (k == l)
    for k in range(K):
        S -= centered[k] @ centered[k].T
    # Q = sum_k X_k X_k^T
    Q = np.zeros((C, C))
    for k in range(K):
        Q += centered[k] @ centered[k].T
    # Regularise Q a hair to keep the eigenproblem well-conditioned for small K.
    Q += 1e-6 * np.trace(Q) / C * np.eye(C)
    # Generalised eig: solve S w = lambda Q w, take w with largest lambda.
    eigvals, eigvecs = eigh(S, Q)
    return eigvecs[:, -1]


@dataclass
class TRCAModel:
    candidate_freqs: List[float]
    target_chars: List[str]
    spatial_filters: np.ndarray   # (n_classes, C)
    templates: np.ndarray         # (n_classes, C, T)


def trca_train(class_to_trials: Dict[str, List[np.ndarray]],
               candidate_freqs: List[float],
               target_chars: List[str]) -> TRCAModel:
    """Train one spatial filter + one template per class.

    class_to_trials[char] is a list of (C, T) calibration epochs for that class.
    All epochs across all classes must share the same (C, T) shape.
    """
    # Determine common (C, T)
    first_char = target_chars[0]
    if first_char not in class_to_trials or not class_to_trials[first_char]:
        raise ValueError("No calibration trials for first target.")
    C, T = class_to_trials[first_char][0].shape

    n_classes = len(target_chars)
    filters = np.zeros((n_classes, C))
    templates = np.zeros((n_classes, C, T))

    for ci, ch in enumerate(target_chars):
        epochs = class_to_trials.get(ch, [])
        if len(epochs) < 2:
            raise ValueError(
                f"Class '{ch}' has only {len(epochs)} calibration trial(s); "
                f"standard TRCA requires at least 2 per class. "
                f"Check that 5 calibration blocks were recorded."
            )
        stack = np.stack([e[:, :T] for e in epochs], axis=0)   # (K, C, T)
        filters[ci] = _trca_spatial_filter(stack)
        templates[ci] = stack.mean(axis=0)

    return TRCAModel(
        candidate_freqs=list(candidate_freqs),
        target_chars=list(target_chars),
        spatial_filters=filters,
        templates=templates,
    )


def trca_predict(X: np.ndarray, model: TRCAModel) -> Tuple[int, np.ndarray]:
    """Score X against each class template using class-specific spatial filters."""
    n_classes = len(model.target_chars)
    scores = np.zeros(n_classes)
    T = model.templates.shape[2]
    Xt = X[:, :T] if X.shape[1] >= T else np.pad(X, ((0, 0), (0, T - X.shape[1])))
    for ci in range(n_classes):
        w = model.spatial_filters[ci]
        a = w @ Xt
        b = w @ model.templates[ci]
        a = a - a.mean(); b = b - b.mean()
        denom = np.sqrt((a @ a) * (b @ b))
        scores[ci] = (a @ b) / denom if denom > 0 else 0.0
    return int(np.argmax(scores)), scores


# ---------------------------------------------------------------------------
# ITR (Wolpaw 2002), N=18 by default
# ---------------------------------------------------------------------------

def itr_bits_per_min(accuracy: float, n_classes: int, t_per_sel_s: float) -> float:
    if accuracy <= 0 or accuracy > 1 or t_per_sel_s <= 0 or n_classes <= 1:
        return 0.0
    if accuracy == 1.0:
        bits = np.log2(n_classes)
    else:
        bits = (np.log2(n_classes)
                + accuracy * np.log2(accuracy)
                + (1 - accuracy) * np.log2((1 - accuracy) / (n_classes - 1)))
    return float(max(0.0, bits * 60.0 / t_per_sel_s))


# ---------------------------------------------------------------------------
# Main evaluation routine
# ---------------------------------------------------------------------------

def run_evaluation(data_dir: Path,
                   out_dir: Path,
                   eval_targets: str = EVAL_TARGET_SEQUENCE,
                   calibration_blocks: int = DEFAULT_CALIBRATION_BLOCKS,
                   window_lengths_s: List[float] = None,
                   visual_delay_s: float = DEFAULT_VISUAL_DELAY_S,
                   itr_gap_s: float = DEFAULT_ITR_GAP_S,
                   harmonics: int = DEFAULT_HARMONICS,
                   bandpass: Tuple[float, float] = DEFAULT_BANDPASS,
                   notch_hz: float = DEFAULT_NOTCH_HZ,
                   channels: List[str] = None,
                   make_confusion: bool = True,
                   ) -> None:
    if window_lengths_s is None:
        window_lengths_s = list(DEFAULT_WINDOW_LENGTHS)
    if channels is None:
        channels = list(DEFAULT_CHANNEL_NAMES)

    out_dir.mkdir(parents=True, exist_ok=True)

    char_to_freq = build_default_char_to_freq()
    target_chars = list(eval_targets)
    candidate_freqs = [char_to_freq[c] for c in target_chars]
    n_classes_eval = len(target_chars)

    # ---- 1. Load all trials -------------------------------------------------
    trials = load_trials(data_dir, calibration_blocks=calibration_blocks,
                         char_to_freq=char_to_freq)
    if not trials:
        raise RuntimeError(f"No trials found under {data_dir}.")

    # Restrict to the 18 evaluation targets (defensive: GUI should already
    # only have presented these, but a session may include extras).
    eval_set = set(target_chars)
    trials = [t for t in trials if t.target_char in eval_set]
    print(f"[load] {len(trials)} trials covering "
          f"{len(set(t.target_char for t in trials))} unique targets.")

    # Sanity-check sampling rate
    fs_set = {t.fs for t in trials}
    if len(fs_set) != 1:
        print(f"[warn] multiple sampling rates found: {fs_set}; "
              f"using the most common one.")
    fs = max(fs_set, key=lambda f: sum(1 for t in trials if t.fs == f))

    # Split calibration vs test
    calib_trials = [t for t in trials if t.block_role == 'calibration']
    test_trials  = [t for t in trials if t.block_role == 'test']
    print(f"[split] calibration trials: {len(calib_trials)}   "
          f"test trials: {len(test_trials)}")

    if not test_trials:
        raise RuntimeError("No test trials (block_role='test') found. "
                           "Check that block 5 was recorded.")

    # ---- 2. Per-window evaluation ------------------------------------------
    summary_rows: List[Dict[str, Any]] = []
    cca_predictions: Dict[float, List[Tuple[str, str]]] = {}
    trca_predictions: Dict[float, List[Tuple[str, str]]] = {}

    for win_s in window_lengths_s:
        print(f"\n=== Window {win_s:.1f} s ===")

        # Preprocess + epoch every trial once for this window length
        def _epoch(tr: Trial) -> Optional[np.ndarray]:
            x = select_channels(tr.eeg, tr.channel_names, channels)
            x = preprocess(x, fs, bandpass=bandpass, notch_hz=notch_hz)
            return extract_window(x, fs, win_s, visual_delay_s)

        # Build calibration set per class
        class_to_calib: Dict[str, List[np.ndarray]] = {c: [] for c in target_chars}
        skipped_calib = 0
        for tr in calib_trials:
            ep = _epoch(tr)
            if ep is None:
                skipped_calib += 1
                continue
            class_to_calib[tr.target_char].append(ep)

        # Test epochs
        test_epochs: List[Tuple[Trial, np.ndarray]] = []
        skipped_test = 0
        for tr in test_trials:
            ep = _epoch(tr)
            if ep is None:
                skipped_test += 1
                continue
            test_epochs.append((tr, ep))

        if skipped_calib or skipped_test:
            print(f"[warn] window {win_s}s: skipped {skipped_calib} calib + "
                  f"{skipped_test} test trials with insufficient EEG length.")

        # Train TRCA on calibration only
        trca_model = trca_train(class_to_calib,
                                candidate_freqs=candidate_freqs,
                                target_chars=target_chars)

        # Evaluate
        cca_correct = 0; trca_correct = 0
        per_window_cca: List[Tuple[str, str]] = []
        per_window_trca: List[Tuple[str, str]] = []
        for tr, ep in test_epochs:
            true_idx = target_chars.index(tr.target_char)

            cca_idx, _ = cca_predict(ep, candidate_freqs, fs, harmonics)
            cca_pred_char = target_chars[cca_idx]
            per_window_cca.append((tr.target_char, cca_pred_char))
            if cca_idx == true_idx:
                cca_correct += 1

            trca_idx, _ = trca_predict(ep, trca_model)
            trca_pred_char = target_chars[trca_idx]
            per_window_trca.append((tr.target_char, trca_pred_char))
            if trca_idx == true_idx:
                trca_correct += 1

        n_test = len(test_epochs)
        cca_acc = cca_correct / n_test if n_test else 0.0
        trca_acc = trca_correct / n_test if n_test else 0.0
        t_per_sel = win_s + itr_gap_s
        cca_itr = itr_bits_per_min(cca_acc, n_classes_eval, t_per_sel)
        trca_itr = itr_bits_per_min(trca_acc, n_classes_eval, t_per_sel)

        print(f"  CCA  : {cca_correct}/{n_test} = {cca_acc*100:5.1f}%   "
              f"ITR = {cca_itr:6.2f} bits/min")
        print(f"  TRCA : {trca_correct}/{n_test} = {trca_acc*100:5.1f}%   "
              f"ITR = {trca_itr:6.2f} bits/min")

        summary_rows.append({
            'window_s': win_s,
            'n_test': n_test,
            'cca_correct': cca_correct,
            'cca_accuracy_pct': round(cca_acc * 100, 2),
            'trca_correct': trca_correct,
            'trca_accuracy_pct': round(trca_acc * 100, 2),
            'cca_itr_bpm': round(cca_itr, 2),
            'trca_itr_bpm': round(trca_itr, 2),
            'n_classes': n_classes_eval,
            't_per_selection_s': t_per_sel,
        })
        cca_predictions[win_s] = per_window_cca
        trca_predictions[win_s] = per_window_trca

    # ---- 3. Save CSV summary -----------------------------------------------
    csv_path = out_dir / 'results_summary.csv'
    _write_csv(csv_path, summary_rows)
    print(f"\n[save] {csv_path}")

    # ---- 4. Plots ----------------------------------------------------------
    _plot_accuracy(summary_rows, out_dir / 'accuracy_plot.png')
    _plot_itr(summary_rows, out_dir / 'itr_plot.png')
    print(f"[save] {out_dir / 'accuracy_plot.png'}")
    print(f"[save] {out_dir / 'itr_plot.png'}")

    # ---- 5. Optional confusion matrices ------------------------------------
    if make_confusion:
        for win_s in window_lengths_s:
            _plot_confusion(cca_predictions[win_s], target_chars,
                            out_dir / f'confusion_cca_{win_s:.1f}s.png',
                            title=f'CCA confusion ({win_s:.1f} s)')
            _plot_confusion(trca_predictions[win_s], target_chars,
                            out_dir / f'confusion_trca_{win_s:.1f}s.png',
                            title=f'TRCA confusion ({win_s:.1f} s)')

    # ---- 6. Print dissertation-style table ---------------------------------
    print("\n" + "=" * 72)
    print("Dissertation-format summary  (N = {} eval targets)".format(n_classes_eval))
    print("=" * 72)
    print(f"{'Window':>8} | {'CCA correct/N':>13} | {'CCA acc':>7} | "
          f"{'TRCA correct/N':>14} | {'TRCA acc':>8} | {'CCA ITR':>9} | {'TRCA ITR':>9}")
    print("-" * 72)
    for r in summary_rows:
        print(f"{r['window_s']:>6.1f} s | "
              f"{r['cca_correct']}/{r['n_test']:<11} | "
              f"{r['cca_accuracy_pct']:>6.1f}% | "
              f"{r['trca_correct']}/{r['n_test']:<12} | "
              f"{r['trca_accuracy_pct']:>7.1f}% | "
              f"{r['cca_itr_bpm']:>6.2f}    | "
              f"{r['trca_itr_bpm']:>6.2f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    import csv
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_accuracy(rows: List[Dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = [r['window_s'] for r in rows]
    cca = [r['cca_accuracy_pct'] for r in rows]
    trca = [r['trca_accuracy_pct'] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, cca, 'o-', label='CCA')
    ax.plot(xs, trca, 's-', label='TRCA')
    ax.set_xlabel('EEG analysis window (s)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Offline accuracy vs window length (18-target evaluation)')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_itr(rows: List[Dict[str, Any]], path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = [r['window_s'] for r in rows]
    cca = [r['cca_itr_bpm'] for r in rows]
    trca = [r['trca_itr_bpm'] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, cca, 'o-', label='CCA')
    ax.plot(xs, trca, 's-', label='TRCA')
    ax.set_xlabel('EEG analysis window (s)')
    ax.set_ylabel('ITR (bits/min)')
    ax.set_title('Offline ITR vs window length (N=18 classes)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_confusion(pairs: List[Tuple[str, str]],
                    labels: List[str], path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(labels)
    idx = {c: i for i, c in enumerate(labels)}
    M = np.zeros((n, n), dtype=int)
    for true_c, pred_c in pairs:
        if true_c in idx and pred_c in idx:
            M[idx[true_c], idx[pred_c]] += 1
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap='Blues')
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline CCA/TRCA evaluation for "
                                "the 18-target SSVEP speller.")
    p.add_argument('--data-dir', type=Path, default=Path('experiment_data'),
                   help='Folder with saved trial data (default: experiment_data)')
    p.add_argument('--out', type=Path, default=Path('offline_results'),
                   help='Output folder for CSV/plots (default: offline_results)')
    p.add_argument('--targets', type=str, default=EVAL_TARGET_SEQUENCE,
                   help=f'Eval target subset (default: {EVAL_TARGET_SEQUENCE})')
    p.add_argument('--calibration-blocks', type=int,
                   default=DEFAULT_CALIBRATION_BLOCKS,
                   help='Number of calibration blocks (default: 5)')
    p.add_argument('--windows', type=float, nargs='+',
                   default=DEFAULT_WINDOW_LENGTHS,
                   help='Window lengths in seconds (default: 1.0 2.0 3.0 5.0)')
    p.add_argument('--delay', type=float, default=DEFAULT_VISUAL_DELAY_S,
                   help='Visual response delay in seconds (default: 0.13)')
    p.add_argument('--harmonics', type=int, default=DEFAULT_HARMONICS,
                   help='CCA reference harmonics (default: 3)')
    p.add_argument('--itr-gap', type=float, default=DEFAULT_ITR_GAP_S,
                   help='Inter-trial gap added to selection time for ITR '
                        '(default: 0.5 s)')
    p.add_argument('--no-confusion', action='store_true',
                   help='Skip 18x18 confusion-matrix plots.')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_evaluation(
        data_dir=args.data_dir,
        out_dir=args.out,
        eval_targets=args.targets,
        calibration_blocks=args.calibration_blocks,
        window_lengths_s=args.windows,
        visual_delay_s=args.delay,
        harmonics=args.harmonics,
        itr_gap_s=args.itr_gap,
        make_confusion=not args.no_confusion,
    )


if __name__ == '__main__':
    main()
