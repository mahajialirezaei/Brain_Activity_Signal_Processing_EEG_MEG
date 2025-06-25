import os
import numpy as np
import mne
from scipy import signal
from scipy.signal import windows, spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from load_data import process_edf_file, getSubject, getRun, getbase_dir, getBands
from Plot_spectrograms import plot_spectrogram_and_dominant
from extract_band_specific_power import frame_signal, compute_psd_dft, band_power_from_psd
from load_data import getSubject, getRun, getbase_dir, getBands

base_dir = getbase_dir()
subjects = getSubject()
runs = getRun()
BANDS = getBands()

def extract_epoch_features(res, ch_idx=0, epoch_sec=1.0):
    fs = res['fs']
    data = res['filtered_data'][ch_idx]
    frames = frame_signal(data, fs, epoch_sec, overlap=0.0)
    freqs, psd = compute_psd_dft(frames, fs)

    alpha_pow = band_power_from_psd(freqs, psd, BANDS['alpha'])
    beta_pow  = band_power_from_psd(freqs, psd, BANDS['beta'])
    return np.vstack([alpha_pow, beta_pow]).T

def load_labels(edf_path):
    evt = edf_path + '.event'
    with open(evt, 'r') as f:
        return [int(line.strip()) for line in f if line.strip() in ('0','1')]

def collect_all_data():
    X_list, y_list = [], []
    last_res = None
    last_subj = None
    last_run = None
    for subj in subjects:
        for run in runs:
            edf_path = os.path.join(base_dir, subj, f"{subj}{run}.edf")
            if not os.path.exists(edf_path):
                continue

            res = process_edf_file(edf_path)
            if res is None:
                continue

            labels = load_labels(edf_path)
            feat_C3 = extract_epoch_features(res, ch_idx=0)
            feat_C4 = extract_epoch_features(res, ch_idx=1)
            feats = np.hstack([feat_C3, feat_C4])
            n_epochs = feats.shape[0]

            if len(labels) >= n_epochs:
                X_list.append(feats)
                y_list.append(labels[:n_epochs])
                last_res = res
                last_subj = subj
                last_run = run

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y, last_res, last_subj, last_run


def train_and_get_feedback(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = LogisticRegression(max_iter=500).fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"BCI classification accuracy: {acc:.2%}")
    return clf,X_train, X_test, y_train, y_test
