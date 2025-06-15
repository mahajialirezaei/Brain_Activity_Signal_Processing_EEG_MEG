import os

import numpy as np
from scipy.signal import windows

import load_data
from load_data import process_edf_file, getSubject, base_dir, getbase_dir, getBands, getRun

subjects = getSubject()
base_dir = getbase_dir()
BANDS = getBands()
runs = getRun()


def frame_signal(data, fs, frame_sec=2.0, overlap=0.5):
    frame_len = int(frame_sec * fs)

    hop = int(frame_len * (1 - overlap))
    frames = []
    for start in range(0, len(data) - frame_len + 1, hop):
        frames.append(data[start:start+frame_len])
    return np.array(frames)


def compute_psd_dft(frames, fs):
    n_frames, N = frames.shape
    psd = np.zeros((n_frames, N//2 + 1))
    for i, frame in enumerate(frames):
        X = np.fft.rfft(frame)
        psd[i] = (1/(fs * N)) * np.abs(X)**2
    freqs = np.fft.rfftfreq(N, d=1/fs)
    return freqs, psd


def band_power_from_psd(freqs, psd, band):
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[:, idx], freqs[idx], axis=1)


def calculate_mean():
    for subject in subjects:
        subject_dir = os.path.join(base_dir, subject)

        for run in runs:
            edf_file = os.path.join(subject_dir, f"{subject}{run}.edf")

            if not os.path.exists(edf_file):
                continue

            result = load_data.process_edf_file(edf_file)
            filtered_data = result['filtered_data']
            channels = result['channels']
            fs = result['fs']
            if result is None:
                continue
            for ch_idx in channels:
                data_channel = filtered_data[ch_idx]  # 1D array
                frames = frame_signal(data_channel, fs)
                freqs, psd = compute_psd_dft(frames, fs)
                alpha_powers = band_power_from_psd(freqs, psd, [8, 12])

                alpha_mean = np.mean(alpha_powers)
                print(f'Alpha mean power: {alpha_mean:.4e}')