import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows

import load_data
from load_data import process_edf_file, getSubject, getRun, getbase_dir, getBands

subjects = getSubject()
runs = getRun()
base_dir = getbase_dir()
BANDS = getBands()


def frame_signal(data, fs, frame_sec=2.0, overlap=0.5):
    frame_len = int(frame_sec * fs)
    hop = int(frame_len * (1 - overlap))
    frames = []
    for start in range(0, len(data) - frame_len + 1, hop):
        frames.append(data[start:start + frame_len])
    if not frames:
        return np.empty((0, frame_len))
    return np.vstack(frames)


def compute_psd_dft(frames, fs):
    if frames.size == 0:
        return np.array([]), np.empty((0, 0))

    n_frames, N = frames.shape
    psd = np.zeros((n_frames, N // 2 + 1))
    for i, frame in enumerate(frames):
        windowed = frame * windows.hann(N)
        X = np.fft.rfft(windowed)
        psd[i] = (1 / (fs * N)) * np.abs(X) ** 2
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    return freqs, psd


def band_power_from_psd(freqs, psd, band):
    if freqs.size == 0 or psd.size == 0:
        return np.array([])
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[:, idx], freqs[idx], axis=1)


def calculate_band_powers():
    for subject in subjects:
        subject_dir = os.path.join(base_dir, subject)
        for run in runs:
            edf_file = os.path.join(subject_dir, f"{subject}{run}.edf")

            result = process_edf_file(edf_file)

            filtered_data = result['filtered_data']
            if not isinstance(filtered_data, np.ndarray):
                print(f"filtered_data not array for {subject} {run}, skipping.")
                continue
            if filtered_data.ndim == 1:
                filtered_data = filtered_data[np.newaxis, :]

            channel_names = result.get('channel_names')
            if len(channel_names) != filtered_data.shape[0]:
                channel_names = [f'ch{i}' for i in range(filtered_data.shape[0])]

            fs = result['fs']
            for ch_idx, ch_name in enumerate(channel_names):
                signal_1d = filtered_data[ch_idx]
                frames = frame_signal(signal_1d, fs, frame_sec=2.0, overlap=0.5)
                if frames.shape[0] == 0:
                    print(f"No frames for {subject} {run} {ch_name}, skipping.")
                    continue

                freqs, psd = compute_psd_dft(frames, fs)

                for band_name, band_range in BANDS.items():
                    band_powers = band_power_from_psd(freqs, psd, band_range)
                    if band_powers.size == 0:
                        print(f"No PSD/band power for {band_name} on {subject} {run} {ch_name}")
                        continue
                    mean_power = np.mean(band_powers)
                    print(f"Subject {subject}, Run {run}, Channel {ch_name}, "
                          f"Band {band_name}: Mean Power = {mean_power:.3e}")
                    plot_results(freqs, psd, band_powers, band_range, subject, run, ch_name, band_name)

def plot_results(freqs, psd, band_powers, band_range, subject, run, ch_name, band_name):
    mean_psd = np.mean(psd, axis=0)
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, mean_psd)
    plt.xlim(0, 50)
    plt.title(f"Average PSD - {subject} {run} {ch_name}")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.tight_layout()
    plt.show()

    if band_powers.size > 0:
        plt.figure(figsize=(6, 2))
        plt.imshow(band_powers[np.newaxis, :], aspect='auto', cmap='viridis',
                   extent=[0, len(band_powers), band_range[0], band_range[1]])
        plt.colorbar(label=f'{band_name.capitalize()} Power')
        plt.title(f"{band_name.capitalize()} Power over Time - {subject} {run} {ch_name}")
        plt.xlabel('Frame Index')
        plt.ylabel(f'Freq Band ({band_range[0]}–{band_range[1]} Hz)')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    calculate_band_powers()
