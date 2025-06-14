import mne
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

base_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'files')
subjects = ['S001', 'S002', 'S003', 'S004']
runs = [f'R{i:02d}' for i in range(1, 15)]


def band_power(param, fs, param1):
    pass


def process_edf_file(file_path):
    try:

        raw = mne.io.read_raw_edf(file_path, preload=True)

        print("\n" + "=" * 50)
        print(f": {file_path}")
        print(raw.info)

        fs = int(raw.info['sfreq'])
        channel_names = raw.info['ch_names']
        data, times = raw[:, :]

        raw.filter(8, 12, method='iir', verbose=False)
        filtered_data, _ = raw[:, :]

        alpha_powers = []
        for i, ch in enumerate(channel_names):
            alpha_power = band_power(filtered_data[i], fs, [8, 12])
            alpha_powers.append(alpha_power)
            print(f"power of alpha in channel {ch}: {alpha_power:.2f}")

        plt.figure(figsize=(12, 6))
        plt.specgram(filtered_data[0], Fs=fs, cmap='viridis')
        plt.colorbar(label='Power (dB)')
        plt.title(f'Spectrogram for {os.path.basename(file_path)} - Channel {channel_names[0]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

        return alpha_powers

    except Exception as e:
        print(f"exception in process {file_path}: {str(e)}")
        return None

