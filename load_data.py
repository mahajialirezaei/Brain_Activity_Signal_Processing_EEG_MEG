import mne
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

base_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'files')
subjects = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008']
runs = ['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08', 'R09', 'R10', 'R11', 'R12', 'R13', 'R14']


def getSubject():
    return subjects
def getRun():
    return runs
def getbase_dir():
    return base_dir

def band_power(data, sfreq, band):
    f, Pxx = signal.welch(data, sfreq, nperseg=1024)
    idx = np.where((f >= band[0]) & (f <= band[1]))[0]
    return np.trapz(Pxx[idx], f[idx])


def process_edf_file(file_path):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)

        print("\n" + "=" * 50)
        print(f": {file_path}")
        print(raw.info)

        fs = int(raw.info['sfreq'])
        channel_names = raw.info['ch_names']
        data, times = raw[:, :]

        b, a = signal.iirfilter(4, [8, 12], btype='bandpass', fs=fs, ftype='butter')

        filtered_data = signal.lfilter(b, a, data)

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

        return {
            'raw_data': data,
            'filtered_data': filtered_data,
            'fs': fs,
            'channel_names': channel_names,
            'filter_coeffs': {'b': b, 'a': a},
            'alpha_powers': alpha_powers
        }

    except Exception as e:
        print(f"exception in process {file_path}: {str(e)}")
        return None


def process_all_files():
    all_alpha_powers = []

    for subject in subjects:
        subject_dir = os.path.join(base_dir, subject)

        for run in runs:
            edf_file = os.path.join(subject_dir, f"{subject}{run}.edf")
            event_file = edf_file + '.event'

            alpha_powers = process_edf_file(edf_file)['alpha_powers']
            if alpha_powers:
                all_alpha_powers.append({
                    'subject': subject,
                    'run': run,
                    'alpha_powers': alpha_powers
                })

            if os.path.exists(event_file):
                with open(event_file, 'r', encoding='utf-8', errors='ignore') as f:
                    events = f.readlines()
                print(f"events {run}: {events}")

    if all_alpha_powers:
        avg_alpha = np.mean([np.mean(p['alpha_powers']) for p in all_alpha_powers])
        print(f"\n average power of alpha in each file : {avg_alpha:.2f}")



# process_all_files()