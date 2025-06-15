import mne
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

base_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'files')
subjects = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008']
runs = ['R01', 'R02', 'R03', 'R04', 'R05', 'R06', 'R07', 'R08',
        'R09', 'R10', 'R11', 'R12', 'R13', 'R14']
BANDS = {
    'alpha': [8, 12],
    'beta': [13, 30],
    'gamma': [30, 45]
}

def getSubject():
    return subjects

def getRun():
    return runs

def getbase_dir():
    return base_dir

def getBands():
    return BANDS

def band_power(data, sfreq, band):
    f, Pxx = signal.welch(data, sfreq, nperseg=256, noverlap=128, scaling='density')
    idx = np.where((f >= band[0]) & (f <= band[1]))[0]
    power = np.trapz(Pxx[idx], f[idx])
    return power * 1e6

def process_edf_file(file_path):
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True)
        fs = int(raw.info['sfreq'])
        channel_names = raw.info['ch_names']
        data, times = raw[:, :]

        filters = {}
        for band_name, band_range in BANDS.items():
            b, a = signal.iirfilter(4, band_range, btype='bandpass', fs=fs, ftype='butter')
            filters[band_name] = {'b': b, 'a': a}

        powers = {band: [] for band in BANDS}
        for ch_idx, ch_name in enumerate(channel_names):
            for band_name, band_range in BANDS.items():
                filtered = signal.filtfilt(
                    filters[band_name]['b'],
                    filters[band_name]['a'],
                    data[ch_idx]
                )
                power = band_power(filtered, fs, band_range)
                powers[band_name].append(power)
                print(f"Power of {band_name} in channel {ch_name}: {power}")

        plt.figure(figsize=(12, 6))
        plt.specgram(data[0], Fs=fs, cmap='viridis')
        plt.colorbar(label='Power (dB)')
        plt.title(f'Spectrogram for {os.path.basename(file_path)} - Channel {channel_names[0]}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.show()

        return {
            'fs': fs,
            'channel_names': channel_names,
            'powers': powers,
            'filtered_data': (raw[:,:], fs),
            'raw_data': data,
            'filters': filters
        }
    except Exception as e:
        print(f"Exception in processing {file_path}: {str(e)}")
        return None

def process_all_files():
    all_results = []

    for subject in subjects:
        subject_dir = os.path.join(base_dir, subject)

        for run in runs:
            edf_file = os.path.join(subject_dir, f"{subject}{run}.edf")
            event_file = edf_file + '.event'

            if not os.path.exists(edf_file):
                print(f"File not found: {edf_file}")
                continue

            result = process_edf_file(edf_file)
            if result is None:
                continue

            all_results.append({
                'subject': subject,
                'run': run,
                'powers': result['powers']
            })

            if os.path.exists(event_file):
                with open(event_file, 'r', encoding='utf-8', errors='ignore') as f:
                    events = f.readlines()
                print(f"Events for {run}: {events}")

    if all_results:
        avg_powers = {band: [] for band in BANDS}
        for res in all_results:
            for band in BANDS:
                avg_powers[band].append(np.mean(res['powers'][band]))

        for band, values in avg_powers.items():
            print(f"\nAverage {band} power across all files: {np.mean(values)}")

if __name__ == '__main__':
    process_all_files()