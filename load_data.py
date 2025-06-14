import os
import mne
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd

base_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'files')
subjects = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008']
runs = [f'R{i:02d}' for i in range(1, 15)]

BANDS = {
    'alpha': (8, 12),
    'beta':  (13, 30),
    'gamma': (30, 45)
}


def band_power(data, sfreq, band):
    f, Pxx = signal.welch(data, sfreq, nperseg=1024)
    mask = (f >= band[0]) & (f <= band[1])
    return np.trapz(Pxx[mask], f[mask])


def process_edf_file(file_path, band_name='alpha'):

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    sfreq = raw.info['sfreq']
    ch_names = raw.info['ch_names']
    data, _ = raw.get_data(return_times=False)

    low, high = BANDS[band_name]
    order = 4
    b, a = signal.butter(order, [low/(sfreq/2), high/(sfreq/2)], btype='bandpass')


    filtered = np.array([signal.filtfilt(b, a, data[ch]) for ch in range(data.shape[0])])


    powers = {band: [] for band in BANDS}
    for name, band in BANDS.items():
        for ch_idx in range(filtered.shape[0]):
            p = band_power(filtered[ch_idx], sfreq, band)
            powers[name].append(p)

    assert any(powers['alpha']), f"Alpha power all zero for {file_path}"


    plt.figure(figsize=(10, 4))
    plt.specgram(filtered[0], Fs=sfreq)
    plt.title(f"Spectrogram: {os.path.basename(file_path)} ({ch_names[0]})")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
    plt.clf()

    return {
        'file': file_path,
        'sfreq': sfreq,
        'filter_coeffs': {'b': b, 'a': a},
        'powers': powers
    }


def process_all_files():
    summary = []
    for subj in subjects:
        subj_dir = os.path.join(base_dir, subj)
        for run in runs:
            edf_file = os.path.join(subj_dir, f"{subj}{run}.edf")
            if not os.path.exists(edf_file):
                continue
            res = process_edf_file(edf_file)
            summary.append({
                'file': res['file'],
                'alpha_mean': np.mean(res['powers']['alpha']),
                'beta_mean':  np.mean(res['powers']['beta']),
                'gamma_mean': np.mean(res['powers']['gamma'])
            })

    all_alpha = [item['alpha_mean'] for item in summary]
    print(f"Overall average alpha power: {np.mean(all_alpha):.2f}")


    df = pd.DataFrame(summary)
    df.to_csv('eeg_band_powers.csv', index=False)
    print("Saved summary to eeg_band_powers.csv")


if __name__ == '__main__':
    process_all_files()
