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
    """
    برای یک کانال مشخص،:
      - به اپوک‌های 1 s تقسیم می‌کند،
      - PSD را استخراج می‌کند،
      - توان α و β را برمی‌گرداند.
    """
    fs = res['fs']
    data = res['filtered_data'][ch_idx]      # فیلترشده کانال ch_idx
    frames = frame_signal(data, fs, epoch_sec, overlap=0.0)
    freqs, psd = compute_psd_dft(frames, fs)

    # α و β دو باندی که برای BCI معمولاً کافی‌اند:
    alpha_pow = band_power_from_psd(freqs, psd, BANDS['alpha'])
    beta_pow  = band_power_from_psd(freqs, psd, BANDS['beta'])
    # برمی‌گرداند ماتریس shape=(n_epochs, 2)
    return np.vstack([alpha_pow, beta_pow]).T