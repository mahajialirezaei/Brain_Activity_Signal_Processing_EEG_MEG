import numpy as np
from scipy.signal import windows

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
