import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram



def plot_spectrogram_and_dominant(signal_1d, fs, frames, freqs, psd, band_ranges, subject, run, ch_name):
    f_sg, t_sg, Sxx = spectrogram(signal_1d, fs=fs, window='hann',
                                  nperseg=int(1.0*fs), noverlap=int(0.5*fs),
                                  scaling='density', mode='psd')
    plt.figure(figsize=(10,4))
    plt.pcolormesh(t_sg, f_sg, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylim(0, 50)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram - {subject} {run} {ch_name}')
    plt.colorbar(label='PSD (dB/Hz)')
    plt.tight_layout()
    plt.show()


    rhythm_idx = []
    for i in range(psd.shape[0]):
        pows = []
        for band in band_ranges.values():

            pows.append(np.trapezoid(psd[i, (freqs>=band[0]) & (freqs<=band[1])],
                                 freqs[(freqs>=band[0]) & (freqs<=band[1])]))
        rhythm_idx.append(int(np.argmax(pows)))

    rhythm_idx = np.array(rhythm_idx)
    times = np.linspace(0, len(signal_1d)/fs, len(rhythm_idx))

    plt.figure(figsize=(10,2))
    cmap = plt.get_cmap('tab10')
    plt.scatter(times, rhythm_idx, c=rhythm_idx, cmap=cmap, s=10)
    plt.yticks([0,1,2], list(band_ranges.keys()))
    plt.ylim(-0.5, 2.5)
    plt.xlabel('Time [sec]')
    plt.title(f'Dominant Rhythm over Time - {subject} {run} {ch_name}')
    plt.tight_layout()
    plt.show()
