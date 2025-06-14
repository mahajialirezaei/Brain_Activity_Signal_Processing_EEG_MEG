import os
import load_data
from scipy.signal import freqz
import numpy as np
import matplotlib.pyplot as plt


subjects = load_data.getSubject()
runs = load_data.getRun()
base_dir = load_data.getbase_dir()


def determineHandW():
    ls = []
    for subject in subjects:
        subject_dir = os.path.join(base_dir, subject)

        for run in runs:
            edf_file = os.path.join(subject_dir, f"{subject}{run}.edf")

            called_func = load_data.process_edf_file(edf_file)
            b, a = called_func()['filter_coeffs'].values()
            fs = called_func()['fs']
            ls.append((b, a, fs))

    return ls


def design_z_transform():

    ls_coeffs = determineHandW()
    for b, a, fs in ls_coeffs:
        w, h = freqz(b, a, worN=1024, fs=fs)

        plt.figure(figsize=(6, 4))
        plt.plot(w, 20 * np.log10(np.abs(h)))
        plt.title('Magnitude Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Gain [dB]')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(6, 4))
        plt.plot(w, np.angle(h))
        plt.title('Phase Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [rad]')
        plt.grid(True)
        plt.tight_layout()
        plt.show()



design_z_transform()

