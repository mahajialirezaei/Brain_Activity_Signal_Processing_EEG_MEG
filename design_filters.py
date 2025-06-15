import os

from scipy import signal

import load_data
from scipy.signal import freqz
import numpy as np
import matplotlib.pyplot as plt

subjects = load_data.getSubject()
runs = load_data.getRun()
base_dir = load_data.getbase_dir()


def determineHandW():
    ls = {'alpha': [], 'beta': [], 'gamma': []}
    for subject in subjects:
        subject_dir = os.path.join(base_dir, subject)

        for run in runs:
            edf_file = os.path.join(subject_dir, f"{subject}{run}.edf")

            if not os.path.exists(edf_file):
                continue

            result = load_data.process_edf_file(edf_file)
            if result is None:
                continue

            band_filter = {'alpha': result['filters'].get('alpha'), 'beta': result['filters'].get('beta'),
                           'gamma': result['filters'].get('gamma')}
            if band_filter:
                band_filter_alpha = band_filter['alpha']
                band_filter_beta = band_filter['beta']
                band_filter_gamma = band_filter['gamma']
                ls['alpha'].append((
                    band_filter_alpha['b'],
                    band_filter_alpha['a'],
                    result['fs'],
                    f"{subject}-{run}"
                ))
                ls['beta'].append((
                    band_filter_beta['b'],
                    band_filter_beta['a'],
                    result['fs'],
                    f"{subject}-{run}"
                ))
                ls['gamma'].append((
                    band_filter_gamma['b'],
                    band_filter_gamma['a'],
                    result['fs'],
                    f"{subject}-{run}"
                ))

    return ls


def design_z_transform():
    ls_coeffs = determineHandW()

    for band_type in ls_coeffs.keys():
        for b, a, fs, label in ls_coeffs[band_type]:
            w, h = freqz(b, a, worN=1024, fs=fs)
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(w, 20 * np.log10(np.abs(h)))
            plt.title(f'{band_type.capitalize()} Band - Magnitude\n{label}')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Gain [dB]')
            plt.grid(True)
            plt.ylim(-30, 5)

            plt.subplot(1, 2, 2)
            plt.plot(w, np.unwrap(np.angle(h)))
            plt.title(f'{band_type.capitalize()} Band - Phase\n{label}')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Phase [rad]')
            plt.grid(True)

            plt.tight_layout()
            plt.show()

            plot_pole_zero(b, a, f"{band_type} Band - {label}")


def plot_pole_zero(b, a, title):
    z, p, k = signal.tf2zpk(b, a)

    plt.figure(figsize=(6, 6))
    plt.scatter(np.real(z), np.imag(z), marker='o', label='Zeros')
    plt.scatter(np.real(p), np.imag(p), marker='x', label='Poles')

    theta = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'k--', linewidth=0.5)

    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title(f'Pole-Zero Plot\n{title}')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    design_z_transform()
