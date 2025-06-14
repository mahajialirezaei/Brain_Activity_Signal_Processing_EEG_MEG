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

            b, a = load_data.process_edf_file(edf_file)['filter_coeffs'].values()
            ls.append((b, a))

    return ls


