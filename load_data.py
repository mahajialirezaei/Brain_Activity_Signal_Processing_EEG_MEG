import mne
import os
from glob import glob
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np

base_dir = os.path.join(os.path.dirname(__file__), 'dataset', 'files')
subjects = ['S001', 'S002', 'S003', 'S004']
runs = [f'R{i:02d}' for i in range(1, 15)]

