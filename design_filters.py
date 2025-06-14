import load_data
from scipy.signal import freqz
import numpy as np
import matplotlib.pyplot as plt


subjects = load_data.getSubject()
runs = load_data.getRun()
base_dir = load_data.getbase_dir()

