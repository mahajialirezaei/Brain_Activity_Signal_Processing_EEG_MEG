import os
import numpy as np
import mne
from scipy import signal
from scipy.signal import windows, spectrogram
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt