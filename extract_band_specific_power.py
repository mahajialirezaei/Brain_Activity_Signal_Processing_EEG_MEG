import numpy as np


def frame_signal(data, fs, frame_sec=2.0, overlap=0.5):
    frame_len = int(frame_sec * fs)
    hop = int(frame_len * (1 - overlap))
    frames = []
    for start in range(0, len(data) - frame_len + 1, hop):
        frames.append(data[start:start+frame_len])
    return np.array(frames)
