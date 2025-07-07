import os
import numpy as np
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, cross_val_score
import joblib
import mne

from load_data import process_edf_file, getSubject, getRun, getbase_dir, getBands

imagery_runs = ['R04', 'R08', 'R12']
target_channels = ['C3..', 'C4..']
window_sec = 2.0

subjects = getSubject()
runs = getRun()
base_dir = getbase_dir()
bands = getBands()


def extract_motor_imagery_features():
    X, y = [], []
    for subject in subjects:
        subj_dir = os.path.join(base_dir, subject)
        for run in runs:
            if run not in imagery_runs:
                continue
            edf_path = os.path.join(subj_dir, f"{subject}{run}.edf")
            if not os.path.exists(edf_path):
                continue
            # Load processed data
            res = process_edf_file(edf_path)
            if res is None:
                continue

            fs = res['fs']
            data = res['raw_data']
            ch_names = res['channel_names']
            events = res['events']
            event_id = res['event_id']

            if len(events) == 0:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                events, event_id = mne.events_from_annotations(raw, event_id=None)

            code_left = event_id.get('T1', event_id.get('1'))
            code_right = event_id.get('T2', event_id.get('2'))
            win_len = int(window_sec * fs)

            for _, onset, code in events:
                if code not in (code_left, code_right):
                    continue
                label = 0 if code == code_left else 1
                start = onset
                end = onset + win_len
                if end > data.shape[1]:
                    continue
                seg = data[:, start:end]
                feats = []
                for ch in target_channels:
                    idx = ch_names.index(ch)
                    sig = seg[idx]
                    for band in bands.values():
                        f, Pxx = signal.welch(sig, fs, nperseg=256, noverlap=128)
                        idxb = np.where((f >= band[0]) & (f <= band[1]))[0]
                        p = np.trapezoid(Pxx[idxb], f[idxb]) * 1e6
                        feats.append(p)
                X.append(feats)
                y.append(label)
    return np.array(X), np.array(y)


def train_lda(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LinearDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    print(f"Test accuracy: {clf.score(X_test, y_test):.2f}")
    cv = cross_val_score(clf, X, y, cv=5)
    print(f"5-fold CV: {cv.mean():.2f} ± {cv.std():.2f}")
    joblib.dump(clf, 'mi_lda_physionet.pkl')
    return clf


def simulate_online(clf):
    print("Simulated online motor imagery classification:")
    subject = subjects[0]
    run = imagery_runs[0]
    edf_path = os.path.join(base_dir, subject, f"{subject}{run}.edf")
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    ch_names = raw.info['ch_names']
    fs = int(raw.info['sfreq'])

    win = int(window_sec * fs)
    hop = int(win * 0.5)
    for start in range(0, data.shape[1] - win + 1, hop):
        seg = data[:, start:start+win]
        feats = []
        for ch in target_channels:
            idx = ch_names.index(ch)
            sig = seg[idx]
            for band in bands.values():
                f, Pxx = signal.welch(sig, fs, nperseg=256, noverlap=128)
                idxb = np.where((f >= band[0]) & (f <= band[1]))[0]
                p = np.trapezoid(Pxx[idxb], f[idxb]) * 1e6
                feats.append(p)
        pred = clf.predict([feats])[0]
        action = 'LEFT' if pred == 0 else 'RIGHT'
        print(f"{start/fs:.2f}-{(start+win)/fs:.2f}s -> {action}")


def main():
    X, y = extract_motor_imagery_features()
    if X.size == 0:
        print("No motor imagery data found. Check your .event files or channels.")
        return
    clf = train_lda(X, y)
    simulate_online(clf)

if __name__ == '__main__':
    main()
