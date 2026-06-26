# 🧠 Brain Activity Signal Processing (EEG/MEG)

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive Python toolbox for processing EEG/MEG recordings, extracting band‑specific power features, designing digital filters, visualizing spectrograms, and building a motor‑imagery **Brain–Computer Interface (BCI)** application.

---

## 📂 Project Structure

```
Brain_Activity_Signal_Processing_EEG_MEG/
│
├── dataset/                           # (Raw data - not tracked by git)
│   └── files/
│       └── S001/
│           ├── S001R01.edf
│           └── S001R01.edf.event
│
├── src/                               # All source code
│   ├── __init__.py                    # Package initialization
│   ├── load_data.py                   # EDF loading, filtering, event parsing
│   ├── extract_band_specific_power.py # Framing & DFT-based PSD extraction
│   ├── design_filters.py              # Frequency response & pole-zero plots
│   ├── plot_spectrograms.py           # Spectrogram & dominant rhythm plots
│   └── application_bci.py             # Motor-imagery BCI (LDA classifier)
│
├── results/                           # (Optional) Saved models and plots
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## ⚙️ Installation & Dependencies

### 1. Clone the Repository
```bash
git clone https://github.com/mahajialirezaei/Brain_Activity_Signal_Processing_EEG_MEG.git
cd Brain_Activity_Signal_Processing_EEG_MEG
```

### 2. (Optional) Create a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Required Packages
```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`** (if you don't have it yet):
```
mne
numpy
scipy
matplotlib
scikit-learn
joblib
```

---

## 🚀 How to Run

> **Note**: All scripts are now inside the `src/` folder. Run them from the **project root** using the `python -m src.<module>` syntax, or directly with `python src/<file>.py`.

### 1. Process & Load Data (Basic Inspection)
```bash
python -m src.load_data
```
or
```bash
python src/load_data.py
```
- Reads the EDF files defined in `load_data.py`.
- Applies bandpass filters (α: 8–12Hz, β: 13–30Hz, γ: 30–45Hz).
- Computes absolute band powers using Welch's method.
- Displays a spectrogram for the first channel.

### 2. Extract Band-Specific Power Dynamics
```bash
python -m src.extract_band_specific_power
```
- Frames the signal into overlapping windows.
- Computes PSD per frame using DFT.
- Generates plots for average PSD and power-time series per band.

### 3. Visualize Filter Designs
```bash
python -m src.design_filters
```
- Plots **Magnitude/Phase response** and **Pole-Zero diagrams** for the α, β, and γ Butterworth bandpass filters.

### 4. Run the Motor-Imagery BCI Application
```bash
python -m src.application_bci
```
- Loads specific runs (`R04`, `R08`, `R12`) for motor imagery.
- Extracts band power features from channels `C3..` and `C4..`.
- Trains an **LDA (Linear Discriminant Analysis)** classifier.
- Prints test accuracy, 5-fold cross-validation scores.
- Displays an **ROC curve**.
- Simulates a real-time online classification (LEFT vs. RIGHT imagery) using a sliding window.

---

## ⚙️ Configuration

All configurable variables are centralized in `src/load_data.py` and `src/application_bci.py`:

| File | Variable | Description |
| :--- | :--- | :--- |
| `load_data.py` | `subjects` | List of subjects (e.g., `['S001', 'S002']`) |
| `load_data.py` | `runs` | List of runs (e.g., `['R01', 'R02', ...]`) |
| `load_data.py` | `BANDS` | Frequency bands (alpha, beta, gamma) |
| `application_bci.py` | `imagery_runs` | Runs used for motor imagery (default: `R04`, `R08`, `R12`) |
| `application_bci.py` | `target_channels` | Channels for feature extraction (default: `C3..`, `C4..`) |
| `application_bci.py` | `window_sec` | Window length in seconds for feature extraction |

> **💡 Path Handling**: After restructuring, `load_data.py` automatically resolves the dataset path using `os.path.dirname(os.path.dirname(__file__))`. It assumes the `dataset/` folder is located in the **project root**.

---

## 📖 Module Descriptions

### `src/load_data.py`
- **Core Functions**:
  - `process_edf_file(file_path)`: Reads EDF, applies filters, extracts events, computes band powers.
  - `getSubject()`, `getRun()`, `getbase_dir()`, `getBands()`: Utility getters.

### `src/extract_band_specific_power.py`
- **Core Logic**:
  - `frame_signal()`: Overlapping sliding windows.
  - `compute_psd_dft()`: Periodogram via FFT.
  - `band_power_from_psd()`: Integrates PSD over specific frequency bands.
  - Calls `plot_spectrogram_and_dominant()` from `plot_spectrograms.py`.

### `src/design_filters.py`
- **Core Logic**:
  - Retrieves filter coefficients from `load_data.py`.
  - Plots magnitude/phase response and pole-zero maps for each filter.

### `src/plot_spectrograms.py`
- **Core Function**:
  - `plot_spectrogram_and_dominant()`: Displays a spectrogram (0–50Hz) and a scatter plot of the dominant rhythm over time.

### `src/application_bci.py`
- **Core Functions**:
  - `extract_motor_imagery_features()`: Builds feature matrix (X) and labels (y) for imagery events.
  - `train_lda(X, y)`: Splits data, trains an LDA, evaluates with accuracy, CV, and ROC.
  - `simulate_online(clf)`: Simulates real-time decoding with overlapping windows.

---

## 🧪 Example Output (BCI Application)

After running `python -m src.application_bci`, you should see something like:

```
Test accuracy: 0.78
5-fold CV: 0.75 ± 0.04
Simulated online motor imagery classification:
0.00-2.00s -> LEFT
2.00-4.00s -> RIGHT
...
```

An ROC curve plot will also appear, showing the AUC score.

---

## 📚 References

- [MNE-Python Documentation](https://mne.tools/stable/index.html)
- Welch's method for Power Spectral Density estimation.
- Linear Discriminant Analysis for feature classification in BCI.
- Dataset: PhysioNet EEG Motor Imagery Dataset.

---

## 👨‍💻 Developer

Developed by **Mohammad Amin Haji Alirezaei**  
[![GitHub](https://img.shields.io/badge/GitHub-mahajialirezaei-181717?style=flat&logo=github)](https://github.com/mahajialirezaei)

Feel free to ⭐️ this repository, open an issue, or contribute!

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
