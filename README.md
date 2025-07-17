# Brain Activity Signal Processing (EEG/MEG)

This repository provides a collection of Python scripts for processing EEG/MEG recordings, extracting band‑specific power features, designing and visualizing digital filters, plotting spectrograms, and building a simple motor‑imagery BCI (Brain–Computer Interface) application.

---

## 📂 Repository Structure


.
├── dataset/
│   └── files/
│       ├── S001/
│       │   ├── S001R01.edf
│       │   └── S001R01.edf.event
│       ├── S002/ …
│       └── …
├── load\_data.py
├── extract\_band\_specific\_power.py
├── design\_filters.py
├── Plot\_spectrograms.py
└── application\_bci.py

- **dataset/files/**  
  Organize your raw EDF recordings and corresponding `.event` annotation files by subject (e.g. `S001/S001R01.edf`, `S001R01.edf.event`).

- **load_data.py**  
  - Reads EDF files using MNE, applies bandpass filters for α (8–12 Hz), β (13–30 Hz), γ (30–45 Hz).
  - Computes absolute band power via Welch’s method.
  - Plots a spectrogram for each file/channel.
  - Loads annotation events if a `.event` file exists.
  - Exposes:
    - `process_edf_file(...)` → returns raw data, filtered data, power per band, filters, event markers.
    - Utility getters: `getSubject()`, `getRun()`, `getbase_dir()`, `getBands()`.

- **extract_band_specific_power.py**  
  - Frames each channel’s filtered signal into overlapping windows.
  - Computes PSD via DFT and extracts band power per frame.
  - Calls `plot_spectrogram_and_dominant` to visualize spectrogram and dominant rhythms.
  - Plots average PSD and band‑power time series.

- **design_filters.py**  
  - Retrieves filters designed in `load_data.py` for each band/subject.
  - Plots frequency responses (magnitude & phase) and pole–zero diagrams for α, β, γ filters.

- **Plot_spectrograms.py**  
  - Defines `plot_spectrogram_and_dominant()`, used by `extract_band_specific_power.py`.
  - Displays:
    - Time–frequency spectrogram (0–50 Hz).
    - Scatter of dominant rhythm per frame.

- **application_bci.py**  
  - Implements a simple motor‑imagery BCI pipeline:
    1. **Feature extraction**: for runs labeled “R04”, “R08”, “R12”, extract α/β/γ band power from C3 and C4 channels around motor‑imagery events.
    2. **Classification**: trains an LDA classifier, reports test accuracy, cross‑validation scores, and plots ROC.
    3. **Online simulation**: applies the trained model in sliding windows to simulate real‑time classification (LEFT vs. RIGHT imagery).

---

## ⚙️ Installation & Dependencies

```bash
# Clone repo
git clone https://github.com/mahajialirezaei/Brain_Activity_Signal_Processing_EEG_MEG.git
cd Brain_Activity_Signal_Processing_EEG_MEG

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install mne numpy scipy matplotlib scikit-learn joblib
````

---

## 🚀 Usage

1. **Prepare your data**

   * Place your EDF files (e.g. `S001R01.edf`) and their annotation files (`S001R01.edf.event`) under `dataset/files/SXXX/`.

2. **Process & Inspect**

   ```bash
   python load_data.py
   ```

   * Computes band powers, plots spectrograms.

3. **Band‑Specific Analysis**

   ```bash
   python extract_band_specific_power.py
   ```

   * Frames signals, computes PSD, and visualizes band‑power dynamics.

4. **Filter Design Visualization**

   ```bash
   python design_filters.py
   ```

   * Plots magnitude/phase response and pole–zero maps for each band’s filter.

5. **Motor‑Imagery BCI**

   ```bash
   python application_bci.py
   ```

   * Extracts features from selected runs, trains an LDA, shows ROC curve, and simulates online decoding.

---

## 🔧 Configuration

* **Subjects & Runs**
  In `load_data.py`, adjust:

  ```python
  subjects = ['S001', 'S002', …]
  runs     = ['R01', 'R02', …]
  ```
* **Frequency Bands**
  Defined in `load_data.py` as:

  ```python
  BANDS = {
      'alpha': [ 8, 12],
      'beta' : [13, 30],
      'gamma': [30, 45]
  }
  ```
* **Motor‑Imagery Runs**
  In `application_bci.py`:

  ```python
  imagery_runs   = ['R04', 'R08', 'R12']
  target_channels = ['C3..', 'C4..']
  ```

---

## 📖 References

* [MNE-Python Documentation](https://mne.tools/stable/index.html)
* Welch’s method for power spectral density estimation
* Linear Discriminant Analysis for BCI feature classification

---

Feel free to adapt parameters (e.g., frame length, overlap, bands) or extend the scripts to other tasks (e.g., time‑frequency decomposition, different classifiers). Pull requests and issues are welcome!
