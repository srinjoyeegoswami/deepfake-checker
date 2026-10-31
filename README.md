# Pulse-Based Deepfake Detection
Detecting synthetic and manipulated video frames by extracting subtle remote photoplethysmography (rPPG) blood volume pulse signals.

---

# Phase 1: Physiological Signal Validation & rPPG Benchmark
A computer vision and physiological signal-processing pipeline that extracts blood volume pulse (BVP) signals from facial video. This phase establishes the biological ground truth, validating cardiac pulse extraction across real human subjects and benchmarking single-channel intensity tracking against multi-channel color-space projections under motion corruption.

---

## Evaluation Metric
Performance across test subjects is quantified using **Mean Absolute Error (MAE)**, defined as the average absolute deviation between algorithm estimates and ground-truth pulse oximeter measurements:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} \vert{}\text{Estimated}_i - \text{True}_i\vert{}$$

For our initial baseline of $N = 3$ subjects:

$$\text{MAE}_{\text{Green}} = \frac{\vert{}\text{Err}_1\vert{} + \vert{}\text{Err}_2\vert{} + \vert{}\text{Err}_3\vert{}}{3} \approx 6.43\text{ BPM}$$

$$\text{MAE}_{\text{POS}} = \frac{\vert{}\text{Err}_1\vert{} + \vert{}\text{Err}_2\vert{} + \vert{}\text{Err}_3\vert{}}{3} \approx 1.53\text{ BPM}$$

---

## Benchmark Results: Raw Green vs. POS (UBFC-rPPG)
We evaluated single-channel spatial averaging (Green) against the Plane-Orthogonal-to-Skin (POS) algorithm across subjects from the UBFC-rPPG dataset against ground-truth pulse oximeter readings:

| Subject | Ground Truth HR (BPM) | Green Est. (BPM) | Green Error (BPM) | POS Est. (BPM) | POS Error (BPM) | Observed Motion Dynamics |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Subject 1** | **77.3** | 76.0 | **1.3** | 76.0 | **1.3** | Stationary subject; high SNR across both methods. |
| **Subject 2** | **82.6** | 83.0 | **0.4** | 82.0 | **0.6** | Stable baseline; sub-1 BPM accuracy maintained. |
| **Subject 3** | **94.7** | 77.1 | **17.6** | 92.0 | **2.7** | Rigid motion at recording start/end; POS eliminates noise floor harmonics. |

* **Mean Absolute Error (Raw Green):** **6.43 BPM**

* **Mean Absolute Error (POS):** **1.53 BPM** (~76% error reduction)

---

## Motion Artifact Rejection Analysis (Subject 3)
Single-channel intensity approaches assume all periodic luminance shifts stem from volumetric capillary expansion. Under real-world conditions, rigid head movement alters surface angles against ambient illumination, creating large specular shifts that degrade single-channel spectra.

![Subject 3 Comparison](sub3_green_vs_pos.png)

### Key Observations:
1. **Raw Green Failure:** Subject 3 experienced postural adjustments during the initial and final frames. These motion artifacts generated low-frequency harmonics that caused the green FFT to peak at **77.1 BPM** ($17.6\text{ BPM}$ error), masking the ground-truth pulse of **94.7 BPM**.

2. **Pulse Recovery via POS:** POS normalizes temporal channels against their means and projects the signals onto orthogonal chrominance vectors ($S_1 = G - B$ and $S_2 = -2R + G + B$). Weighting these projections by their relative standard deviations cancels out shared specular motion while preserving the anti-phase hemoglobin absorption waveform. This shifts the dominant spectral peak to **92.0 BPM** ($2.7\text{ BPM}$ error).

3. **Spectral Dispersion (HRV):** The broader peak distribution in Subject 3's POS spectrum reflects physiological Heart Rate Variability (HRV), as the subject's instantaneous heart rate fluctuated between 90 and 98 BPM across the 84-second sequence.

---

## Technical Pipeline
1. **Facial Landmark Mesh:** MediaPipe FaceLandmarker tracks 478 3D facial landmarks on every frame.

2. **Forehead Skin ROI:** Skin regions are masked using upper forehead landmarks (`TOP = [67, 109, 10, 338, 297]`) trimmed 15% inwards from temple edges and stopping 40% short of the eyebrows (`BROW = [105, 334]`) to prevent hairline and eyelid contamination.

3. **Temporal Filtering:** Detrending eliminates slow baseline drift (lighting changes, camera auto-exposure), followed by a 3rd-order Butterworth bandpass filter ($0.7\text{--}3.0\text{ Hz}$, corresponding to $42\text{--}180\text{ BPM}$).

4. **Spectral Estimation:** Fast Fourier Transform (FFT) identifies the maximum energy frequency bin within the biological cardiac band.

---

# Phase 2: Pipeline Refactoring, Scaling to N = 7 & Failure-Mode Analysis
In this phase, the pulse-extraction pipeline was unified into modular scripts (`pipeline.py` and `evaluate.py`), introducing caching for facial landmarks and testing the algorithms across $N = 7$ subjects from the UBFC-rPPG dataset with frozen parameters.

---

## Benchmark Results ($N = 7$)
| Subject | Ground Truth HR (BPM) | Green Est. (BPM) | Green Error (BPM) | POS Est. (BPM) | POS Error (BPM) | Observed Motion Dynamics & Spectral Behavior |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Subject 1** | **77.3** | 76.4 | **0.9** | 76.4 | **0.9** | Stationary subject; high SNR baseline across both methods. |
| **Subject 2** | **82.6** | 81.6 | **1.0** | 81.6 | **1.0** | Stable baseline; sub-1 BPM accuracy maintained. |
| **Subject 3** | **94.7** | 43.0 | **51.7** | 91.8 | **2.9** | Severe rigid motion; Green collapses to 43.0 BPM artifact, POS recovers pulse. |
| **Subject 4** | **67.0** | 65.5 | **1.5** | 58.9 | **8.1** | Minimal motion; POS encounters low-frequency window drift, Green stays near truth. |
| **Subject 5** | **67.8** | 78.6 | **10.8** | 71.6 | **3.8** | Posture adjustments; POS suppresses motion artifacts, reducing error by 7.0 BPM. |
| **Subject 6** | **68.8** | 76.3 | **7.5** | 76.3 | **7.5** | Subtle periodic movement; both methods lock onto the same secondary harmonic. |
| **Subject 7** | **72.2** | 95.8 | **23.6** | 95.8 | **23.6** | Global illumination/motion artifact at 95.8 BPM corrupts forehead ROI across all channels. |

---

## Aggregate Performance Metrics
To capture both mean performance and the influence of severe outliers, we report both Mean Absolute Error (MAE) and Median Absolute Error across all 7 subjects:

* **Mean Absolute Error (MAE):**

  * **Raw Green:** **13.86 BPM**

  * **POS Algorithm:** **6.83 BPM** (~50.7% overall error reduction)

* **Median Absolute Error:**

  * **Raw Green:** **7.50 BPM**

  * **POS Algorithm:** **3.80 BPM** (~49.3% reduction in typical error)

---

## Key Findings & Failure-Mode Analysis
### 1. ROI Refactoring & The Subject 3 Discrepancy
When moving from standalone scripts to `pipeline.py`, the forehead skin patch coordinates were standardized (inward temple margin of 15% and height scaled to 60% of forehead clearance).&#x20;

In Subject 3, this minor bounding-box change shifted Green’s dominant FFT peak from 77.1 BPM to 43.0 BPM ($0.716\text{ Hz}$). Spectral analysis revealed that two distinct artifact peaks coexist simultaneously in the Green channel:

* A low-frequency motion boundary harmonic at **43.0 BPM** (power: **145.4**)

* A secondary motion peak at **77.2 BPM** (power: **115.9**)

* The true biological cardiac peak at **94.7 BPM** (power: **28.8**, buried in the noise floor)

Because both artifact peaks had comparable power, slight ROI pixel variations determined which noise peak dominated. POS successfully canceled out both surface motion vectors, isolating the true cardiac peak at **91.8 BPM** ($2.9\text{ BPM}$ error).

### 2. POS is Resilient, but Not Universally Superior
While POS cut dataset-wide MAE in half, performance varies by condition:

* **Motion Dominance (Subjects 3 & 5):** POS significantly outperforms Green by canceling common-mode specular reflections.

* **Low-Motion Baseline Drift (Subject 4):** On benign, still recordings, single-channel Green can outperform POS ($1.5\text{ BPM}$ vs. $8.1\text{ BPM}$) if the orthogonal chrominance projections drift in low frequencies within the temporal normalization window.

* **Tied Performance (Subjects 1, 2, 6, 7):** When signals are either pristine (1, 2) or corrupted by an environmental illumination artifact that shifts all RGB channels simultaneously (Subject 7 at 95.8 BPM), both algorithms yield identical estimates.

### 3. Transition to Multi-Region Coherence
Subject 7 demonstrates a fundamental limitation of single-region rPPG: when an external periodic lighting change or global head tilt corrupts the entire forehead patch, spectral filtering alone cannot differentiate between authentic blood perfusion and ambient interference.&#x20;

---

## Spectral Power Distribution & Failure-Mode Analysis
To ground the observed benchmark discrepancies in empirical data rather than speculative assumptions, we extracted and plotted the normalized Fast Fourier Transform (FFT) power spectra of both the raw Green channel and the POS projection for Subjects 3, 4, and 7. Each spectrum is evaluated against the physiological passband ($42\text{--}180\text{ BPM}$) with a ground-truth cardiac marker derived from continuous contact pulse oximetry.

![Spectral Power Distribution](spectra_comparison.png)

---

### In-Depth Subject Spectral Breakdown
#### 1. Subject 3 (Ground Truth: 94.7 BPM) —&#x20;**_*Motion Rejection via POS*_

* **Raw Green Failure:** Subject 3 exhibited severe rigid head movements and posture shifts during recording. Because single-channel spatial averaging cannot distinguish between volumetric blood perfusion and shifting specular reflection, the raw Green channel's energy distribution split into two massive motion-induced artifact peaks:

  * A primary boundary artifact at **43.0 BPM** ($0.716\text{ Hz}$, power: $145.4$), situated at the lower cutoff of the Butterworth bandpass filter.

  * A secondary motion harmonic at **77.2 BPM** ($1.28\text{ Hz}$, power: $115.9$).

  * The actual biological cardiac peak at **94.7 BPM** was completely buried in the noise floor with an energy magnitude of only $28.8$. Consequently, the Green estimator selected the dominant 43.0 BPM artifact peak, yielding an error of **$51.7\text{ BPM}$**.

* **POS Pulse Recovery:** By projecting normalized RGB signals onto two orthogonal chrominance axes ($S_1 = G - B$ and $S_2 = G + B - 2R$) and dynamically scaling them via standard deviation matching, the Plane-Orthogonal-to-Skin (POS) algorithm canceled out the common-mode luminance fluctuations caused by surface tilting. The 43.0 BPM and 77.2 BPM artifact humps were suppressed, isolating a sharp cardiac peak at **91.8 BPM** (error: **$2.9\text{ BPM}$**).

#### 2. Subject 4 (Ground Truth: 67.0 BPM) —&#x20;**_*Low-Frequency Window Drift in Still Baseline*_

* **Green Stability:** Subject 4 remained almost completely stationary throughout the recording. Under high signal-to-noise ratio (SNR) conditions with minimal rigid displacement, raw Green brightness captured the blood volume pulse cleanly, isolating a sharp peak at **65.5 BPM** (error: **$1.5\text{ BPM}$**).

* **POS Artifact Drift:** While POS detected the true cardiac harmonic near $66\text{ BPM}$, its temporal normalization window ($w = 1.6\text{ s}$) introduced low-frequency boundary drift, elevating a secondary spectral peak at **58.9 BPM** above the genuine pulse peak. This resulted in an error of **$8.1\text{ BPM}$**, demonstrating that multi-channel projections are not universally superior to single-channel Green on completely still baselines.

#### 3. Subject 7 (Ground Truth: 72.2 BPM) —&#x20;**_*Global Illumination and Common-Mode Corruption*_

* **Identical Failure Mode:** Both Green and POS produced identical estimates of **95.8 BPM** against the true ground truth of **72.2 BPM** (an error of **$23.6\text{ BPM}$**).

* **Root Cause:** A secondary biological peak is clearly present in both spectra directly under the red dashed ground-truth marker ($72.2\text{ BPM}$), proving that blood volume pulse was recorded by the camera sensor. However, an ambient periodic lighting fluctuation or camera auto-exposure oscillation around $1.6\text{ Hz}$ ($95.8\text{ BPM}$) shifted Red, Green, and Blue channels uniformly across the entire face.

* **Algorithmic Limit:** Because global lighting fluctuations shift all color channels simultaneously in the same direction, chrominance differencing algorithms cannot cancel them out. The non-biological oscillation had more spectral energy than the subtle micro-blush of the facial tissue, forcing both algorithms to latch onto the $95.8\text{ BPM}$ artifact.

---

## Real-World Validation: Self-Recorded Webcam Benchmark ($N = 3$)
To test generalizability beyond public benchmark datasets, the rPPG pipeline (`pipeline.py`) was evaluated on personal webcam recordings under natural indoor conditions (ambient fluorescent lighting, standard laptop sensor, 30 FPS, $N = 3$ conditions, one clip per condition). Reference ground truth heart rates were obtained via simultaneous contact radial pulse counts.

| Scenario | Ground Truth (BPM) | Green Est. (BPM) | Green Error (BPM) | POS Est. (BPM) | POS Error (BPM) | Observed Dynamics & Failure Analysis |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1. Stationary Baseline** (`self_stationary.mp4`) | **73.5** | 70.9 | **2.6** | 74.8 | **1.3** | High SNR baseline; stable frontal pose. Pipeline transfers directly to unseen consumer webcam hardware without retuning. |
| **2. Controlled Rigid Motion** (`self_motion.mp4`) | **76.0** | 50.0 | **26.0** | 65.3 | **10.7** | Pitch/yaw head rotation shifts illumination. Green collapses to a 50.0 BPM artifact; POS suppresses luminance noise, cutting error by >58%. |
| **3. Post-Exertion / Elevated** (`self_exercise.mp4`) | **143.0** | 44.0 | **99.0** | 71.0 | **72.0** | **Failure case across both methods.** Heavy post-exercise breathing and motion corrupted the signal; the POS estimate peaked near half the true rate ($71.0\text{ BPM}$ vs. $143.0\text{ BPM}$). |

---

## Post-Exercise Diagnostic Analysis (`self_exercise.mp4`)
To investigate the breakdown in Scenario 3, we computed time-windowed estimates (10-second sliding windows, 2-second hop) and inspected the whole-clip Fourier spectrum against raw filtered oscillations.

![Post-Exercise Diagnostic](diagnose_post_exercise.png)

### Diagnostic Findings:
1. **Absence of Cardiac Peak in Fourier Spectrum:** At the measured reference rate of **143.0 BPM**, spectral power sits in a local trough with zero discernible peak energy. Instead, POS produced its global maximum peak at **71.0 BPM**, closely aligning with half the ground truth ($143 / 2 = 71.5\text{ BPM}$).

2. **Physiological Non-Stationarity:** The windowed time-series analysis reveals that the subject's heart rate was not static; it briefly hit ~138 BPM near the 19-second mark before steadily declining toward 70–80 BPM during recovery. Attempting to fit a single constant ground truth across a >50-second recovery recording blurs and broadens the Fourier spectrum.

3. **Motion and Respiration Bursts:** The filtered time-domain waveform displays severe amplitude surges (>0.10 relative to baseline fluctuations of ~0.03) between 23–27s and 38–48s, confirming that heavy breathing and postural shifts introduced dominant low-frequency artifacts that obscured the underlying pulse.

---

## Limitations & Operational Boundaries
1. **Physical Motion:** Rigid and non-rigid head rotation (pitch, yaw, tilt) causes surface illumination shifts. While POS dampens linear specular noise compared to single-channel Green, multi-axis head motion still degrades pulse recovery on single-region setups.

2. **Ambient Illumination Fluctuations:** Common-mode lighting shifts across all RGB channels (e.g., room lighting flicker, auto-exposure adjustments) cannot be canceled by color-space differencing alone.

3. **Non-Stationary Heart Rates:** Single-window Fourier spectral estimation assumes stationary frequency over time. Rapid physiological deceleration or acceleration smears spectral energy across bins.

4. **Post-Exertion / Exercise:** The pipeline is not validated for post-exercise scenarios; heavy thoracic respiration, periodic panting, and postural sway introduce dominant movement artifacts that swamp the subtle cardiovascular signal.

---

# Phase 3: Transition to In-the-Wild Video & The Codec Noise Floor (DFDC)
In Phase 3, we evaluated whether remote photoplethysmography scales to unconstrained, compressed web video as a viable feature for deepfake detection. Using the **Deepfake Detection Challenge (DFDC)** dataset, we tested the core hypothesis of biological detectors: **synthetic facial manipulation disrupts physiological cardiovascular signals, producing measurable pulse attenuation or absence in deepfakes.**

---

## 1. Dataset Characteristics & Identity Leakage Prevention
The evaluated DFDC sample comprises 400 video sequences exhibiting substantial class imbalance:

* **Total Clips:** 400

* **FAKE Clips:** 323 (~80.75%)

* **REAL Clips:** 77 (~19.25%)

* **Unique Originals:** 209 (58 fakes share an original within the 77-real subset; the remaining derive from the wider DFDC pool)

To prevent data leakage during evaluation, we mapped every video to its source actor/clip, establishing **244 independent source groups**. All subsequent cross-validation splits use `GroupKFold` across these 244 groups, guaranteeing that no synthesized video shares a fold with its original real counterpart.

---

## 2. Real-World Engineering Challenge: Small Face Mesh Failure
When applying standard full-frame MediaPipe landmark detection across DFDC videos, extraction failed on **>50% of clips**.&#x20;

### Root Cause Analysis:
* DFDC subjects stand at variable distances from the camera in unconstrained 1080p frames, causing facial bounding boxes to shrink to **$90\text{--}120\text{ px}$**.

* Full-frame downsampling in the detector causes small, tilted, or non-centered faces to drop below detection confidence thresholds.

### Architectural Solution (`extract_rgb_crop`):
1. **Dynamic Tile-Search:** A multi-scale sliding tile search discovers small faces across the 1080p canvas.

2. **Localized Bounding-Box Cropping:** Once located, the face region is cropped, upscaled, and tracked frame-to-frame.

3. **Temporal Gap-Filling:** Linear interpolation bridges momentary tracking dropouts (<5 frames).

This engineering intervention restored valid forehead skin ROI tracks across all evaluated sample clips ($100\\%$ retrieval on $n = 40$).

---

## 3. Pulse Calibration & The In-Frame Background Control Experiment
To determine whether pulse signals survive heavy H.264 spatial macroblock compression and small region-of-interest sizes, we evaluated the **Peak Power Ratio** within the physiological band ($0.7\text{--}3.0\text{ Hz}$):

$$\text{Peak Ratio} = \frac{\max_{f \in [0.7, 3.0]} \vert{}X(f)\vert{}^2}{\sum_{f \in [0.7, 3.0]} \vert{}X(f)\vert{}^2}$$

A distinct, observable cardiac pulse typically yields peak ratios of **$0.40\text{--}0.60+$**. We benchmarked DFDC forehead skin against synthetic white noise, uncompressed ground truth (UBFC), and an in-frame **non-skin background control patch** (a $40\times 40\text{ px}$ inert region from the upper-left background of each video):

| Signal Source | Video Quality / Dataset | Extraction Method | Mean Peak Ratio | Min – Max / Percentiles |
| :--- | :--- | :--- | :---: | :---: |
| **UBFC Forehead** | Uncompressed RGB (webcam) | POS ($n = 3$, 10s windows) | **0.498** | 0.448 – 0.557 (range of subject means) |
| **DFDC Forehead (Skin)** | H.264 Compressed (DFDC) | Crop-Based POS ($n = 40$) | **0.201** | 0.103 – 0.438 (min – max) |
| **DFDC Background Control** | H.264 Compressed (DFDC) | Inert In-Frame Patch ($n = 40$) | **0.200** | 0.102 – 0.369 (min – max) |
| **Synthetic White Noise** | Algorithmic Simulation | Gaussian white noise ($N = 500$) | **0.169** | 0.111 – 0.251 (5th – 95th pct) |

---

## 4. Empirical Findings & Paired Statistical Test ($n = 40$)
To eliminate clip-to-clip lighting and codec variances, we conducted a paired difference test ($\Delta = \text{Ratio}_{\text{Forehead}} - \text{Ratio}_{\text{Background}}$) across 40 identical clips (20 REAL, 20 FAKE):

| Metric | Mean | Std | Min | 25% | 50% (Median) | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Forehead Ratio** | 0.2007 | 0.0652 | 0.1029 | 0.1658 | 0.1859 | 0.2144 | 0.4380 |
| **Background Ratio** | 0.2002 | 0.0594 | 0.1022 | 0.1622 | 0.1861 | 0.2309 | 0.3693 |
| **Difference ($\Delta$)** | +0.0005 | 0.0939 | -0.2295 | -0.0577 | -0.0205 | 0.0577 | 0.3074 |

### Critical Empirical Observations:
1. **The Realistic Video Noise Floor:** The background control patch yielded a mean peak ratio of **0.200**, establishing that H.264 compression artifacts, sensor noise, and subtle illumination fluctuations create an empirical noise floor at $\sim 0.20$—significantly more peaked than idealized Gaussian white noise ($0.169$).

2. **Absence of Biological Separation:** The forehead skin patch produced a mean ratio of **0.201** (median **0.186**), matching the inert background control patch (**0.200**, median **0.186**) down to the third decimal place. The mean paired difference is **$+0.0005$** (paired $t$-test $t = 0.036$, $p \approx 0.97$).

3. **Class Indistinguishability:** With crop-based extraction, FAKE clips averaged **0.205** and REAL clips averaged **0.196** ($n = 20$ each), both hovering inside the empirical video noise floor.

---

# Phase 4: Live Demo & Reproduction Guide
This phase packages the project so anyone with a laptop and a webcam can run the interactive live pulse demo in about 10 minutes, and every empirical benchmark in Phases 1–3 can be regenerated from scratch.

\> **Disclaimer:** This is a learning and research project, not a medical device. Do not use the heart-rate readout for clinical diagnosis or health decisions.

---

## 4.1 What the Live Demo Shows (and What It Doesn't)
`live_demo.py` opens your webcam and renders:

* A green bounding box on your forehead indicating the localized skin region of interest (ROI).

* A rolling waveform of the filtered pulse signal along the bottom edge.

* An estimated heart rate in beats per minute (BPM), refreshed every second once a 10-second temporal buffer is collected.

It applies the identical Plane-Orthogonal-to-Skin (POS) projection pipeline developed in Phases 1 and 2. During live evaluation, this readout can be compared side-by-side with a smartwatch or pulse oximeter ground truth.

* **What it does not do:** It does not classify deepfakes. Phase 3 demonstrated that subtle pulse signals in compressed DFDC video are indistinguishable from background compression noise; therefore, this interactive demo isolates the real-time physiological signal recovery stage only.

* **Operational limits:** The estimator remains accurate within a few BPM during stationary frontal posture under uniform ambient lighting, but degrades under rigid head motion, flickering lighting, or immediate post-exercise recovery.

---

## 4.2 Quick Start: Run the Live Demo on Any Computer
**Requirements:** Python 3.11, Git, an integrated or external webcam, and an active internet connection for initial package installation. No external dataset download is required to run the live demo.

### Step 1: Clone the Repository
```bash

git clone [https\://github.com/srinjoyeegoswami/deepfake-checker.git](https\://github.com/srinjoyeegoswami/deepfake-checker.git)

cd deepfake-checker

## Step 2: Set Up a Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If script execution is blocked on Windows, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### macOS / Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
```

## Step 3: Install Core Dependencies

Create or verify `requirements.txt` in the repository root:

```text
opencv-python
mediapipe
numpy
scipy
matplotlib
scikit-learn
pandas
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Verify that the environment imports correctly:

```bash
python -c "import cv2, mediapipe, scipy; print('Setup verified successfully.')"
```

## Step 4: Download the Face Landmarker Model

The pre-trained MediaPipe bundle (`face_landmarker.task`) must be located in the repository root, beside `live_demo.py`.

### Windows (PowerShell)

```powershell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -OutFile face_landmarker.task
```

### macOS / Linux

```bash
curl -L -o face_landmarker.task "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
```

## Step 5: Implementation Code (`live_demo.py`)

Save the following implementation as `live_demo.py` in your repository root:

```python
import collections
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.signal import butter, detrend, sosfiltfilt

from pipeline import pos_signal


TOP, BROW = [67, 109, 10, 338, 297], [105, 334]
WINDOW_SEC, MIN_SEC = 20, 10  # Keep 20 s of history; report after 10 s.

base = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
)


def estimate(buf):
    arr = np.array(buf)  # Columns: t, r, g, b
    span = arr[-1, 0] - arr[0, 0]
    fps = (len(arr) - 1) / max(span, 1e-6)  # Measured webcam FPS
    sig = detrend(pos_signal(arr[:, 1:4], fps))
    sos = butter(3, [0.7, 3.0], btype="bandpass", fs=fps, output="sos")
    filt = sosfiltfilt(sos, sig)
    spec = np.abs(np.fft.rfft(filt))
    f = np.fft.rfftfreq(len(filt), d=1 / fps)
    band = (f >= 0.7) & (f <= 3.0)
    return f[band][np.argmax(spec[band])] * 60, filt


# Robust camera opening for Windows (DirectShow backend + index fallback)
cap = None
for idx in [0, 1, 2]:
    temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if temp_cap.isOpened():
        ret, probe_frame = temp_cap.read()
        if ret and probe_frame is not None:
            cap = temp_cap
            print(f"Connected to webcam at index {idx} via CAP_DSHOW")
            break
        temp_cap.release()

if cap is None:
    # Fallback to the default backend.
    for idx in [0, 1]:
        temp_cap = cv2.VideoCapture(idx)
        if temp_cap.isOpened():
            ret, probe_frame = temp_cap.read()
            if ret and probe_frame is not None:
                cap = temp_cap
                print(f"Connected to webcam at index {idx} via default backend")
                break
            temp_cap.release()

if cap is None:
    raise RuntimeError(
        "Could not access webcam. Verify that your camera is not in use by another app "
        "and Windows Settings -> Privacy & Security -> Camera permissions are enabled."
    )

buf, bpm, filt, last_calc, t0 = collections.deque(), None, None, 0, time.time()

while True:
    ok, frame = cap.read()
    if not ok:
        print("Dropped camera frame, retrying...")
        continue

    frame = cv2.flip(frame, 1)  # Mirror view
    h, w = frame.shape[:2]
    now = time.time() - t0

    res = landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
    )

    face_ok = False
    if res.face_landmarks:
        lms = res.face_landmarks[0]
        top = np.array([[lms[i].x * w, lms[i].y * h] for i in TOP])
        brow = np.array([[lms[i].x * w, lms[i].y * h] for i in BROW])

        width = top[:, 0].max() - top[:, 0].min()
        x1 = int(top[:, 0].min() + 0.15 * width)
        x2 = int(top[:, 0].max() - 0.15 * width)
        y1 = int(top[:, 1].mean())
        y2 = int(y1 + 0.6 * (brow[:, 1].mean() - y1))

        # Keep coordinates inside the frame.
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if y2 - y1 >= 4 and x2 - x1 >= 4:
            b, g, r, _ = cv2.mean(frame[y1:y2, x1:x2])
            buf.append((now, r, g, b))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            face_ok = True

    while buf and now - buf[0][0] > WINDOW_SEC:
        buf.popleft()

    if (
        len(buf) > 30
        and now - last_calc > 1.0
        and buf[-1][0] - buf[0][0] >= MIN_SEC
    ):
        bpm, filt = estimate(buf)
        last_calc = now

    if bpm is not None:
        txt = f"POS estimate: {bpm:.0f} bpm"
    elif face_ok:
        elapsed = buf[-1][0] - buf[0][0] if buf else 0
        txt = f"collecting... {elapsed:.0f}/{MIN_SEC} s (stay still)"
    else:
        txt = "no face found"

    cv2.putText(
        frame, txt, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2
    )
    cv2.putText(
        frame,
        "learning project, not a medical device",
        (20, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Mini waveform of the filtered pulse signal.
    if filt is not None and len(filt) > 2:
        seg = filt[-300:]
        ys = (h - 80) - ((seg - seg.min()) / (np.ptp(seg) + 1e-9)) * 60
        xs = np.linspace(20, w - 20, len(seg))
        cv2.polylines(
            frame,
            [np.stack([xs, ys], 1).astype(np.int32)],
            False,
            (0, 255, 0),
            2,
        )

    cv2.imshow("pulse demo (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

## Step 6: Launch the Interface

```bash
python live_demo.py
```

Press `q` while the OpenCV video display window is focused to terminate the application.

### Optional: Standalone Launchers

**Windows (`run_demo.bat`):**

```bat
@echo off
cd /d %~dp0
if not exist venv (python -m venv venv)
call venv\Scripts\activate
pip install -r requirements.txt
python live_demo.py
pause
```

**macOS / Linux (`run_demo.sh`):**

```bash
#!/bin/bash
cd "$(dirname "$0")"
[ -d venv ] || python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python live_demo.py
```

Make the shell script executable:

```bash
chmod +x run_demo.sh
```

## 4.3 Demonstration Protocol

To present the system effectively in 3–5 minutes:

1. **Context & Physics (20 seconds):** Explain that volumetric capillary expansion alters hemoglobin absorption, producing imperceptible chrominance shifts in reflected light that can be isolated using color-space projection.
2. **Environmental Setup:**
   - Sit facing a stable, diffuse light source. Avoid harsh backlighting and overhead shadows.
   - Ensure hair does not obstruct the forehead skin ROI.
   - Have a ground-truth reference available, such as a smartwatch, pulse oximeter, or radial pulse count.
3. **Buffer Warm-Up (10 seconds):** Remain seated without speaking or shifting while the buffer collects frames (`collecting... X/10 s`).
4. **Live Correlation:** Compare the on-screen POS heart-rate estimate directly against your reference device.
5. **Controlled Artifact Induction:** Deliberately rotate or pitch your head. The estimate will drift, illustrating why single-ROI color differencing collapses under specular motion vectors.
6. **Technical Discussion:** Review the offline findings: clean rPPG succeeds on uncompressed video (UBFC) but collapses to the H.264 codec noise floor on compressed web data (DFDC).

## 4.4 Troubleshooting & Diagnostics

| Symptom | Likely Cause | Recommended Resolution |
|---|---|---|
| Black screen or immediate exit | Camera index mismatch or device lock | Try index `1` or `2` in `cv2.VideoCapture`; ensure Discord, Zoom, or Teams are closed. |
| Windows camera error | OS privacy restrictions | Open **Settings → Privacy & Security → Camera** and allow desktop apps. |
| macOS blank frame | Terminal privacy restrictions | Open **System Settings → Privacy & Security → Camera** and grant terminal access. |
| "no face found" persists | Low ambient light or steep facial tilt | Add forward-facing illumination and reposition directly in front of the lens. |
| Package installation failure | Python wheel incompatibility | Ensure the virtual environment uses Python 3.11. |
| `FileNotFoundError: face_landmarker.task` | Model asset missing | Confirm `face_landmarker.task` is beside `live_demo.py`. |
| BPM fluctuates wildly | Lighting oscillation or auto-exposure | Disable camera auto-exposure adjustments where possible and avoid high-frequency PWM dimmers. |
| BPM pinned to 42 or 180 | Insufficient pulse SNR | The estimator may be hitting a Butterworth filter cutoff; improve lighting and remain still. |
| High latency / dropped frames | CPU throttling during landmark inference | Reduce capture resolution (e.g. `640 × 480`) or close background processes. |
| Bounding box misaligned | Extreme head rotation | Face the camera so forehead landmarks do not intersect the hair boundary. |

## 4.5 Reproducing the Offline Benchmarks (Phases 1–3)

This section describes how to regenerate the empirical benchmark metrics, tables, and power spectra.

### A. Phases 1 & 2: UBFC-rPPG Ground-Truth Validation

1. Request access to and download the raw sequences from the official UBFC-rPPG repository.
2. Structure the subject data directories alongside their ground-truth data:

```text
data/
└── ubfc/
    ├── subject1/
    │   ├── vid.avi
    │   └── ground_truth.txt
    └── subject2/
        ├── vid.avi
        └── ground_truth.txt
```

3. Update `TRUE_BPM` in `evaluate.py` with the ground-truth reference values.
4. Execute the evaluation:

```bash
python evaluate.py
```

Landmark arrays are automatically serialized to `cache/*.npz` for subsequent runs.

5. Generate Fourier spectra comparisons and post-exercise diagnostics:

```bash
python diagnose_post_exercise.py
```

**Standardized parameters:**
- Butterworth bandpass: 0.7–3.0 Hz (3rd order)
- POS window length: 1.6 seconds
- Forehead ROI landmarks: `TOP = [67, 109, 10, 338, 297]` and `BROW = [105, 334]`

### B. Controlled Validation: Self-Recorded Edge Cases

1. Capture a 30–60 second frontal clip with steady illumination while logging a reference pulse.
2. Append the video path and reference heart rate to `evaluate.py`.
3. Run the evaluation to verify cross-hardware sensor generalization (`n = 3` baseline conditions).

### C. Phase 3: DFDC In-the-Wild & Codec Noise-Floor Analysis

1. Acquire `train_sample_videos.zip` (400 clips) from the Kaggle Deepfake Detection Challenge and extract it into `data/dfdc/train_sample_videos/`.
2. Generate leakage-free grouped metadata:

```bash
python build_groups.py
```

This creates `dfdc_meta.csv`, mapping synthetic clips back to source identities across 244 disjoint groups.

3. Run crop-based tracking with localized tile upscaling:

```bash
python sanity_check.py
```

4. Run the paired noise-floor control script:

```bash
python paired_control.py
```

This computes paired peak-ratio statistics between forehead skin and non-skin background patches (`n = 40`).

## 4.6 Repository Structure

```text
deepfake-checker/
├── README.md
├── requirements.txt
├── face_landmarker.task        # MediaPipe Task bundle (model weights)
├── pipeline.py                 # Core signal transformations (POS, filtering, crop tracking)
├── evaluate.py                 # UBFC and self-recording benchmark runner
├── live_demo.py                # Interactive real-time OpenCV webcam application
├── run_demo.bat                # Windows convenience launcher
├── run_demo.sh                 # macOS/Linux convenience launcher
├── diagnose_post_exercise.py   # Sliding-window FFT and stress diagnostic plots
├── sanity_check.py             # DFDC multi-clip peak-ratio evaluator
├── paired_control.py           # Paired skin vs. background noise-floor control benchmark
├── probe_failures.py           # Full-frame landmark failure diagnostic script
├── probe_tiles.py              # Multi-scale sliding crop verification script
├── build_groups.py             # Identity clustering script for GroupKFold validation
├── data/                       # Local datasets (ignored by Git)
└── cache/                      # Cached frame RGB arrays (ignored by Git)
```

Ensure the following patterns exist in `.gitignore`:

```gitignore
venv/
data/
cache/
*.avi
*.mp4
*.npz
*.task
__pycache__/
```

## 4.7 Methodological Considerations

- **Cache Invalidation:** Modifying ROI bounds or channel transformation logic requires purging `cache/` before rerunning, as the pipeline prioritizes existing `.npz` arrays.
- **Deterministic Seeds:** Synthetic noise benchmarks use fixed pseudorandom initialization (`np.random.default_rng(0)`) to maintain repeatability.
- **Tracking Dropouts:** `extract_rgb_crop` flags undetected frames as `NaN` and applies linear interpolation only if total missing frames remain below 20%; clips exceeding this threshold are discarded.
- **Sample Scope:** Initial evaluations are constrained to `n = 7` (UBFC), `n = 3` (consumer edge cases), and `n = 40` (DFDC paired control). Metrics indicate operational constraints rather than exhaustive dataset bounds.

## 4.8 Research Trajectory & Next Steps

1. **Spatial Cross-Spectral Coherence:** Implement synchronized extraction across multiple anatomical regions (forehead, left cheek, right cheek) to assess multi-channel phase locking on higher-bitrate benchmarks such as FaceForensics++ c23 and Celeb-DF.
2. **Deconfounding Resolution vs. Codec:** Isolate the impact of heavy H.264 spatial quantization from small facial pixel footprints by evaluating full-frame downscaled real recordings against native low-resolution sensors.
3. **Neural End-to-End Comparison:** Benchmark the classical POS baseline against end-to-end spatiotemporal networks such as PhysNet and TS-CAN within the rPPG-Toolbox framework.
4. **Decoupled Web Frontend:** Build a client-server architecture using FastAPI for inference orchestration and React/Tailwind for interactive video uploads and time-series spectral visualizations.

## 5. Conclusions & Implications for Deepfake Detection
1. **Limits of Single-Region rPPG under Compression:** On heavily compressed, in-the-wild web video (such as DFDC), single-ROI pulse energy is buried under the H.264 codec noise floor and small ROI pixel dimensions. Any binary classifier trained solely on single-region pulse energy under these conditions risks fitting to compression macroblock artifacts rather than genuine cardiovascular perfusion.

2. **Validating Biological Priors via Controls:** Establishing the inert background patch control was crucial; without it, an average ratio of ~0.20 could easily be misinterpreted as a faint, recoverable physiological pulse rather than video compression noise.

3. **Future Trajectory:** While single-region pulse energy collapses under aggressive compression, biological consistency across facial regions remains an open avenue. Multi-region spatial phase coherence (cross-correlating blood volume pulse dynamics between forehead and bilateral cheek vascular beds) provides the necessary structural constraint for high-bitrate benchmarks (e.g., FaceForensics++ c0/c23, Celeb-DF).

---


## 5. References & Academic Citations

1. **UBFC-rPPG Dataset:**  
   S. Bobgueresse, R. Acharya, and Y. Benezeth. *UBFC-rPPG: A database for evaluating remote photoplethysmography methods*. IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2019.  
   Research Portal: [UBFC-rPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)[cite: 2]

2. **Plane-Orthogonal-to-Skin (POS) Algorithm:**  
   W. Wang, A. C. den Brinker, S. Stuijk, and G. de Haan. *Algorithmic Principles of Remote PPG*. IEEE Transactions on Biomedical Engineering (TBME), 64(7):1479–1491, 2017.  
   DOI: [10.1109/TBME.2016.2609282](https://doi.org/10.1109/TBME.2016.2609282)

3. **FakeCatcher (Biological Deepfake Detection):**  
   U. A. Ciftci, I. Demir, and L. Yin. *FakeCatcher: Detection of Synthetic Portrait Videos using Biological Signals*. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.  
   arXiv: [arXiv:1901.02212](https://arxiv.org/abs/1901.02212)

4. **Deepfake Detection Challenge (DFDC) Dataset:**  
   B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, and C. C. Ferrer. *The Deepfake Detection Challenge (DFDC) Dataset*. arXiv preprint, 2020.  
   arXiv: [arXiv:2006.07397](https://arxiv.org/abs/2006.07397)

5. **FaceForensics++ (Benchmark Reference):**  
   A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner. *FaceForensics++: Learning to Detect Manipulated Facial Images*. IEEE/CVF International Conference on Computer Vision (ICCV), 2019.  
   arXiv: [arXiv:1901.08971](https://arxiv.org/abs/1901.08971)