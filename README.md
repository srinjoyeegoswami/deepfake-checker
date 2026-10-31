# Pulse-Based Deepfake Detection

Detecting synthetic and manipulated video frames by extracting subtle remote photoplethysmography (rPPG) blood volume pulse signals.

---

---

# Phase 1: Physiological Signal Validation & rPPG Benchmark

A computer vision and physiological signal-processing pipeline that extracts blood volume pulse (BVP) signals from facial video. This phase establishes the biological ground truth, validating cardiac pulse extraction across real human subjects and benchmarking single-channel intensity tracking against multi-channel color-space projections under motion corruption.

---

## Evaluation Metric

Performance across test subjects is quantified using **Mean Absolute Error (MAE)**, defined as the average absolute deviation between algorithm estimates and ground-truth pulse oximeter measurements:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |\text{Estimated}_i - \text{True}_i|$$

For our initial baseline of $N = 3$ subjects:

$$\text{MAE}_{\text{Green}} = \frac{|\text{Err}_1| + |\text{Err}_2| + |\text{Err}_3|}{3} \approx 6.43\text{ BPM}$$

$$\text{MAE}_{\text{POS}} = \frac{|\text{Err}_1| + |\text{Err}_2| + |\text{Err}_3|}{3} \approx 1.53\text{ BPM}$$

---

# Remote Photoplethysmography (rPPG) & Temporal Deepfake Detection

A computer vision and physiological signal-processing pipeline that extracts blood volume pulse (BVP) signals from facial video. The project validates cardiac pulse extraction across real subjects and benchmarks single-channel intensity tracking against multi-channel color-space projections under motion corruption.

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

1. **Facial Landmark Mesh:** MediaPipe FaceLandmarker tracks 468 3D facial landmarks on every frame.
2. **Forehead Skin ROI:** Skin regions are masked using upper forehead landmarks (`TOP = [67, 109, 10, 338, 297]`) trimmed 15% inwards from temple edges and stopping 40% short of the eyebrows (`BROW = [105, 334]`) to prevent hairline and eyelid contamination.
3. **Temporal Filtering:** Detrending eliminates slow baseline drift (lighting changes, camera auto-exposure), followed by a 3rd-order Butterworth bandpass filter ($0.7\text{--}3.0\text{ Hz}$, corresponding to $42\text{--}180\text{ BPM}$).
4. **Spectral Estimation:** Fast Fourier Transform (FFT) identifies the maximum energy frequency bin within the biological cardiac band.
