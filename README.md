# 🛡️ DeepFakeSentry: Audio Forensics & Synthetic Media Detector

DeepFakeSentry is an explainable audio-forensics pipeline and interactive dashboard designed to detect acoustic artifacts in AI-synthesized speech, voice clones, and manipulated audio/video media (`.wav`, `.mp3`, `.mp4`).

![DeepFakeSentry Pipeline](https://img.shields.io/badge/Pipeline-Audio_Forensics-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)

---

## 📐 Architecture & Pipeline

```text
Video / Audio (.mp4, .wav, .mp3)
     ↓
Audio Extraction & Preprocessing (Librosa / SoundFile)
     ↓
STFT → Log-Mel Spectrogram Matrix
     ↓
┌────────────────────────────────────────────────────────┐
│ Artifact Detectors                                     │
│ • High-Frequency Cutoff (Vocoder bandwidth cap)        │
│ • Robotic Silence Patterns (Uniform gaps & 0dB floor)  │
│ • Spectral Discontinuities (Boundary jump spikes)      │
└────────────────────────────────────────────────────────┘
     ↓
Weighted Authenticity Risk Scoring Engine
     ↓
Streamlit Forensic Dashboard with Anomaly Overlays
```

---

## 🔍 Core Forensic Signals

1. **High-Frequency Cutoff**:
   - Analyzes power spectral density above 6.5 kHz.
   - Detects brick-wall frequency suppression typical of neural vocoders (e.g. Tacotron, Bark, HiFi-GAN at lower sampling rates).
2. **Robotic Silence Patterns**:
   - Detects artificially uniform silence gap durations ($CV < 0.25$) and zeroed digital noise floors.
3. **Spectral Discontinuities**:
   - Computes frame-to-frame log-mel distance metrics to pinpoint unnatural boundary jumps.

---

## 🚀 Quick Start

### 1. Installation
```bash
py -m pip install -r requirements.txt
```

### 2. Generate Sample Media Files
```bash
py generate_samples.py
```

### 3. Launch Forensic Dashboard
```bash
py -m streamlit run app.py
```

---

## 🎙️ 60–90 Second Hackathon Demo Script

> **Framing Note:** *"DeepFakeSentry identifies acoustic artifacts associated with synthetic or manipulated media and produces an explainable authenticity-risk score."*

* **0–10 sec:** Upload clip / Select Demo Sample.
  > *"DeepFakeSentry analyzes audio artifacts that aren't necessarily obvious to human listeners."*
* **10–30 sec:** Show Spectrogram Generation.
  > *"We convert the audio into a time-frequency representation. Here, frequency is vertical and time is horizontal."*
* **30–50 sec:** Highlight Anomaly Overlays.
  > *"This sudden loss of high-frequency energy is suspicious. We also see unnaturally regular silence intervals."*
* **50–70 sec:** Show Scoring Engine.
  > *"Each forensic signal contributes to an explainable risk score rather than relying on a black-box prediction."*
* **70–90 sec:** Final Result.
  > *"This clip receives an authenticity score of 34 out of 100, with the major evidence coming from frequency cutoff and repetitive silence."*
