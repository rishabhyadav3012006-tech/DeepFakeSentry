import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Custom Modules
from audio import load_media_audio, compute_spectrogram
from detector import (
    detect_frequency_cutoff,
    detect_robotic_silence,
    detect_spectral_discontinuity
)
from scoring import calculate_authenticity_score
from visualization import plot_forensic_spectrogram

# Page Configuration
st.set_page_config(
    page_title="DeepFakeSentry - Audio Forensics Pipeline",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark CSS Styling
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0D1117;
        color: #C9D1D9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card */
    .main-header {
        background: linear-gradient(135deg, #161B22 0%, #0D1117 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58A6FF 0%, #BC8CFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #8B949E;
        font-size: 1.0rem;
        margin: 0;
    }
    
    /* Score Metric Card */
    .score-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }
    
    .score-number {
        font-size: 3.5rem;
        font-weight: 900;
        line-height: 1.1;
        margin: 10px 0;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        margin-bottom: 12px;
    }
    
    /* Forensic Evidence Cards */
    .evidence-card {
        background: #161B22;
        border: 1px solid #30363D;
        border-left: 4px solid #58A6FF;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    
    .evidence-title {
        font-weight: 700;
        font-size: 1.0rem;
        color: #F0F6FC;
        margin-bottom: 4px;
    }
    
    .evidence-desc {
        font-size: 0.88rem;
        color: #8B949E;
        margin: 0;
    }
    
    /* Sidebar Styling */
    .css-1d351ef, [data-testid="stSidebar"] {
        background-color: #161B22 !important;
        border-right: 1px solid #30363D;
    }
</style>
""", unsafe_allow_html=True)

# Main Application Layout
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛡️ DeepFakeSentry</h1>
    <p class="subtitle">Multimodal Synthetic Media Detector & Forensic Spectrogram Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ── Ensure demo samples exist at cold-start (Streamlit Cloud ephemeral filesystem) ──
import generate_samples as _gs

@st.cache_resource(show_spinner=False)
def _ensure_samples():
    _samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(_samples_dir, exist_ok=True)
    synth_path = os.path.join(_samples_dir, "synthetic_cloned.wav")
    auth_path  = os.path.join(_samples_dir, "authentic_speech.wav")
    if not os.path.exists(synth_path):
        _gs.generate_synthetic_sample(synth_path)
    if not os.path.exists(auth_path):
        _gs.generate_authentic_sample(auth_path)

_ensure_samples()

# Sidebar Controls
st.sidebar.markdown("## 🛡️ DeepFakeSentry")
st.sidebar.title("Forensic Controls")

input_mode = st.sidebar.radio(
    "Select Input Source",
    options=["Upload Audio/Video", "Demo: Synthetic AI Voice", "Demo: Authentic Natural Voice"],
    index=1
)

audio_file = None
samples_dir = os.path.join(os.path.dirname(__file__), "samples")

if input_mode == "Upload Audio/Video":
    audio_file = st.sidebar.file_uploader(
        "Upload media clip (.mp4, .wav, .mp3)",
        type=["wav", "mp3", "mp4", "m4a", "flac"]
    )
elif input_mode == "Demo: Synthetic AI Voice":
    audio_file = os.path.join(samples_dir, "synthetic_cloned.wav")
else:
    audio_file = os.path.join(samples_dir, "authentic_speech.wav")

st.sidebar.markdown("---")
st.sidebar.subheader("Overlay Settings")
show_cutoff = st.sidebar.checkbox("High-Frequency Cutoff Overlay", value=True)
show_silence = st.sidebar.checkbox("Robotic Silence Overlay", value=True)
show_discontinuity = st.sidebar.checkbox("Spectral Discontinuity Overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Detection Sensitivity")
cutoff_thresh = st.sidebar.slider("Cutoff Threshold (Hz)", 4000, 10000, 6500, 500)

if audio_file is not None:
    with st.spinner("Analyzing acoustic forensics & computing Mel Spectrogram..."):
        # 1. Load Audio
        y, sr = load_media_audio(audio_file)
        duration_sec = len(y) / sr
        
        # 2. Compute Spectrogram
        times, stft_freqs, mel_freqs, S_stft_db, S_mel_db = compute_spectrogram(y, sr)
        
        # 3. Run Forensic Detectors
        cutoff_res = detect_frequency_cutoff(S_stft_db, stft_freqs, times, cutoff_threshold_hz=cutoff_thresh)
        silence_res = detect_robotic_silence(y, sr)
        discontinuity_res = detect_spectral_discontinuity(S_mel_db, times)
        
        # 4. Calculate Score
        score_data = calculate_authenticity_score(cutoff_res, silence_res, discontinuity_res)

    # Top Section: Dashboard Split (Score Card vs Player & Audio Overview)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="score-card">
            <div style="color: #8B949E; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;">
                AUTHENTICITY RISK SCORE
            </div>
            <div class="score-number" style="color: {score_data['risk_color']}">
                {score_data['score']} <span style="font-size: 1.4rem; color: #8B949E;">/ 100</span>
            </div>
            <div class="risk-badge" style="background-color: {score_data['risk_color']}22; color: {score_data['risk_color']}; border: 1px solid {score_data['risk_color']};">
                {score_data['risk_label']}
            </div>
            <p style="color: #C9D1D9; font-size: 0.88rem; margin-top: 8px;">
                {score_data['risk_description']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.subheader("🔊 Audio Player & Clip Info")
        
        # Audio Player
        if isinstance(audio_file, str):
            st.audio(audio_file)
        else:
            audio_file.seek(0)
            st.audio(audio_file)
            
        # Audio Metadata Grid
        m1, m2, m3 = st.columns(3)
        m1.metric("Sampling Rate", f"{sr} Hz")
        m2.metric("Duration", f"{duration_sec:.2f} s")
        m3.metric("Bandwidth", f"{stft_freqs[-1]:.0f} Hz")

    st.markdown("---")

    # Main Spectrogram Section
    st.subheader("📊 Audio Mel Spectrogram & Anomaly Map")
    fig = plot_forensic_spectrogram(
        times,
        mel_freqs,
        S_mel_db,
        cutoff_res=cutoff_res,
        silence_res=silence_res,
        discontinuity_res=discontinuity_res,
        show_cutoff=show_cutoff,
        show_silence=show_silence,
        show_discontinuity=show_discontinuity
    )
    st.pyplot(fig)

    st.markdown("---")

    # Detected Artifacts Breakdown Section
    st.subheader("🔍 Detected Forensic Artifact Breakdown")
    
    art_col1, art_col2, art_col3 = st.columns(3)
    
    _, _, _, _, cutoff_details = cutoff_res
    _, _, _, silence_details = silence_res
    _, _, _, disc_details = discontinuity_res
    
    with art_col1:
        sev = cutoff_details.get('severity', 0)
        color = "#FF4D4D" if sev > 30 else "#00E676"
        st.markdown(f"""
        <div class="evidence-card" style="border-left-color: {color}">
            <div class="evidence-title">⚡ High-Frequency Cutoff</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {color}; margin: 4px 0;">
                {sev:.0f}% <span style="font-size: 0.8rem; color: #8B949E;">Severity</span>
            </div>
            <p class="evidence-desc">
                Detected cutoff: <b>{cutoff_details.get('detected_cutoff_freq_hz', 0):.0f} Hz</b><br>
                Collapsed frames: {cutoff_details.get('collapse_percentage', 0):.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

    with art_col2:
        sev = silence_details.get('severity', 0)
        color = "#FF4D4D" if sev > 30 else "#00E676"
        st.markdown(f"""
        <div class="evidence-card" style="border-left-color: {color}">
            <div class="evidence-title">░ Robotic Silence Patterns</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {color}; margin: 4px 0;">
                {sev:.0f}% <span style="font-size: 0.8rem; color: #8B949E;">Severity</span>
            </div>
            <p class="evidence-desc">
                Regularity Index: <b>{silence_details.get('regularity_index', 0):.1f}%</b><br>
                Zero-floor detected: {'Yes ⚠' if silence_details.get('zero_floor_detected') else 'No ✓'}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with art_col3:
        sev = disc_details.get('severity', 0)
        color = "#FF4D4D" if sev > 30 else "#00E676"
        st.markdown(f"""
        <div class="evidence-card" style="border-left-color: {color}">
            <div class="evidence-title">⚠ Spectral Discontinuities</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {color}; margin: 4px 0;">
                {sev:.0f}% <span style="font-size: 0.8rem; color: #8B949E;">Severity</span>
            </div>
            <p class="evidence-desc">
                Boundary Spikes: <b>{disc_details.get('num_discontinuities', 0)}</b><br>
                Max Jump Dist: {disc_details.get('max_spike_distance', 0):.3f}
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Detailed Tabs
    tab1, tab2 = st.tabs(["🔬 Technical Signal Inspector", "🎙️ 60–90 Sec Hackathon Demo Script"])
    
    with tab1:
        st.subheader("Acoustic Signal Diagnostics")
        col_sig1, col_sig2 = st.columns(2)
        
        with col_sig1:
            st.markdown("#### Frame Distance Profile")
            fig_dist, ax_dist = plt.subplots(figsize=(6, 2.5), dpi=100)
            fig_dist.patch.set_facecolor('#161B22')
            ax_dist.set_facecolor('#161B22')
            
            distances = disc_details.get('distances', np.zeros(len(times)))
            ax_dist.plot(times[:len(distances)], distances, color='#00DFD8', linewidth=1.2)
            ax_dist.axhline(disc_details.get('spike_threshold', 0), color='#FF3366', linestyle='--', label='Spike Threshold')
            ax_dist.set_xlabel("Time (s)", color='#8B949E', fontsize=8)
            ax_dist.set_ylabel("Frame Distance", color='#8B949E', fontsize=8)
            ax_dist.tick_params(colors='#8B949E', labelsize=8)
            ax_dist.legend(facecolor='#0D1117', labelcolor='#C9D1D9', fontsize=7)
            for spine in ax_dist.spines.values():
                spine.set_color('#30363D')
            st.pyplot(fig_dist)

        with col_sig2:
            st.markdown("#### Penalties Breakdown")
            penalties = score_data['penalties']
            fig_pen, ax_pen = plt.subplots(figsize=(6, 2.5), dpi=100)
            fig_pen.patch.set_facecolor('#161B22')
            ax_pen.set_facecolor('#161B22')
            
            categories = ['Freq Cutoff', 'Silence Pattern', 'Discontinuity']
            values = [penalties['cutoff'], penalties['silence'], penalties['discontinuity']]
            ax_pen.bar(categories, values, color=['#FF3366', '#7928CA', '#00DFD8'], width=0.5)
            ax_pen.set_ylabel("Penalty Points", color='#8B949E', fontsize=8)
            ax_pen.tick_params(colors='#8B949E', labelsize=8)
            for spine in ax_pen.spines.values():
                spine.set_color('#30363D')
            st.pyplot(fig_pen)

    with tab2:
        st.subheader("🎙️ Hackathon Demo Pitch Script")
        st.markdown("""
        > **Framing Note:** *"DeepFakeSentry identifies acoustic artifacts associated with synthetic or manipulated media and produces an explainable authenticity-risk score."*
        
        ---
        
        * **0–10 sec:** Upload the suspicious clip / Select Demo Sample.
          > *"DeepFakeSentry analyzes audio artifacts that aren't necessarily obvious to human listeners."*
        
        * **10–30 sec:** Show Spectrogram Generation.
          > *"We convert the audio into a time-frequency representation. Here, frequency is vertical and time is horizontal."*
        
        * **30–50 sec:** Highlight Anomaly Overlays.
          > *"This sudden loss of high-frequency energy above 6.5kHz is suspicious. We also see unnaturally regular silence intervals."*
        
        * **50–70 sec:** Show Explainable Scoring Engine.
          > *"Each forensic signal contributes to an explainable risk score rather than relying on a black-box prediction."*
        
        * **70–90 sec:** Final Result & Defensive Takeaway.
          > *"This clip receives an authenticity score of 34 out of 100, with the major evidence coming from frequency cutoff and repetitive silence."*
        """)

else:
    st.info("👈 Please select or upload an audio/video file from the sidebar to start forensic analysis.")
