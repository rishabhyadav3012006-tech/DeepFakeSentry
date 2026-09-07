import os
import numpy as np
import soundfile as sf

def generate_authentic_sample(filename: str, sr: int = 22050, duration: float = 5.0):
    """
    Generates a realistic simulation of natural human speech:
    - Full frequency bandwidth up to 10.5 kHz
    - Natural intonation / pitch contours
    - Varied, organic silence durations with background breath/room noise floor
    - Smooth spectral transitions
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t)

    # 1. Pitch contour with natural vibrato/prosody (f0 ~ 135 Hz)
    f0 = 135.0 + 18.0 * np.sin(2 * np.pi * 0.7 * t) + 6.0 * np.sin(2 * np.pi * 3.2 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # 2. Rich speech harmonics spanning up to high frequencies
    harmonics = [1.0, 0.8, 0.6, 0.45, 0.35, 0.28, 0.22, 0.18, 0.14, 0.10, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01]
    for i, amp in enumerate(harmonics, start=1):
        audio += amp * np.sin(i * phase)

    # Formant resonance dynamics
    formant_filter = 1.0 + 0.35 * np.sin(2 * np.pi * 2.8 * t)
    audio *= formant_filter

    # 3. Add high-frequency sibilance/fricatives (natural speech consonants & breath noise spanning 6kHz - 10kHz)
    high_freq_noise = np.random.normal(0, 0.08, len(t))
    # Filter noise so it remains strong in high frequencies organically
    audio += high_freq_noise * (0.15 + 0.05 * np.sin(2 * np.pi * 1.8 * t))

    # 4. Organic variable speech pause lengths (e.g. Pause 1: 0.42s at 1.1s, Pause 2: 0.78s at 3.1s)
    speech_env = np.ones_like(t)
    speech_env[(t >= 1.1) & (t <= 1.52)] = 0.05  # 0.42s pause
    speech_env[(t >= 3.1) & (t <= 3.88)] = 0.05  # 0.78s pause

    # Smooth pause edges
    kernel_size = int(sr * 0.04)
    kernel = np.hanning(kernel_size)
    kernel /= kernel.sum()
    speech_env = np.convolve(speech_env, kernel, mode='same')

    audio *= speech_env

    # 5. Continuous natural room acoustics / ambient noise floor
    room_noise = np.random.normal(0, 0.005, len(t))
    audio += room_noise

    # Normalize audio peak
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio.astype(np.float32), sr)
    print(f"Generated authentic audio: {filename}")

def generate_synthetic_sample(filename: str, sr: int = 22050, duration: float = 5.0):
    """
    Generates a simulation of AI-cloned speech containing prominent forensic artifacts:
    1. High-frequency brick-wall cutoff above 5.8 kHz (low-bandwidth vocoder)
    2. Robotic, perfectly regular silence intervals (exact 0.30s gaps) with digital zero floor
    3. Sharp spectral frame-boundary discontinuities
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.zeros_like(t)

    # 1. Monotone/Robotic fundamental pitch (f0 = 140 Hz)
    f0 = 140.0 * np.ones_like(t)
    phase = 2 * np.pi * np.cumsum(f0) / sr

    # Rich mid-range energy up to 5.5 kHz
    for i in range(1, 40):
        freq = i * 140.0
        if freq < 5800:
            amp = 1.0 / (i ** 0.6)
            audio += amp * np.sin(i * phase)

    # Add dense mid-frequency fricative noise (500 Hz to 5800 Hz)
    mid_noise = np.random.normal(0, 0.15, len(t))
    audio += mid_noise

    # 2. Hard brick-wall frequency cutoff above 5800 Hz (zero energy in top mel bands)
    fft_audio = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(t), 1/sr)
    fft_audio[freqs > 5800] = 0.0  # Hard brick-wall cutoff!
    audio = np.fft.irfft(fft_audio, n=len(t))

    # 3. Rigid, robotic silence gaps: Exact 0.300s gaps every 1.000s (zeroed floor)
    speech_env = np.ones_like(t)
    for start_sec in [1.0, 2.0, 3.0, 4.0]:
        silence_mask = (t >= start_sec) & (t < start_sec + 0.30)
        speech_env[silence_mask] = 0.0  # Digital zeroing!

    audio *= speech_env

    # 4. Spectral discontinuities: Introduce sudden artificial frame boundary spikes
    for jump_time in [0.75, 1.75, 2.75, 3.75, 4.55]:
        jump_idx = int(jump_time * sr)
        if jump_idx < len(audio) - 500:
            audio[jump_idx:jump_idx + 250] *= 3.5  # Artificial amplitude spike

    # Normalize audio peak
    audio = audio / np.max(np.abs(audio)) * 0.9
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    sf.write(filename, audio.astype(np.float32), sr)
    print(f"Generated synthetic audio: {filename}")

if __name__ == "__main__":
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    generate_authentic_sample(os.path.join(samples_dir, "authentic_speech.wav"))
    generate_synthetic_sample(os.path.join(samples_dir, "synthetic_cloned.wav"))
