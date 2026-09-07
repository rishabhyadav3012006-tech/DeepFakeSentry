import numpy as np
import librosa
from scipy.signal import find_peaks

def detect_frequency_cutoff(S_stft_db: np.ndarray, stft_freqs: np.ndarray, times: np.ndarray, cutoff_threshold_hz: float = 6500.0):
    """
    Detects high-frequency cutoff/collapse artifacts typical of low-bandwidth neural vocoders.
    """
    n_freqs, n_frames = S_stft_db.shape
    
    # 1. Identify active speech frames vs silent frames based on overall frame power
    power_stft = 10.0 ** (S_stft_db / 10.0)
    frame_total_power = np.sum(power_stft, axis=0)
    max_frame_power = np.max(frame_total_power) + 1e-12
    rel_frame_power = frame_total_power / max_frame_power
    speech_frame_mask = rel_frame_power > 0.02  # Active speech frames (> -17 dB from max)
    
    # Define high frequency band vs mid frequency band
    high_band_indices = np.where(stft_freqs >= cutoff_threshold_hz)[0]
    mid_band_indices = np.where((stft_freqs >= 1000.0) & (stft_freqs < cutoff_threshold_hz))[0]
    
    if len(high_band_indices) == 0:
        high_band_indices = np.arange(int(n_freqs * 0.7), n_freqs)
        cutoff_threshold_hz = stft_freqs[high_band_indices[0]]

    high_band_energy = np.mean(power_stft[high_band_indices, :], axis=0)
    mid_band_energy = np.mean(power_stft[mid_band_indices, :], axis=0) + 1e-12
    energy_ratio_per_frame = high_band_energy / mid_band_energy
    
    # Only evaluate energy ratio on ACTIVE speech frames
    if np.sum(speech_frame_mask) > 0:
        speech_energy_ratios = energy_ratio_per_frame[speech_frame_mask]
        cutoff_mask_speech = speech_energy_ratios < 0.005  # Extremely low high-freq power (< -23 dB)
        collapse_percentage = float(np.mean(cutoff_mask_speech) * 100.0)
        mean_speech_ratio = float(np.mean(speech_energy_ratios))
    else:
        collapse_percentage = 0.0
        mean_speech_ratio = 1.0

    # Calculate average frequency profile across active speech frames
    if np.sum(speech_frame_mask) > 0:
        freq_power_profile = np.mean(power_stft[:, speech_frame_mask], axis=1)
    else:
        freq_power_profile = np.mean(power_stft, axis=1)
        
    max_power = np.max(freq_power_profile) + 1e-12
    rel_power_profile = freq_power_profile / max_power
    
    # Find cutoff point where energy drops sharply below 0.2% (-27 dB)
    cutoff_bin_idx = np.where(rel_power_profile < 0.002)[0]
    if len(cutoff_bin_idx) > 0 and cutoff_bin_idx[0] > 5:
        detected_cutoff_freq = float(stft_freqs[cutoff_bin_idx[0]])
    else:
        detected_cutoff_freq = float(stft_freqs[-1])

    cutoff_detected = (detected_cutoff_freq < cutoff_threshold_hz - 200.0) or (collapse_percentage > 35.0)
    
    # Severity score 0-100%
    if cutoff_detected:
        freq_deficit = max(0.0, cutoff_threshold_hz - detected_cutoff_freq)
        severity = min(100.0, (freq_deficit / cutoff_threshold_hz) * 140.0 + collapse_percentage * 0.5)
    else:
        severity = 0.0

    cutoff_mask = np.zeros(n_frames, dtype=bool)
    cutoff_mask[speech_frame_mask] = energy_ratio_per_frame[speech_frame_mask] < 0.005

    details = {
        "detected_cutoff_freq_hz": round(detected_cutoff_freq, 1),
        "target_bandwidth_hz": round(float(stft_freqs[-1]), 1),
        "collapse_percentage": round(collapse_percentage, 1),
        "high_freq_energy_ratio": round(mean_speech_ratio, 4),
        "severity": round(severity, 1)
    }

    return cutoff_detected, detected_cutoff_freq, collapse_percentage, cutoff_mask, details

def detect_robotic_silence(y: np.ndarray, sr: int, hop_length: int = 512, frame_length: int = 2048):
    """
    Detects unnaturally regular silence intervals and zeroed noise floors.
    """
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    threshold = max(np.percentile(rms, 25), 0.008)
    is_silence = rms < threshold
    
    silence_intervals = []
    in_silence = False
    start_t = 0.0
    
    for idx, (t, sil) in enumerate(zip(times, is_silence)):
        if sil and not in_silence:
            in_silence = True
            start_t = t
        elif not sil and in_silence:
            in_silence = False
            duration = t - start_t
            if duration >= 0.15:
                silence_intervals.append((start_t, t, duration))
                
    if in_silence:
        duration = times[-1] - start_t
        if duration >= 0.15:
            silence_intervals.append((start_t, times[-1], duration))
            
    durations = [d for _, _, d in silence_intervals]
    
    silence_detected = False
    regularity_index = 0.0
    zero_floor_detected = False
    
    if len(durations) >= 2:
        mean_dur = np.mean(durations)
        std_dur = np.std(durations)
        cv = std_dur / (mean_dur + 1e-6)
        
        # Rigorous check: CV < 0.18 indicates robotic uniformity (<18% duration variation)
        if cv < 0.18:
            silence_detected = True
            regularity_index = float((1.0 - cv) * 100.0)
            
    # Check for absolute digital zero noise floor during silence
    silence_rms_vals = rms[is_silence]
    if len(silence_rms_vals) > 0:
        min_rms = float(np.min(silence_rms_vals))
        if min_rms < 1e-5:  # Digital zero
            zero_floor_detected = True
            if len(durations) >= 1:
                silence_detected = True

    severity = 0.0
    if silence_detected:
        severity = max(regularity_index, 85.0 if zero_floor_detected else 60.0)
        
    details = {
        "num_silence_intervals": len(durations),
        "avg_silence_duration_sec": round(float(np.mean(durations)), 2) if durations else 0.0,
        "silence_duration_std_sec": round(float(np.std(durations)), 3) if durations else 0.0,
        "regularity_index": round(regularity_index, 1),
        "zero_floor_detected": zero_floor_detected,
        "severity": round(severity, 1)
    }

    return silence_detected, [(s, e) for s, e, _ in silence_intervals], regularity_index, details

def detect_spectral_discontinuity(S_mel_db: np.ndarray, times: np.ndarray):
    """
    Detects abrupt, non-biological spectral frame-to-frame distance spikes.
    """
    n_mels, n_frames = S_mel_db.shape
    
    norms = np.linalg.norm(S_mel_db, axis=0, keepdims=True) + 1e-6
    norm_mel = S_mel_db / norms
    
    diffs = norm_mel[:, 1:] - norm_mel[:, :-1]
    distances = np.linalg.norm(diffs, axis=0)
    distances = np.pad(distances, (1, 0), mode='edge')
    
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)
    threshold = mean_dist + 3.2 * std_dist  # Outlier spike threshold
    
    spike_indices, _ = find_peaks(distances, height=threshold, distance=4)
    spike_times = [float(times[idx]) for idx in spike_indices if idx < len(times)]
    
    discontinuity_detected = len(spike_times) > 0
    severity = min(100.0, len(spike_times) * 25.0) if discontinuity_detected else 0.0
    
    details = {
        "num_discontinuities": len(spike_times),
        "mean_frame_distance": round(float(mean_dist), 4),
        "spike_threshold": round(float(threshold), 4),
        "max_spike_distance": round(float(np.max(distances)), 4) if len(distances) > 0 else 0.0,
        "severity": round(severity, 1)
    }

    return discontinuity_detected, spike_times, distances, details

