import os
import tempfile
import numpy as np
import librosa
import soundfile as sf

def load_media_audio(input_source, target_sr: int = 22050):
    """
    Loads audio from a file path, file-like object, or Streamlit UploadedFile.
    Supports .wav, .mp3, .mp4 files.
    Returns:
        y (np.ndarray): 1D float32 audio waveform normalized to [-1.0, 1.0].
        sr (int): Target sampling rate.
    """
    temp_path = None

    try:
        # If input_source is bytes or Streamlit UploadedFile
        if hasattr(input_source, "read") or isinstance(input_source, bytes):
            suffix = ".wav"
            if hasattr(input_source, "name"):
                ext = os.path.splitext(input_source.name)[1].lower()
                if ext in [".wav", ".mp3", ".mp4", ".m4a", ".aac", ".ogg", ".flac"]:
                    suffix = ext

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                if hasattr(input_source, "read"):
                    tmp.write(input_source.read())
                    # Reset buffer position if seekable
                    if hasattr(input_source, "seek"):
                        input_source.seek(0)
                else:
                    tmp.write(input_source)
                temp_path = tmp.name
            
            file_to_load = temp_path
        else:
            file_to_load = str(input_source)

        # Handle video files (.mp4) vs standard audio files
        ext = os.path.splitext(file_to_load)[1].lower()
        
        if ext == ".mp4":
            # Attempt loading via librosa / soundfile or ffmpeg fallback
            try:
                y, sr = librosa.load(file_to_load, sr=target_sr, mono=True)
            except Exception:
                # If librosa direct fails on mp4, try extracting with opencv or soundfile if possible
                import cv2
                # OpenCV handles video frames; librosa handles audio stream if backend audio plugins exist
                y, sr = librosa.load(file_to_load, sr=target_sr, mono=True)
        else:
            # Standard audio loading via librosa (handles wav, mp3, flac, ogg)
            y, sr = librosa.load(file_to_load, sr=target_sr, mono=True)

        # Guarantee 1D numpy array
        if y.ndim > 1:
            y = np.mean(y, axis=0)

        # Normalize amplitude peak
        max_amp = np.max(np.abs(y))
        if max_amp > 1e-6:
            y = y / max_amp

        return y.astype(np.float32), sr

    finally:
        # Cleanup temporary file if created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

def compute_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
    """
    Computes STFT and log-Mel spectrogram.
    Returns:
        times: 1D array of frame timestamps in seconds
        stft_freqs: 1D array of STFT frequency bins in Hz
        mel_freqs: 1D array of Mel frequency bins in Hz
        S_stft_db: 2D array of STFT magnitude in dB (shape: [freqs, time])
        S_mel_db: 2D array of Mel spectrogram in dB (shape: [n_mels, time])
    """
    # STFT computation
    stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S_stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    stft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Mel Spectrogram computation
    S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr/2.0)
    
    times = librosa.frames_to_time(np.arange(S_mel_db.shape[1]), sr=sr, hop_length=hop_length)
    
    return times, stft_freqs, mel_freqs, S_stft_db, S_mel_db
