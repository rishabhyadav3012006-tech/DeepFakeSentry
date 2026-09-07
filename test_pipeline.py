import os
from audio import load_media_audio, compute_spectrogram
from detector import (
    detect_frequency_cutoff,
    detect_robotic_silence,
    detect_spectral_discontinuity
)
from scoring import calculate_authenticity_score

def test_sample(file_path):
    print(f"\n==========================================")
    print(f"Testing File: {os.path.basename(file_path)}")
    print(f"==========================================")
    
    y, sr = load_media_audio(file_path)
    times, stft_freqs, mel_freqs, S_stft_db, S_mel_db = compute_spectrogram(y, sr)
    
    cutoff_res = detect_frequency_cutoff(S_stft_db, stft_freqs, times)
    silence_res = detect_robotic_silence(y, sr)
    discontinuity_res = detect_spectral_discontinuity(S_mel_db, times)
    
    score_data = calculate_authenticity_score(cutoff_res, silence_res, discontinuity_res)
    
    print(f"Authenticity Score : {score_data['score']} / 100")
    print(f"Risk Label         : {score_data['risk_label']}")
    print(f"Cutoff Penalty     : {score_data['penalties']['cutoff']}")
    print(f"Silence Penalty    : {score_data['penalties']['silence']}")
    print(f"Discontinuity Pen. : {score_data['penalties']['discontinuity']}")
    print("Evidence:")
    for ev in score_data['evidence']:
        print(f" - [{ev['confidence']}] {ev['title']}: {ev['desc']}")
        
    return score_data

if __name__ == "__main__":
    samples_dir = os.path.join(os.path.dirname(__file__), "samples")
    synth_score = test_sample(os.path.join(samples_dir, "synthetic_cloned.wav"))
    auth_score = test_sample(os.path.join(samples_dir, "authentic_speech.wav"))
    
    assert synth_score['score'] < 50, f"Expected synthetic score < 50, got {synth_score['score']}"
    assert auth_score['score'] >= 75, f"Expected authentic score >= 75, got {auth_score['score']}"
    print("\nSUCCESS: PIPELINE INTEGRITY TEST PASSED!")
