import numpy as np

def calculate_authenticity_score(cutoff_res, silence_res, discontinuity_res, custom_weights=None):
    """
    Computes a weighted authenticity risk score (0 - 100) and risk diagnostic report.
    
    Formula:
        Authenticity Score = 100 - (Cutoff Penalty + Silence Penalty + Discontinuity Penalty)
        
    Args:
        cutoff_res (tuple): (detected, freq, collapse_pct, mask, details)
        silence_res (tuple): (detected, intervals, reg_score, details)
        discontinuity_res (tuple): (detected, spike_times, distances, details)
        custom_weights (dict): Optional custom weighting factors for penalties.
        
    Returns:
        results (dict): Dictionary containing score, risk level, breakdown, evidence bullets.
    """
    _, _, _, _, cutoff_details = cutoff_res
    _, _, _, silence_details = silence_res
    _, _, _, discontinuity_details = discontinuity_res
    
    # Extract severities (0.0 to 100.0)
    cutoff_sev = cutoff_details.get("severity", 0.0)
    silence_sev = silence_details.get("severity", 0.0)
    discontinuity_sev = discontinuity_details.get("severity", 0.0)
    
    # Default weights
    w = custom_weights or {"cutoff": 0.45, "silence": 0.30, "discontinuity": 0.25}
    
    cutoff_penalty = cutoff_sev * w["cutoff"]
    silence_penalty = silence_sev * w["silence"]
    discontinuity_penalty = discontinuity_sev * w["discontinuity"]
    
    total_penalty = cutoff_penalty + silence_penalty + discontinuity_penalty
    authenticity_score = max(0, min(100, int(round(100.0 - total_penalty))))
    
    # Risk Level Classification
    if authenticity_score < 45:
        risk_label = "HIGH RISK"
        risk_color = "#FF4D4D"  # Bright Crimson / Red
        risk_description = "Strong synthetic acoustic markers detected. High probability of AI speech synthesis or voice cloning."
    elif authenticity_score < 75:
        risk_label = "MODERATE RISK"
        risk_color = "#FFC107"  # Warning Amber / Yellow
        risk_description = "Noticeable acoustic anomalies detected. May indicate partial voice synthesis, audio editing, or heavy compression."
    else:
        risk_label = "LOW RISK / AUTHENTIC"
        risk_color = "#00E676"  # Emerald Green
        risk_description = "Natural acoustic characteristics present. Speech exhibits expected organic spectral continuity and bandwidth."

    # Forensic Evidence Report Bullets
    evidence_points = []
    
    if cutoff_details.get("severity", 0) > 20:
        freq_cutoff = cutoff_details.get("detected_cutoff_freq_hz", 0)
        sev = cutoff_details.get("severity", 0)
        evidence_points.append({
            "title": f"Abrupt High-Frequency Cutoff at {freq_cutoff:.0f} Hz",
            "confidence": f"{int(sev)}%",
            "severity": sev,
            "desc": f"Energy in upper frequency bands collapses dramatically (affected {cutoff_details.get('collapse_percentage')}% of frames), characteristic of low-bandwidth neural vocoders.",
            "icon": "⚡"
        })
        
    if silence_details.get("severity", 0) > 20:
        reg_idx = silence_details.get("regularity_index", 0)
        zero_fl = silence_details.get("zero_floor_detected", False)
        sev = silence_details.get("severity", 0)
        desc = "Silences have unnaturally uniform gap durations and exact digital zero noise floors, typical of TTS pause insertion." if zero_fl else "Silences show high periodic uniformity."
        evidence_points.append({
            "title": "Robotic Silence & Gap Pattern",
            "confidence": f"{int(sev)}%",
            "severity": sev,
            "desc": desc,
            "icon": "░"
        })

    if discontinuity_details.get("severity", 0) > 20:
        num_spikes = discontinuity_details.get("num_discontinuities", 0)
        sev = discontinuity_details.get("severity", 0)
        evidence_points.append({
            "title": f"Spectral Discontinuities ({num_spikes} boundary spikes)",
            "confidence": f"{int(sev)}%",
            "severity": sev,
            "desc": "Unnatural frame-to-frame spectral jumps detected at synthesis chunk boundaries.",
            "icon": "⚠"
        })

    if not evidence_points:
        evidence_points.append({
            "title": "No Significant Anomaly Detected",
            "confidence": "95%",
            "severity": 0,
            "desc": "Spectral continuity and acoustic noise floor conform to natural human speech dynamics.",
            "icon": "✓"
        })

    return {
        "score": authenticity_score,
        "risk_label": risk_label,
        "risk_color": risk_color,
        "risk_description": risk_description,
        "penalties": {
            "cutoff": round(cutoff_penalty, 1),
            "silence": round(silence_penalty, 1),
            "discontinuity": round(discontinuity_penalty, 1)
        },
        "severities": {
            "cutoff": round(cutoff_sev, 1),
            "silence": round(silence_sev, 1),
            "discontinuity": round(discontinuity_sev, 1)
        },
        "evidence": evidence_points
    }
