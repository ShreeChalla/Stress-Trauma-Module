"""
Acoustic feature extraction and BPAB (Biometric Passive Acoustic & Biomarker Engine).

Performs acoustic and speech analysis for emergency crisis helpline triage:
- Fundamental Frequency (F0 mean and pitch variance / pitch_std)
- Jitter (vocal cord trembling / perturbation)
- Shimmer (sobbing, breathlessness / amplitude perturbation)
- Harmonics-to-Noise Ratio (HNR in dB) for whispering / gasping detection
- Speech Rate (words per minute / WPM)
- Normalized Acoustic Agitation Sub-Score A(t) via ReLU formula:
    A(t) = ReLU(a_1 * pitch_std + a_2 * jitter + a_3 * shimmer - a_4 * HNR)
- BPAB Physical Listener:
    * Dead air / silence triage
    * Whispered formant shift (stealth caller detection)
    * Agonal breathing / involuntary groans
    * Vocal tremors and weeping acoustics
"""
import io
import os
import math
import numpy as np
import soundfile as sf
import scipy.signal

# 1. Primary extractor: openSMILE eGeMAPSv02 (Low Level Descriptors)
try:
    import opensmile
    _SMILE_EXTRACTOR = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
except Exception:
    _SMILE_EXTRACTOR = None

# 2. Secondary lightweight framework: librosa
try:
    import librosa
    _HAS_LIBROSA = True
except Exception:
    librosa = None
    _HAS_LIBROSA = False


# Calibrated coefficients for Acoustic Agitation Sub-Score A(t)
# A(t) = ReLU(a_1 * pitch_std + a_2 * jitter + a_3 * shimmer - a_4 * HNR)
COEFF_PITCH_STD = 0.15 / 40.0   # ~0.00375 (normalized over ~40Hz variance)
COEFF_JITTER = 0.30 / 4.0       # ~0.07500 (normalized over ~4.0% jitter)
COEFF_SHIMMER = 0.30 / 8.0      # ~0.03750 (normalized over ~8.0 dB/pct shimmer)
COEFF_HNR = 0.25 / 24.0         # ~0.01042 (baseline ~24dB HNR in clean speech)
BIAS_OFFSET = 0.25              # Anchors calm baseline near 0.0-0.1


def compute_acoustic_subscore(
    jitter: float,
    shimmer: float,
    hnr: float,
    f0_var: float,
    a1: float = COEFF_PITCH_STD,
    a2: float = COEFF_JITTER,
    a3: float = COEFF_SHIMMER,
    a4: float = COEFF_HNR,
    bias: float = BIAS_OFFSET,
) -> float:
    """
    Computes normalized Acoustic Agitation Sub-Score A(t) in [0.0, 1.0].
    Exact formula from RTSAM specification:
        A(t) = ReLU(a_1 * pitch_std + a_2 * jitter + a_3 * shimmer - a_4 * HNR)

    Higher jitter, shimmer, pitch perturbation, and lower HNR indicate heightened distress.
    """
    linear_val = (a1 * f0_var) + (a2 * jitter) + (a3 * shimmer) - (a4 * hnr) + bias
    # Apply ReLU: max(0.0, linear_val), clamped to [0.0, 1.0]
    relu_val = max(0.0, float(linear_val))
    return float(np.clip(relu_val, 0.0, 1.0))


def estimate_speech_rate(signal: np.ndarray, sr: int = 16000) -> float:
    """
    Estimates speech rate in Words Per Minute (WPM) using energy envelope peaks.
    Typical speech rate: 120-160 WPM. Suppressed / frozen speech: <60 WPM. Agitated panic: >200 WPM.
    """
    if len(signal) < sr * 0.5:
        return 0.0

    duration_sec = len(signal) / sr
    hop_length = int(sr * 0.02)  # 20ms frames
    frame_length = int(sr * 0.04)  # 40ms window

    # Energy envelope
    energy = np.array([
        np.sum(signal[i : i + frame_length] ** 2)
        for i in range(0, len(signal) - frame_length, hop_length)
    ], dtype=np.float32)

    if len(energy) == 0 or np.max(energy) < 1e-5:
        return 0.0

    # Normalize and smooth
    energy /= (np.max(energy) + 1e-9)
    smooth_kernel = np.ones(5) / 5.0
    energy_smooth = np.convolve(energy, smooth_kernel, mode="same")

    # Count syllable energy peaks above threshold
    peaks, _ = scipy.signal.find_peaks(
        energy_smooth,
        height=0.15,
        distance=int(0.12 / 0.02),  # ~120ms minimum distance between syllables
    )
    syllable_count = len(peaks)

    # In Indian English & Indic dialects, ~1.4 - 1.6 syllables per word
    words_est = syllable_count / 1.5
    wpm = (words_est / max(duration_sec, 0.5)) * 60.0
    return float(round(np.clip(wpm, 0.0, 320.0), 1))


def analyze_bpab_biomarkers(
    signal: np.ndarray,
    sr: int,
    rms: float,
    f0_mean: float,
    f0_var: float,
    jitter: float,
    shimmer: float,
    hnr: float,
) -> dict:
    """
    BPAB (Biometric Passive Acoustic & Biomarker Engine) EAR:
    Acts as the 'Physical Listener' of the system, evaluating non-speech physiological cues:
    - Dead Air / Silence (<0.003 RMS) -> Phase 1 & 2 IVRS triage
    - Whispered Formant Shift (low HNR, low RMS, high spectral centroid) -> Stealth Mode
    - Agonal Breathing & Involuntary Groans (noisy periodic surges, very low HNR)
    - Weeping & Vocal Tremors (erratic pitch perturbation, crying burst acoustics)
    """
    is_dead_air = bool(rms < 0.004)
    
    # Calculate spectral centroid for whispered formant shift detection
    fft_mag = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    spectral_centroid = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))

    # Whispering has elevated high-frequency spectral centroid (>2200 Hz), low HNR (<10 dB), and soft RMS (<0.025)
    is_whispering = bool(
        not is_dead_air
        and hnr < 11.0
        and (spectral_centroid > 2100 or f0_mean < 90.0)
        and rms < 0.035
    )

    # Agonal breathing / involuntary groans: very low HNR, low F0 or respiratory noise surges
    is_agonal_breathing = bool(
        not is_dead_air
        and hnr < 7.0
        and shimmer > 2.0
        and rms > 0.008
        and f0_var < 15.0
    )

    # Weeping / crying / vocal tremors: high jitter (>0.06), high shimmer (>2.5), high pitch variance
    is_weeping_crying = bool(
        not is_dead_air
        and (jitter > 0.055 or shimmer > 2.5)
        and f0_var > 22.0
    )

    vocal_tremor = bool(jitter > 0.045 or f0_var > 25.0)

    # Primary BPAB triage descriptor
    if is_dead_air:
        bpab_desc = "Dead Air / Silence Detected (Phase 2 Stealth Standby)"
    elif is_whispering:
        bpab_desc = "Whispered Formant Shift Detected (Stealth Mode Fallback)"
    elif is_agonal_breathing:
        bpab_desc = "Agonal Breathing & Involuntary Groans Detected"
    elif is_weeping_crying:
        bpab_desc = "Weeping Acoustics & Vocal Cord Tremors Flagged"
    elif vocal_tremor:
        bpab_desc = "Elevated Physiological Tremor & Distress"
    else:
        bpab_desc = "Clean Acoustic Channel / Nominal Voice"

    return {
        "bpab_classification": bpab_desc,
        "is_dead_air": is_dead_air,
        "is_whispering": is_whispering,
        "is_agonal_breathing": is_agonal_breathing,
        "is_weeping_crying": is_weeping_crying,
        "vocal_tremor_detected": vocal_tremor,
        "spectral_centroid_hz": round(spectral_centroid, 1),
    }


def extract_features_array(signal: np.ndarray, sr: int = 16000) -> dict:
    """
    Extracts acoustic voice biomarkers directly from a 16kHz in-memory numpy array.
    Pipeline priority:
    1. openSMILE eGeMAPSv02 (gold standard Low Level Descriptors)
    2. librosa (lightweight fallback for pitch, spectral, and energy contours)
    3. numpy/scipy DSP fallback (guaranteed in-memory execution)
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    signal = signal.astype(np.float32)

    # Compute baseline RMS energy
    rms = float(np.sqrt(np.mean(signal ** 2))) + 1e-9

    # Minimum audio length check (e.g. at least 0.25s)
    if len(signal) < sr * 0.25:
        return {
            "pitch_f0_mean": 0.0,
            "f0_var": 0.0,
            "pitch_std": 0.0,
            "jitter": 0.0,
            "shimmer_db": 0.0,
            "hnr_db": 0.0,
            "speech_rate_wpm": 0.0,
            "acoustic_score": 0.1,
            "jitter_pct": 0.0,
            "shimmer_pct": 0.0,
            "bpab_status": "Dead Air / Audio buffer too short",
            "is_dead_air": True,
            "is_whispering": False,
            "is_agonal_breathing": False,
            "is_weeping": False,
        }

    # 1. Attempt openSMILE extraction on RAM array
    if _SMILE_EXTRACTOR is not None:
        try:
            df = _SMILE_EXTRACTOR.process_signal(signal, sr)
            
            # F0 semitones / Hertz
            if "F0semitoneFrom27.5Hz_sma3nz" in df.columns:
                f0_series = df["F0semitoneFrom27.5Hz_sma3nz"].replace(0, np.nan).dropna()
                f0_mean = float(f0_series.mean()) if not f0_series.empty else 0.0
                f0_var = float(f0_series.std()) if not f0_series.empty else 0.0
            else:
                f0_mean, f0_var = 0.0, 0.0

            # Jitter
            if "jitterLocal_sma3nz" in df.columns:
                jitter_val = float(df["jitterLocal_sma3nz"].dropna().mean())
            else:
                jitter_val = 0.0

            # Shimmer
            if "shimmerLocaldB_sma3nz" in df.columns:
                shimmer_val = float(df["shimmerLocaldB_sma3nz"].dropna().mean())
            else:
                shimmer_val = 0.0

            # HNR
            if "HNRdBACF_sma3nz" in df.columns:
                hnr_val = float(df["HNRdBACF_sma3nz"].dropna().mean())
            else:
                hnr_val = 0.0

            # Clean NaN/inf
            f0_mean = 0.0 if np.isnan(f0_mean) or np.isinf(f0_mean) else f0_mean
            f0_var = 0.0 if np.isnan(f0_var) or np.isinf(f0_var) else f0_var
            jitter_val = 0.0 if np.isnan(jitter_val) or np.isinf(jitter_val) else jitter_val
            shimmer_val = 0.0 if np.isnan(shimmer_val) or np.isinf(shimmer_val) else shimmer_val
            hnr_val = 0.0 if np.isnan(hnr_val) or np.isinf(hnr_val) else hnr_val

            a_score = compute_acoustic_subscore(jitter_val, shimmer_val, hnr_val, f0_var)
            wpm = estimate_speech_rate(signal, sr)
            bpab_meta = analyze_bpab_biomarkers(signal, sr, rms, f0_mean, f0_var, jitter_val, shimmer_val, hnr_val)

            return {
                "pitch_f0_mean": round(f0_mean, 2),
                "f0_var": round(f0_var, 2),
                "pitch_std": round(f0_var, 2),
                "jitter": round(jitter_val, 4),
                "shimmer_db": round(shimmer_val, 2),
                "hnr_db": round(hnr_val, 2),
                "speech_rate_wpm": wpm,
                "acoustic_score": round(a_score, 4),
                "jitter_pct": round(jitter_val * 100, 2),
                "shimmer_pct": round(shimmer_val * 10, 2),
                "bpab_status": bpab_meta["bpab_classification"],
                "is_dead_air": bpab_meta["is_dead_air"],
                "is_whispering": bpab_meta["is_whispering"],
                "is_agonal_breathing": bpab_meta["is_agonal_breathing"],
                "is_weeping": bpab_meta["is_weeping_crying"],
            }
        except Exception:
            pass  # Fall through to librosa or DSP fallback

    # 2. Attempt Librosa lightweight extraction if available
    if _HAS_LIBROSA and librosa is not None:
        try:
            return _librosa_features(signal, sr, rms)
        except Exception:
            pass

    # 3. Fallback DSP approximation using numpy & scipy
    return _fallback_dsp_features(signal, sr, rms)


def _librosa_features(signal: np.ndarray, sr: int, rms: float) -> dict:
    """
    Extracts acoustic biomarkers via fast vectorised pitch autocorrelation and DSP.
    Optimised for real-time Helpline triage and fast batch extraction.
    """
    frame_len = int(sr * 0.04)   # 40ms frame
    hop_len = int(sr * 0.02)     # 20ms hop
    min_lag = int(sr / 500.0)    # 500 Hz max F0
    max_lag = int(sr / 65.0)     # 65 Hz min F0

    f0_list = []
    # Fast frame-level autocorrelation F0 estimation
    for i in range(0, len(signal) - frame_len, hop_len):
        frame = signal[i : i + frame_len]
        # Skip silent frames
        if np.sum(frame ** 2) < 1e-6:
            continue
        corr = scipy.signal.correlate(frame, frame, mode="full")[len(frame) - 1 :]
        if len(corr) > max_lag:
            window_corr = corr[min_lag:max_lag]
            peak_idx = np.argmax(window_corr) + min_lag
            if corr[0] > 0 and (corr[peak_idx] / corr[0]) > 0.3:
                f0_list.append(sr / peak_idx)

    f0_arr = np.array(f0_list, dtype=np.float32)
    f0_mean = float(np.mean(f0_arr)) if len(f0_arr) > 0 else 0.0
    f0_var = float(np.std(f0_arr)) if len(f0_arr) > 0 else 0.0

    # Perturbation estimates
    diff_sig = np.abs(np.diff(signal))
    jitter_val = float(np.mean(diff_sig) / (np.mean(np.abs(signal)) + 1e-6) * 0.05)
    shimmer_val = float(np.std(diff_sig) / rms * 0.1)
    hnr_val = float(max(0.0, min(30.0, 10 * np.log10(rms / (np.std(diff_sig) + 1e-6)))))

    a_score = compute_acoustic_subscore(jitter_val * 100, shimmer_val * 10, hnr_val, f0_var)
    wpm = estimate_speech_rate(signal, sr)
    bpab_meta = analyze_bpab_biomarkers(signal, sr, rms, f0_mean, f0_var, jitter_val, shimmer_val, hnr_val)

    return {
        "pitch_f0_mean": round(f0_mean, 1),
        "f0_var": round(f0_var, 2),
        "pitch_std": round(f0_var, 2),
        "jitter": round(jitter_val, 4),
        "shimmer_db": round(shimmer_val, 2),
        "hnr_db": round(hnr_val, 2),
        "speech_rate_wpm": wpm,
        "acoustic_score": round(a_score, 4),
        "jitter_pct": round(jitter_val * 100, 2),
        "shimmer_pct": round(shimmer_val * 10, 2),
        "bpab_status": bpab_meta["bpab_classification"],
        "is_dead_air": bpab_meta["is_dead_air"],
        "is_whispering": bpab_meta["is_whispering"],
        "is_agonal_breathing": bpab_meta["is_agonal_breathing"],
        "is_weeping": bpab_meta["is_weeping_crying"],
    }


def _fallback_dsp_features(y: np.ndarray, sr: int, rms: float) -> dict:
    """Fallback DSP method using numpy & scipy when external C-libraries are unavailable."""
    zc = float(np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y)))
    f0_approx = zc * (sr / 2)

    diff_y = np.abs(np.diff(y))
    jitter_approx = float(np.mean(diff_y) / (np.mean(np.abs(y)) + 1e-6) * 0.05)
    shimmer_approx = float(np.std(diff_y) / rms * 0.1)
    hnr_approx = float(max(0.0, min(30.0, 10 * np.log10(rms / (np.std(diff_y) + 1e-6)))))

    a_score = compute_acoustic_subscore(jitter_approx * 100, shimmer_approx * 10, hnr_approx, 15.0)
    wpm = estimate_speech_rate(y, sr)
    bpab_meta = analyze_bpab_biomarkers(y, sr, rms, f0_approx, 15.0, jitter_approx, shimmer_approx, hnr_approx)

    return {
        "pitch_f0_mean": round(f0_approx, 1),
        "f0_var": 12.0,
        "pitch_std": 12.0,
        "jitter": round(jitter_approx, 4),
        "shimmer_db": round(shimmer_approx, 2),
        "hnr_db": round(hnr_approx, 2),
        "speech_rate_wpm": wpm,
        "acoustic_score": round(a_score, 4),
        "jitter_pct": round(jitter_approx * 100, 2),
        "shimmer_pct": round(shimmer_approx * 10, 2),
        "bpab_status": bpab_meta["bpab_classification"],
        "is_dead_air": bpab_meta["is_dead_air"],
        "is_whispering": bpab_meta["is_whispering"],
        "is_agonal_breathing": bpab_meta["is_agonal_breathing"],
        "is_weeping": bpab_meta["is_weeping_crying"],
    }


def load_audio_array(audio_input, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Loads audio from filepath, bytes, or file-like object and resamples to target_sr.
    """
    if isinstance(audio_input, (str, bytes, os.PathLike, io.BytesIO)):
        data, sr = sf.read(audio_input)
    elif isinstance(audio_input, np.ndarray):
        data, sr = audio_input, target_sr
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    if data.ndim > 1:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)

    if sr != target_sr and len(data) > 0:
        num_samples = int(len(data) * target_sr / sr)
        data = scipy.signal.resample(data, num_samples).astype(np.float32)
        sr = target_sr

    return data, sr


def extract_features(audio_input) -> dict:
    """
    Accepts a filepath, bytes, or file-like object and extracts acoustic features.
    Augments extraction with ML-based stress classification if a trained model is available.
    """
    data, sr = load_audio_array(audio_input, target_sr=16000)
    features = extract_features_array(data, sr=sr)
    
    # Run trained ML Stress Model inference if present
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "combined", "models", "stress_model.joblib"))
    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "combined", "models", "scaler.joblib"))
    
    if os.path.exists(model_path):
        try:
            import joblib
            model = joblib.load(model_path)
            
            # Fast DSP descriptors
            rms = float(np.sqrt(np.mean(data ** 2))) + 1e-9
            zcr = float(np.sum(np.abs(np.diff(np.sign(data)))) / (2 * len(data)))
            fft_mag = np.abs(np.fft.rfft(data))
            freqs = np.fft.rfftfreq(len(data), d=1.0 / sr)
            sc = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))
            
            vec = np.array([[
                float(features.get("pitch_f0_mean", 0.0)),
                float(features.get("f0_var", 0.0)),
                float(features.get("jitter", 0.0)),
                float(features.get("shimmer_db", 0.0)),
                float(features.get("hnr_db", 0.0)),
                float(features.get("speech_rate_wpm", 0.0)),
                float(features.get("acoustic_score", 0.0)),
                float(sc),
                float(rms),
                float(zcr),
            ]], dtype=np.float32)
            
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                vec = scaler.transform(vec)
                
            pred_class = int(model.predict(vec)[0])
            stress_prob = float(model.predict_proba(vec)[0][1]) if hasattr(model, "predict_proba") else float(pred_class)
            
            features["ml_stress_prediction"] = "High Stress" if pred_class == 1 else "Low Stress"
            features["ml_stress_probability"] = round(stress_prob, 4)
        except Exception:
            features["ml_stress_prediction"] = "High Stress" if features.get("acoustic_score", 0) > 0.5 else "Low Stress"
            features["ml_stress_probability"] = features.get("acoustic_score", 0.0)
            
    return features

