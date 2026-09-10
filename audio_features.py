"""
Acoustic feature extraction using openSMILE eGeMAPSv02 (Low Level Descriptors).

Extracts voice biomarkers directly from 16kHz RAM arrays in <50ms:
- Fundamental Frequency (F0 semitones)
- Jitter (jitterLocal_sma3nz) for vocal cord trembling
- Shimmer (shimmerLocaldB_sma3nz) for sobbing / breathlessness
- Harmonics-to-Noise Ratio (HNR) for whispering / gasping detection
- Normalized Acoustic Sub-Score: A(t) in [0, 1]
"""
import io
import os
import numpy as np
import soundfile as sf

# Try importing opensmile, with a resilient fallback if not installed yet
try:
    import opensmile
    _SMILE_EXTRACTOR = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
except Exception:
    _SMILE_EXTRACTOR = None


def compute_acoustic_subscore(
    jitter: float,
    shimmer: float,
    hnr: float,
    f0_var: float,
) -> float:
    """
    Computes normalized Acoustic Sub-Score A(t) in [0.0, 1.0].
    Higher jitter, shimmer, pitch perturbation, and lower HNR indicate heightened distress.
    """
    # Normalized components (clamped 0 to 1)
    jitter_n = min(max(jitter / 4.0, 0.0), 1.0)       # typical range 0-4%
    shimmer_n = min(max(shimmer / 8.0, 0.0), 1.0)     # typical range 0-8% or dB scale
    hnr_n = 1.0 - min(max(hnr / 24.0, 0.0), 1.0)      # lower HNR = more noise/gasping = higher score
    f0_n = min(max(f0_var / 40.0, 0.0), 1.0)          # typical range 0-40 Hz variance

    a_t = 0.30 * jitter_n + 0.30 * shimmer_n + 0.25 * hnr_n + 0.15 * f0_n
    return float(np.clip(a_t, 0.0, 1.0))


def extract_features_array(signal: np.ndarray, sr: int = 16000) -> dict:
    """
    Extracts acoustic voice biomarkers directly from a 16kHz in-memory numpy array.
    """
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)
    signal = signal.astype(np.float32)

    # Minimum audio length check (e.g. at least 0.25s)
    if len(signal) < sr * 0.25:
        return {
            "pitch_f0_mean": 0.0,
            "f0_var": 0.0,
            "jitter": 0.0,
            "shimmer_db": 0.0,
            "hnr_db": 0.0,
            "acoustic_score": 0.1,
            "jitter_pct": 0.0,
            "shimmer_pct": 0.0,
        }

    if _SMILE_EXTRACTOR is not None:
        try:
            # openSMILE extraction on RAM array
            df = _SMILE_EXTRACTOR.process_signal(signal, sr)
            
            # F0 semitones
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

            return {
                "pitch_f0_mean": round(f0_mean, 2),
                "f0_var": round(f0_var, 2),
                "jitter": round(jitter_val, 4),
                "shimmer_db": round(shimmer_val, 2),
                "hnr_db": round(hnr_val, 2),
                "acoustic_score": round(a_score, 4),
                "jitter_pct": round(jitter_val * 100, 2),
                "shimmer_pct": round(shimmer_val * 10, 2),
            }
        except Exception:
            pass  # Fall through to DSP fallback below

    # Fallback DSP approximation using numpy/soundfile
    return _fallback_dsp_features(signal, sr)


def _fallback_dsp_features(y: np.ndarray, sr: int) -> dict:
    """Fallback DSP method when openSMILE is unavailable."""
    # Zero crossings and RMS
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    zc = float(np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y)))
    f0_approx = zc * (sr / 2)

    # Simplified perturbation approximations
    diff_y = np.abs(np.diff(y))
    jitter_approx = float(np.mean(diff_y) / (np.mean(np.abs(y)) + 1e-6) * 0.05)
    shimmer_approx = float(np.std(diff_y) / rms * 0.1)
    hnr_approx = float(max(0.0, min(30.0, 10 * np.log10(rms / (np.std(diff_y) + 1e-6)))))

    a_score = compute_acoustic_subscore(jitter_approx * 100, shimmer_approx * 10, hnr_approx, 15.0)

    return {
        "pitch_f0_mean": round(f0_approx, 1),
        "f0_var": 12.0,
        "jitter": round(jitter_approx, 4),
        "shimmer_db": round(shimmer_approx, 2),
        "hnr_db": round(hnr_approx, 2),
        "acoustic_score": round(a_score, 4),
        "jitter_pct": round(jitter_approx * 100, 2),
        "shimmer_pct": round(shimmer_approx * 10, 2),
    }


def load_audio_array(audio_input, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """
    Loads audio from filepath, bytes, or file-like object and resamples to target_sr.
    """
    import scipy.signal
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
    """
    data, sr = load_audio_array(audio_input, target_sr=16000)
    return extract_features_array(data, sr=sr)

