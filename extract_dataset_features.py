"""
extract_dataset_features.py
Extracts acoustic feature vectors from all audio files listed in combined_labels.csv
using audio_features.py (F0, jitter, shimmer, HNR, speech rate, spectral characteristics, etc.).

Performs Speaker-Independent Split:
- Train (~70%)
- Validation (~15%)
- Test (~15%)

Saves:
- X_train.npy, y_train.npy
- X_val.npy, y_val.npy
- X_test.npy, y_test.npy
- feature_names.json
- split_metadata.csv
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import feature extractor from current directory
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import audio_features

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
COMBINED_DIR = os.path.join(DATA_DIR, "combined")
CSV_PATH = os.path.join(COMBINED_DIR, "combined_labels.csv")

FEATURE_NAMES = [
    "pitch_f0_mean",
    "f0_var",
    "jitter",
    "shimmer_db",
    "hnr_db",
    "speech_rate_wpm",
    "acoustic_score",
    "spectral_centroid",
    "rms_energy",
    "zero_crossing_rate",
]


def extract_vector(filepath: str) -> np.ndarray:
    try:
        sig, sr = audio_features.load_audio_array(filepath, target_sr=16000)
        if len(sig) == 0:
            return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

        feat_dict = audio_features.extract_features_array(sig, sr=sr)
        
        # Fast DSP descriptors
        rms = float(np.sqrt(np.mean(sig ** 2))) + 1e-9
        zcr = float(np.sum(np.abs(np.diff(np.sign(sig)))) / (2 * len(sig)))
        
        # Fast spectral centroid via FFT
        fft_mag = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1.0 / sr)
        sc = float(np.sum(freqs * fft_mag) / (np.sum(fft_mag) + 1e-9))

        vec = [
            float(feat_dict.get("pitch_f0_mean", 0.0)),
            float(feat_dict.get("f0_var", 0.0)),
            float(feat_dict.get("jitter", 0.0)),
            float(feat_dict.get("shimmer_db", 0.0)),
            float(feat_dict.get("hnr_db", 0.0)),
            float(feat_dict.get("speech_rate_wpm", 0.0)),
            float(feat_dict.get("acoustic_score", 0.0)),
            float(sc),
            float(rms),
            float(zcr),
        ]
        return np.nan_to_num(np.array(vec, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)


def main():
    print("==================================================")
    print("Acoustic Feature Extraction & Speaker Disjoint Split")
    print("==================================================")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} utterances across {df['speaker_id'].nunique()} speakers.")

    # Speaker-Independent Split:
    # Shuffle unique speakers and allocate 70% train, 15% val, 15% test
    unique_speakers = list(df["speaker_id"].unique())
    random.seed(42)
    random.shuffle(unique_speakers)

    n_speakers = len(unique_speakers)
    train_end = int(0.70 * n_speakers)
    val_end = int(0.85 * n_speakers)

    train_spks = set(unique_speakers[:train_end])
    val_spks = set(unique_speakers[train_end:val_end])
    test_spks = set(unique_speakers[val_end:])

    df["split"] = df["speaker_id"].apply(
        lambda spk: "train" if spk in train_spks else ("val" if spk in val_spks else "test")
    )

    print("\nSplit Distribution by Utterance:")
    print(df["split"].value_counts())
    print("\nStress Labels by Split:")
    print(pd.crosstab(df["split"], df["stress_label"]))

    # Batch extraction
    print("\nExtracting acoustic features with audio_features.py...")
    features = []
    valid_mask = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
        fpath = row["filepath"]
        vec = extract_vector(fpath)
        features.append(vec)
        # Check if audio was valid (non-all-zero)
        valid_mask.append(bool(np.any(vec != 0)))

    X_all = np.array(features, dtype=np.float32)
    df["is_valid"] = valid_mask

    # Save metadata
    df.to_csv(os.path.join(COMBINED_DIR, "split_metadata.csv"), index=False)

    train_idx = df[(df["split"] == "train") & df["is_valid"]].index
    val_idx = df[(df["split"] == "val") & df["is_valid"]].index
    test_idx = df[(df["split"] == "test") & df["is_valid"]].index

    X_train, y_train = X_all[train_idx], df.loc[train_idx, "stress_label"].to_numpy(dtype=np.int64)
    X_val, y_val = X_all[val_idx], df.loc[val_idx, "stress_label"].to_numpy(dtype=np.int64)
    X_test, y_test = X_all[test_idx], df.loc[test_idx, "stress_label"].to_numpy(dtype=np.int64)

    np.save(os.path.join(COMBINED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(COMBINED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(COMBINED_DIR, "X_val.npy"), X_val)
    np.save(os.path.join(COMBINED_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(COMBINED_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(COMBINED_DIR, "y_test.npy"), y_test)

    with open(os.path.join(COMBINED_DIR, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print("\nExtraction Complete!")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    print(f"Artifacts saved in: {COMBINED_DIR}")


if __name__ == "__main__":
    main()
