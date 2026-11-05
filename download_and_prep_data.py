"""
download_and_prep_data.py
Automated acquisition and preparation of multi-dataset audio files for RTSAM:
- RAVDESS (Zenodo direct zip)
- CREMA-D (Hugging Face / Git audio subset)
- SAVEE (Hugging Face parquet)

Normalizes file names, unifies stress labels, and generates combined_labels.csv.
Stress Mapping:
- Low-stress (0): neutral, calm, happy, sad
- High-stress (1): angry, fearful, disgust, frustrated, excited, surprised
"""
import os
import sys
import zipfile
import io
import urllib.request
import pandas as pd
import numpy as np
import soundfile as sf
from tqdm import tqdm

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
COMBINED_DIR = os.path.join(DATA_DIR, "combined")
AUDIO_COMBINED_DIR = os.path.join(COMBINED_DIR, "audio")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)
os.makedirs(AUDIO_COMBINED_DIR, exist_ok=True)

# Emotion to Stress Label Mapping
EMOTION_TO_STRESS = {
    # Low-stress (0)
    "neutral": 0,
    "calm": 0,
    "happy": 0,
    "sad": 0,
    "sadness": 0,
    "joy": 0,
    # High-stress (1)
    "angry": 1,
    "anger": 1,
    "fearful": 1,
    "fear": 1,
    "disgust": 1,
    "frustrated": 1,
    "frustration": 1,
    "surprised": 1,
    "surprise": 1,
    "excited": 1,
}

RAVDESS_EMO_CODE = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

CREMA_EMO_CODE = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def download_file(url: str, dest_path: str, desc: str = "Downloading"):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"File already exists: {dest_path}")
        return
    print(f"{desc} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as response:
        total_size = int(response.info().get("Content-Length", 0))
        chunk_size = 1024 * 512
        with open(dest_path, "wb") as f, tqdm(
            desc=desc,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    print(f"Saved: {dest_path}")


def prepare_ravdess() -> list[dict]:
    print("\n--- Processing RAVDESS Dataset ---")
    zip_path = os.path.join(RAW_DIR, "Audio_Speech_Actors_01-24.zip")
    ravdess_extract_dir = os.path.join(RAW_DIR, "ravdess")
    ravdess_url = "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip"

    download_file(ravdess_url, zip_path, desc="Downloading RAVDESS")

    if not os.path.exists(ravdess_extract_dir):
        print(f"Extracting RAVDESS to {ravdess_extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(ravdess_extract_dir)

    records = []
    # Scan through extracted files
    for root, _, files in os.walk(ravdess_extract_dir):
        for f in files:
            if f.endswith(".wav"):
                parts = f.replace(".wav", "").split("-")
                if len(parts) >= 7:
                    emo_num = parts[2]
                    speaker_id = f"ravdess_spk_{parts[6]}"
                    orig_emotion = RAVDESS_EMO_CODE.get(emo_num, "unknown")
                    if orig_emotion in EMOTION_TO_STRESS:
                        src_path = os.path.join(root, f)
                        dest_fname = f"ravdess_{parts[6]}_{parts[2]}_{parts[1]}_{parts[0]}_{f}"
                        dest_path = os.path.join(AUDIO_COMBINED_DIR, dest_fname)
                        if not os.path.exists(dest_path):
                            # Copy or link
                            data, sr = sf.read(src_path)
                            sf.write(dest_path, data, sr)

                        records.append({
                            "filepath": os.path.abspath(dest_path),
                            "filename": dest_fname,
                            "dataset": "RAVDESS",
                            "speaker_id": speaker_id,
                            "orig_emotion": orig_emotion,
                            "stress_label": EMOTION_TO_STRESS[orig_emotion],
                        })

    print(f"RAVDESS utterances processed: {len(records)}")
    return records


def prepare_savee() -> list[dict]:
    print("\n--- Processing SAVEE Dataset ---")
    savee_url = "https://huggingface.co/datasets/AbstractTTS/SAVEE/resolve/main/data/train-00000-of-00001.parquet"
    savee_parquet = os.path.join(RAW_DIR, "savee_train.parquet")
    download_file(savee_url, savee_parquet, desc="Downloading SAVEE Parquet")

    df = pd.read_parquet(savee_parquet)
    records = []
    print(f"Extracting {len(df)} audio records from SAVEE...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="SAVEE Audio"):
        orig_emo = str(row.get("emotion", "")).lower()
        if orig_emo not in EMOTION_TO_STRESS:
            continue
        
        audio_dict = row.get("audio", {})
        audio_bytes = audio_dict.get("bytes")
        sr = audio_dict.get("sampling_rate", 16000)
        file_id = str(row.get("file", f"savee_{idx}.wav")).replace(".wav", "")
        # Speaker info from file name e.g. DC_a01 -> DC
        spk = file_id.split("_")[0] if "_" in file_id else "savee_spk"
        speaker_id = f"savee_{spk}"

        dest_fname = f"savee_{file_id}.wav"
        dest_path = os.path.join(AUDIO_COMBINED_DIR, dest_fname)

        if not os.path.exists(dest_path) and audio_bytes:
            data, cur_sr = sf.read(io.BytesIO(audio_bytes))
            sf.write(dest_path, data, cur_sr)

        records.append({
            "filepath": os.path.abspath(dest_path),
            "filename": dest_fname,
            "dataset": "SAVEE",
            "speaker_id": speaker_id,
            "orig_emotion": orig_emo,
            "stress_label": EMOTION_TO_STRESS[orig_emo],
        })

    print(f"SAVEE utterances processed: {len(records)}")
    return records


def prepare_crema_d(max_shards: int = 2) -> list[dict]:
    print("\n--- Processing CREMA-D Dataset ---")
    base_url = "https://huggingface.co/datasets/cfahlgren1/crema-d/resolve/main/data"
    records = []
    
    # Process shards to assemble 1,500+ diverse utterances
    for i in range(max_shards):
        shard_name = f"train-0000{i}-of-00004.parquet"
        shard_url = f"{base_url}/{shard_name}"
        shard_dest = os.path.join(RAW_DIR, f"crema_d_{shard_name}")
        download_file(shard_url, shard_dest, desc=f"Downloading CREMA-D Shard {i+1}")

        df = pd.read_parquet(shard_dest)
        print(f"Extracting audio from CREMA-D Shard {i+1} ({len(df)} samples)...")
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"CREMA-D Shard {i}"):
            emo_code = str(row.get("emotion_code", "")).upper()
            orig_emo = CREMA_EMO_CODE.get(emo_code, str(row.get("label", "")).lower())
            
            if orig_emo not in EMOTION_TO_STRESS:
                continue

            actor_id = str(row.get("actor_id", f"spk_{idx}"))
            speaker_id = f"crema_{actor_id}"
            
            audio_dict = row.get("audio", {})
            audio_bytes = audio_dict.get("bytes") if isinstance(audio_dict, dict) else None
            
            dest_fname = f"crema_{actor_id}_{emo_code}_{idx}.wav"
            dest_path = os.path.join(AUDIO_COMBINED_DIR, dest_fname)

            if not os.path.exists(dest_path) and audio_bytes:
                data, cur_sr = sf.read(io.BytesIO(audio_bytes))
                sf.write(dest_path, data, cur_sr)

            records.append({
                "filepath": os.path.abspath(dest_path),
                "filename": dest_fname,
                "dataset": "CREMA-D",
                "speaker_id": speaker_id,
                "orig_emotion": orig_emo,
                "stress_label": EMOTION_TO_STRESS[orig_emo],
            })

    print(f"CREMA-D utterances processed: {len(records)}")
    return records


def main():
    print("==================================================")
    print("Multi-Dataset Voice Stress Pipeline Acquisition")
    print("==================================================")

    all_records = []
    # 1. RAVDESS
    try:
        rav_records = prepare_ravdess()
        all_records.extend(rav_records)
    except Exception as e:
        print(f"Error in RAVDESS processing: {e}")

    # 2. SAVEE
    try:
        sav_records = prepare_savee()
        all_records.extend(sav_records)
    except Exception as e:
        print(f"Error in SAVEE processing: {e}")

    # 3. CREMA-D
    try:
        crema_records = prepare_crema_d(max_shards=2)
        all_records.extend(crema_records)
    except Exception as e:
        print(f"Error in CREMA-D processing: {e}")

    combined_df = pd.DataFrame(all_records)
    csv_path = os.path.join(COMBINED_DIR, "combined_labels.csv")
    combined_df.to_csv(csv_path, index=False)

    print("\n================ SUMMARY ================")
    print(f"Total Combined Utterances: {len(combined_df)}")
    print("Dataset distribution:")
    print(combined_df["dataset"].value_counts())
    print("\nStress label distribution (0=Low Stress, 1=High Stress):")
    print(combined_df["stress_label"].value_counts())
    print("\nTotal Unique Speakers:", combined_df["speaker_id"].nunique())
    print(f"Single Source of Truth Saved: {csv_path}")


if __name__ == "__main__":
    main()

