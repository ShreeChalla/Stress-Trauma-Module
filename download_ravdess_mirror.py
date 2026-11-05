"""
download_ravdess_mirror.py
Pulls RAVDESS from the viks66/ravdess_speech mirror on Hugging Face
and integrates it with the existing combined_labels.csv.
"""
import os
import zipfile
import urllib.request
import pandas as pd
import soundfile as sf
from tqdm import tqdm

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAW_DIR = os.path.join(DATA_DIR, "raw")
COMBINED_DIR = os.path.join(DATA_DIR, "combined")
AUDIO_COMBINED_DIR = os.path.join(COMBINED_DIR, "audio")

EMOTION_TO_STRESS = {
    "neutral": 0, "calm": 0, "happy": 0, "sad": 0,
    "angry": 1, "fearful": 1, "disgust": 1, "surprised": 1
}

RAVDESS_EMO_CODE = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

def main():
    ravdess_url = "https://huggingface.co/datasets/viks66/ravdess_speech/resolve/main/ravdess.zip"
    zip_path = os.path.join(RAW_DIR, "ravdess_mirror.zip")
    extract_dir = os.path.join(RAW_DIR, "ravdess")

    if not os.path.exists(zip_path) or os.path.getsize(zip_path) < 1000000:
        print("Downloading RAVDESS mirror from Hugging Face...")
        req = urllib.request.Request(ravdess_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as response:
            total_size = int(response.info().get("Content-Length", 0))
            chunk_size = 1024 * 512
            with open(zip_path, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024, desc="RAVDESS"
            ) as bar:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    bar.update(len(chunk))

    if not os.path.exists(extract_dir):
        print(f"Extracting RAVDESS to {extract_dir}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

    # Process files
    records = []
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f.endswith(".wav"):
                parts = f.replace(".wav", "").split("-")
                if len(parts) >= 7:
                    emo_num = parts[2]
                    orig_emotion = RAVDESS_EMO_CODE.get(emo_num, "unknown")
                    if orig_emotion in EMOTION_TO_STRESS:
                        speaker_id = f"ravdess_{parts[6]}"
                        dest_fname = f"ravdess_spk{parts[6]}_{parts[2]}_{f}"
                        dest_path = os.path.join(AUDIO_COMBINED_DIR, dest_fname)
                        src_path = os.path.join(root, f)
                        if not os.path.exists(dest_path):
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

    print(f"RAVDESS records prepared: {len(records)}")
    
    # Merge with existing combined_labels.csv
    csv_path = os.path.join(COMBINED_DIR, "combined_labels.csv")
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, pd.DataFrame(records)], ignore_index=True)
    else:
        combined_df = pd.DataFrame(records)

    # Deduplicate by filename
    combined_df = combined_df.drop_duplicates(subset=["filename"]).reset_index(drop=True)
    combined_df.to_csv(csv_path, index=False)

    print("\n================ UPDATED MULTI-DATASET SUMMARY ================")
    print(f"Total Combined Utterances: {len(combined_df)}")
    print(combined_df["dataset"].value_counts())
    print("\nStress label distribution:")
    print(combined_df["stress_label"].value_counts())
    print("Unique speakers:", combined_df["speaker_id"].nunique())
    print(f"Updated CSV: {csv_path}")

if __name__ == "__main__":
    main()

