# Maximising Accuracy – 3‑Dataset Blend & Workflow

Below is a **road‑map** that keeps the project lightweight (still using Classic ML) while giving you the **largest, most varied acoustic training set** you can feasibly assemble today.

### 1️⃣ Recommended Triple‑Dataset Mix  
| Dataset | Approx. # utterances (speech) | Stress‑relevant emotions | Why it boosts accuracy |
|--------|------------------------------|--------------------------|------------------------|
| **RAVDESS** | 735 | Calm/Neutral → low, Angry/Fearful/Disgust → high | Clean studio recordings → low‑noise baseline features. |
| **IEMOCAP** | ~12 k | Angry/Frustrated/Fearful → high, Neutral/Happy → low | Conversational style = better generalisation to real‑world voices; many speakers. |
| **MELD** | ~13 k (audio‑only subset) | Anger/Disgust/Fear → high, Neutral → low | TV‑show dialogue = varied background noise, different microphones, and a lot of speakers → robustness. |

**Total** ≈ 26 k utterances – well above the **1 k** threshold for a reliable Classic ML model and enough variety to avoid over‑fitting on any single recording condition.

### 2️⃣ Unified Label Scheme (Binary Stress)

| Original emotion | Mapped stress label |
|------------------|----------------------|
| **Low‑stress** | `neutral`, `calm`, `happy`, `sad` (low‑arousal) |
| **High‑stress** | `angry`, `fearful`, `disgust`, `frustrated`, `excited`, `surprised` (high‑arousal) |

*Why “sad” goes low*: In most affect‑recognition corpora sadness is low‑arousal, which correlates with **relaxed** rather than **stressed** physiology.  
*Why “surprised” goes high*: Surprise in the datasets tends to be abrupt and spikes pitch/energy – a good acoustic proxy for sudden stress.

> **Tip:** Keep a CSV (e.g., `combined_labels.csv`) with three columns: `filepath`, `orig_emotion`, `stress_label`. This file becomes the single source of truth for later splits.

### 3️⃣ Data‑Preparation Checklist (no code, just steps)

| Step | Action | Key Considerations |
|------|--------|--------------------|
| **A. Download** | ‑ RAVDESS → `Audio_Speech_Actors_01-24` <br>‑ IEMOCAP → `Session*` folder (use only the **audio** (`*.wav`) files) <br>‑ MELD → `audio` sub‑folder (approx. 13 k wav files) | Verify you have enough free disk space (≈ 5‑7 GB). |
| **B. Normalise file names** | Rename or create symbolic links so every file has a **unique, collision‑free** name (e.g., `ravdess_001.wav`, `iemocap_0045.wav`, `meld_12000.wav`). | Avoid duplicate basenames when you later glob the directory. |
| **C. Build the label CSV** | For each audio file, write a row: `filepath,orig_emotion,stress_label`. Use the official label files supplied with each dataset (RAVDESS `emotion_labels.txt`, IEMOCAP `session*_dialogue.txt`, MELD `metadata.csv`). | Make sure the mapping from dataset‑specific labels to the unified stress label follows the table above. |
| **D. Balance the classes (optional but helpful)** | Compute the ratio of low‑/high‑stress rows. If one class > 1.5× the other, **undersample** the majority (randomly drop rows) or **oversample** the minority (duplicate rows). | Classic ML (Random Forest, Gradient Boosting) is fairly robust, but a roughly balanced set gives a higher macro‑F1. |
| **E. Split speaker‑independently** | – Identify speaker IDs from each dataset (`01`, `02` … for RAVDESS; `session1/speaker1` for IEMOCAP; `speaker_id` column in MELD). <br>– Assign entire speakers to **train / val / test** (e.g., 70 % train, 15 % val, 15 % test). | Guarantees that the model never sees the same voice in training and testing, a must for a **real‑world stress detector**. |
| **F. Extract acoustic vectors** | Run the existing `audio_features.py` on every file in the **train**, **val**, and **test** splits. <br> – If the script already returns a fixed‑length vector (e.g., mean + std of each LLD), store them as `X_train.npy`, `X_val.npy`, `X_test.npy`. <br>– Save corresponding label arrays (`y_train.npy` etc.). | The extractor already handles openSMILE & librosa; you only need to loop over the file list. |
| **G. Persist the artefacts** | Put the `.npy` files and the label CSV inside a folder `data/combined/` under the project root. Add a short `README.md` describing the source URLs, licences, and the mapping table. | Future collaborators can regenerate the features with a single command. |

### 4️⃣ Model Choice & Hyper‑Parameter Focus (Classic ML)

| Model | Why it works well on this data | Typical “quick‑tune” knobs |
|-------|-------------------------------|----------------------------|
| **Random Forest** | Handles heterogeneous acoustic features, tolerant to noisy dimensions, gives built‑in feature importance. | `n_estimators` (200‑500), `max_depth` (None or 15‑30), `min_samples_leaf` (1‑5). |
| **Gradient Boosting (e.g., XGBoost / LightGBM)** | Often pushes accuracy a notch higher on tabular data; can capture subtle interactions between pitch‑jitter and HNR. | `learning_rate` (0.05‑0.2), `n_estimators` (300‑800), `max_depth` (6‑12). |
| **Linear SVM (RBF kernel)** | Good baseline when the classes are linearly separable after scaling; fast to train on 26 k rows. | `C` (0.1‑10), kernel `rbf`, `gamma` (`scale` or `auto`). |

*Recommendation*: **Start with a Random Forest** (fast, interpretable). If you need a few extra points, run a **light Gradient Boosting** sweep (only a few hundred trees) – the extra training time is still under a minute on a typical laptop.

### 5️⃣ Evaluation Blueprint (no code)

1. **Predict on the test set** → obtain binary predictions.  
2. Compute:  
   - **Accuracy** (overall correct rate).  
   - **Macro‑averaged Precision, Recall, F1** (treat low‑ and high‑stress equally).  
   - **Confusion matrix** (to see whether the model confuses low with high stress).  
   - **ROC‑AUC** ( gives a sense of ranking quality).  
3. **Feature importance** (Random Forest) → plot the top 10 acoustic descriptors (e.g., pitch‑std, jitter, shimmer, HNR). This tells you which acoustic cues drive the stress decision.  
4. **Error analysis**: look at a handful of mis‑classified audio files (listen to them) to see whether background noise or speaker idiosyncrasies are responsible. This can guide future data cleaning or augmentation.

### 6️⃣ Expected Accuracy Gains

| Model | Approx. test‑set macro‑F1 (based on prior studies with similar data) |
|-------|-----------------------------------------------------------------------|
| Random Forest (baseline) | **0.78 – 0.82** |
| Gradient Boosting (tuned) | **0.83 – 0.87** |
| Deep Learning (CNN on spectrograms) | **0.86 – 0.90** (requires > 5 k labelled recordings and a GPU) |

Because you are using **≈ 26 k** recordings, the **Gradient Boosting** variant is likely to push you into the **high‑80 % F1** range without the overhead of a deep network. If after you run the baseline you still crave a few more points, you can later prototype a **light CNN** on MFCC images; the data you already have will be sufficient.

### 7️⃣ Quick “What‑If” Extensions (future‑proof)

| Extension | What it adds | When to try it |
|-----------|--------------|----------------|
| **Data augmentation** (speed‑perturbation, noise addition) | Increases robustness to unseen recording conditions. | If validation loss is significantly lower than test loss. |
| **Speaker‑level cross‑validation** | Guarantees truly speaker‑independent performance. | If you suspect over‑fitting to a few dominant speakers. |
| **Hybrid text‑+‑acoustic** | Concatenate ASR‑generated embeddings (e.g., Whisper transcripts → BERT) with acoustic vectors. | When you have access to a reliable ASR and want to push > 0.90 F1. |
| **Ensemble** (average predictions of RF + GBM) | Often yields a modest bump (≈ 1‑2 % F1). | After both models are trained and you have time to evaluate ensemble. |

### 8️⃣ Summary of the Full Pipeline (high‑level checklist)

1. **Download** RAVDESS, IEMOCAP, MELD.  
2. **Rename / de‑duplicate** files, create `combined_labels.csv`.  
3. **Map** original emotions → binary stress label (low / high).  
4. **Balance** classes if needed.  
5. **Split** speakers → train / val / test.  
6. **Run** `audio_features.py` on every file → `X_*.npy`, `y_*.npy`.  
7. **Train** a Random Forest (quick) → evaluate.  
8. **Optional**: Grid‑search a Gradient Boosting model for a final boost.  
9. **Report**: accuracy, macro‑F1, confusion matrix, top feature importances.

---

### Next Action

Let me know:
- **Do you want to go ahead with the 3‑dataset combination (RAVDESS + IEMOCAP + MELD) as outlined?**  
- **Do you need any help locating the download URLs or clarifying the licensing?**  
- **Would you like a short checklist file (`data_preparation_checklist.md`) created in the repo to track each of the steps above?**

Once you confirm, we can flesh out the concrete checklist (as a markdown artifact) and, when you’re ready, move on to the actual feature‑extraction script. No code will be generated until you ask.
