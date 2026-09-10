# RTSAM Backend (CUDA GPU & Multimodal Helpline Triage)

An accelerated FastAPI backend for the NHAA Helpline Counselor Dashboard, featuring:
- **Sigmoid Late Fusion SVI Engine** with deterministic hard overrides ($\delta_{\text{critical}} = 0.50$) for active weapon/mob attacks, $P_{\text{suicide}} > 0.80$, or $P_{\text{panic}} > 0.80$.
- **Real-Time ASR Engine** (`faster-whisper` on CUDA with Silero VAD suppression and context prompt injection).
- **Acoustic Paralinguistics** via `opensmile` eGeMAPSv02 ($F_0$, Jitter, Shimmer, HNR) extracting voice biomarkers directly from 16kHz RAM arrays in <50ms.
- **Unified WebSocket Telemetry Contract** (`/ws/live-call`) supporting both `scripted` demo loops and live GPU audio processing.
- **DPDP Act 2023 Compliance Gate** that decouples AI processing in RAM when consent is withheld.
- **Ephemeral RAM Isolation** with explicit memory deallocation and `gc.collect()` post-session.
- **Hash-Chained Audit Ledger** for tamper-evident compliance.

---

## 1. Virtual Environment & Dependencies

This repository uses a local virtual environment (`.venv`).

```powershell
# Activate local virtual environment
.\.venv\Scripts\Activate.ps1

# Install core dependencies
pip install -r requirements.txt

# To install PyTorch with CUDA 12.1 / 13.0 bindings:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 2. Launch the Backend Server

```powershell
.\.venv\Scripts\uvicorn main:app --reload --port 8000
```

---

## 3. Real-Time Telemetry WebSocket

Connect to:
```
ws://localhost:8000/ws/live-call?mode=scripted&consent=true
```

### Query Parameters:
- `mode`:
  - `"scripted"` (default for dashboard demo): Emits realistic pre-recorded 3-second edge case payloads (Monotone "Freeze", Stealth Whispering, SC/ST Overrides).
  - `"real"`: Production GPU ingestion receiving live 16kHz audio chunks, executing `faster-whisper`, `openSMILE`, and PyTorch transformer heads.
- `consent`:
  - `true` (default): Full AI biometric and transcription pipeline active.
  - `false` (DPDP Act 2023 Gate): Decouples AI models from RAM and immediately routes to an unassisted queue.

### Expected JSON Telemetry Schema (Emitted Every 3 Seconds):
```json
{
  "window_id": 1,
  "svi_metrics": {
    "score": 88,
    "category": "CRITICAL_RISK",
    "primary_trigger": "Active Mob Attack & Atrocity Threat"
  },
  "acoustic_telemetry": {
    "pitch_f0_mean": 182.4,
    "jitter": 0.082,
    "shimmer_db": 1.45,
    "hnr_db": 6.2
  },
  "nlp_telemetry": {
    "transcript": "They broke our front gate... they won't let us use the well!",
    "p_suicide": 0.02,
    "p_panic": 0.89
  },
  "engine_flags": {
    "bpab": "Background Shouting & Breaking Glass",
    "asd_nse": "Acoustic-Semantic Dissonance High",
    "slp_akg": "SC/ST Act Violation (Enforced Untouchability)",
    "nc_acsm": "Stealth Mode Standby"
  },
  "system_actions": {
    "supervisor_barge_in": true,
    "atss_timer_active": true,
    "atss_seconds_remaining": 22,
    "witness_protection_prefilled": true,
    "police_112_dispatch_ready": true
  }
}
```

---

## 4. Other Endpoints

- `GET /health` — Service health and engine diagnostics.
- `POST /calls/{call_id}/analyze-audio` — Upload a `.wav`/`.mp3` clip, extracts voice biomarkers in RAM, computes SVI.
- `GET /calls/{call_id}/audit` — Cryptographic hash-chained audit log.
- `POST /calls/{call_id}/dispatch/police` or `/dispatch/witness-protection` — Human-In-The-Loop gate requiring `X-Supervisor-Token` and `CRITICAL_RISK` status.
- `GET /calls/{call_id}/state` — Reconnection snapshot.

---

## 5. Verification & Tests

Run the automated test suite:
```powershell
.\.venv\Scripts\python.exe -m pytest -v
```
Verifies:
- Sigmoid Late Fusion mathematics and steepness.
- Deterministic overrides snapping straight to Critical Risk ($81-100$).
- Acoustic biomarker normalization ($A(t) \in [0, 1]$).
- WebSocket contract schema matching the required specification.
- DPDP Act 2023 compliance gate routing when `consent=false`.
