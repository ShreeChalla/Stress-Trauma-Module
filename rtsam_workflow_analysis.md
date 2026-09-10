# RTSAM — Complete Project Workflow

![RTSAM Architecture Diagram](C:\Users\ShreeMain\.gemini\antigravity-ide\brain\ce7941af-97a6-4358-8655-81c87954516c\rtsam_workflow_1789012137800.jpg)

---

## Project Overview

**RTSAM** (Real-Time Stress Assessment & Multimodal Helpline Triage) is a CUDA-accelerated FastAPI backend for the **NHAA 14566 Helpline Counselor Dashboard**. It performs real-time multimodal analysis of helpline calls to triage victims of SC/ST atrocities, suicide risk, and active threats.

---

## File Architecture

| File | Role |
|------|------|
| [`main.py`](file:///c:/miniProj/RTSAM/main.py) | FastAPI server — all endpoints, WebSocket handlers, orchestration |
| [`svi_engine.py`](file:///c:/miniProj/RTSAM/svi_engine.py) | SVI fusion math, text classifier (suicidal-bert), EMA smoothing |
| [`asr_engine.py`](file:///c:/miniProj/RTSAM/asr_engine.py) | Real-time ASR via faster-whisper (CUDA/CPU), audio buffer manager |
| [`audio_features.py`](file:///c:/miniProj/RTSAM/audio_features.py) | Acoustic biomarker extraction via openSMILE eGeMAPSv02 |
| [`audit_log.py`](file:///c:/miniProj/RTSAM/audit_log.py) | SHA-256 hash-chained, append-only audit ledger |
| [`call_script.py`](file:///c:/miniProj/RTSAM/call_script.py) | 8 pre-recorded edge-case windows for scripted demo mode |
| [`dashboard.html`](file:///c:/miniProj/RTSAM/dashboard.html) | React + Tailwind counselor dashboard (single-file SPA) |
| [`demo_client.py`](file:///c:/miniProj/RTSAM/demo_client.py) | Python CLI WebSocket client for terminal demos |
| [`test_api.py`](file:///c:/miniProj/RTSAM/test_api.py) | API endpoint tests |
| [`test_pipeline.py`](file:///c:/miniProj/RTSAM/test_pipeline.py) | SVI math, overrides, acoustic normalization tests |

---

## End-to-End Workflow

### 1️⃣ Client Connection

**Two clients** can connect to the system:
- **Dashboard** ([`dashboard.html`](file:///c:/miniProj/RTSAM/dashboard.html)) — React SPA served at `GET /dashboard`, connects via WebSocket
- **Demo Client** ([`demo_client.py`](file:///c:/miniProj/RTSAM/demo_client.py)) — Python CLI using `websockets` library

Both connect to:
```
ws://localhost:8000/ws/live-call?mode=scripted&consent=true&call_id=live-call-01
```

---

### 2️⃣ DPDP Act 2023 Compliance Gate

> [!IMPORTANT]
> The **first decision** the server makes is checking `consent`.

| `consent` | Behavior |
|-----------|----------|
| `true` | Full AI biometric + transcription pipeline activates |
| `false` | AI models decoupled from RAM → caller routed to **unassisted human queue** → `gc.collect()` purge |

When consent is denied, the server sends a single JSON response with `"status": "UNASSISTED_QUEUE_ROUTED"` and closes the connection immediately.

**Code**: [`main.py:86-107`](file:///c:/miniProj/RTSAM/main.py#L86-L107)

---

### 3️⃣ Mode Selection

| Mode | Description |
|------|-------------|
| `"scripted"` | Loops through 8 pre-recorded edge-case windows from [`call_script.py`](file:///c:/miniProj/RTSAM/call_script.py), emitting one every 3 seconds |
| `"real"` | Production GPU pipeline — ingests live 16kHz audio chunks from the client |

---

### 4️⃣ Real-Time GPU Processing Pipeline (`mode=real`)

When a live audio chunk arrives, **three parallel engines** process it:

#### 🎤 A. Acoustic Engine ([`audio_features.py`](file:///c:/miniProj/RTSAM/audio_features.py))
1. Receives 16kHz float32 audio array in RAM
2. Extracts voice biomarkers via **openSMILE eGeMAPSv02** (LLD):
   - **F0** (fundamental frequency in semitones)
   - **Jitter** (vocal cord trembling)
   - **Shimmer** (sobbing/breathlessness)
   - **HNR** (whispering/gasping detection)
3. Computes normalized **A(t) ∈ [0,1]**:
   ```
   A(t) = 0.30·jitter_n + 0.30·shimmer_n + 0.25·hnr_n + 0.15·f0_n
   ```
4. Falls back to numpy DSP approximation if openSMILE is unavailable

#### 🗣️ B. ASR Engine ([`asr_engine.py`](file:///c:/miniProj/RTSAM/asr_engine.py))
1. **faster-whisper** (small model) on CUDA float16 (CPU int8 fallback)
2. **Silero VAD** filtering (`min_silence_duration_ms=500`) rejects noise
3. **Context prompt injection**: _"NHAA 14566 helpline call involving SC/ST atrocity, trauma, FIR, legal aid"_
4. **3-second sliding window** with **0.5-second overlap** via `AudioBufferManager`
5. Returns transcription text

#### 🧠 C. NLP / Text Classifier ([`svi_engine.py`](file:///c:/miniProj/RTSAM/svi_engine.py))
1. **suicidal-bert** (`gohjiayi/suicidal-bert`) on GPU → outputs `P(suicide)`
2. **Panic heuristics** via keyword matching → outputs `P(panic)`
3. **SC/ST Act violation detection** via regex lexicons (landlord, well, untouchable, caste, etc.)
4. **Critical marker regex** for weapons, mob attacks, hostages, murder, rape
5. Falls back to **lexicon heuristic** if transformers unavailable

---

### 5️⃣ SVI Fusion Engine ([`svi_engine.py`](file:///c:/miniProj/RTSAM/svi_engine.py))

All three streams merge into the **Sigmoid Late Fusion** formula:

```
SVI(t) = σ(w_a·A(t) + w_t·T(t) + w_b·B(t) + δ_critical) × 100
```

Where:
- `σ(x) = 1 / (1 + exp(-10·(x - 0.5)))` — steep sigmoid centered at 0.5
- `w_a = 0.35` (acoustic), `w_t = 0.40` (text), `w_b = 0.25` (behavior)

#### Deterministic Hard Overrides
If **any** of these are true, `δ_critical = 0.50`, forcing the score into **CRITICAL_RISK (81-100)**:
- `P(suicide) > 0.80`
- `P(panic) > 0.80`
- Explicit markers (rape, weapon, gun, knife, murder, hostage, lynch, mob attack)

#### EMA Smoothing
```
SVI_smooth(t) = λ·SVI(t) + (1-λ)·SVI_smooth(t-1),  λ = 0.3
```
Hard overrides use `force()` (instant snap) instead of smooth transition.

#### Risk Categories
| Range | Category | Color |
|-------|----------|-------|
| 0–25 | `LOW_RISK` | 🟢 Green |
| 26–50 | `MODERATE_RISK` | 🟡 Amber |
| 51–80 | `HIGH_RISK` | 🟠 Orange |
| 81–100 | `CRITICAL_RISK` | 🔴 Red |

---

### 6️⃣ JSON Telemetry Emission

Every **3 seconds**, a full telemetry payload is sent to the client via WebSocket:

```json
{
  "window_id": 6,
  "svi_metrics": { "score": 88, "category": "CRITICAL_RISK", "primary_trigger": "..." },
  "acoustic_telemetry": { "pitch_f0_mean": 182.4, "jitter": 0.082, "shimmer_db": 1.45, "hnr_db": 6.2 },
  "nlp_telemetry": { "transcript": "...", "p_suicide": 0.02, "p_panic": 0.89 },
  "engine_flags": { "bpab": "...", "asd_nse": "...", "slp_akg": "...", "nc_acsm": "..." },
  "system_actions": { "supervisor_barge_in": true, "atss_timer_active": true, ... }
}
```

---

### 7️⃣ System Actions & Dispatch

When SVI reaches **CRITICAL_RISK**, the system auto-triggers:

| Action | Description |
|--------|-------------|
| **Supervisor Barge-In** | Supervisor audio channel activated |
| **ATSS Timer** | Countdown timer for auto-escalation |
| **Police 112 Dispatch** | Human-in-the-loop gate — requires `X-Supervisor-Token` |
| **Witness Protection** | Docket pre-filled for SC/ST Act violations |

Dispatch endpoints require:
1. Valid `X-Supervisor-Token` header
2. Current call status must be `CRITICAL_RISK`

**Code**: [`main.py:393-421`](file:///c:/miniProj/RTSAM/main.py#L393-L421)

---

### 8️⃣ Security & Compliance Layer

| Feature | Implementation |
|---------|---------------|
| **Hash-Chained Audit Ledger** | SHA-256 blockchain-style chaining in [`audit_log.py`](file:///c:/miniProj/RTSAM/audit_log.py) — tamper-evident for evidentiary use |
| **Ephemeral RAM Isolation** | Explicit `del` + `gc.collect()` after every session |
| **DPDP Act 2023 Gate** | AI processing fully decoupled when consent withheld |
| **Chain Verification** | `verify_chain()` recomputes every hash to detect tampering |

---

## API Endpoints Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Service health + engine diagnostics |
| `GET` | `/dashboard` | Serves the React counselor dashboard |
| `WS` | `/ws/live-call` | **Primary** — real-time telemetry stream |
| `WS` | `/ws/calls/{id}/stream` | Legacy backward-compatible stream |
| `POST` | `/calls/{id}/analyze-audio` | Upload `.wav`/`.mp3` for offline analysis |
| `GET` | `/calls/{id}/audit` | Hash-chained audit log |
| `GET` | `/calls/{id}/state` | Reconnection snapshot |
| `POST` | `/calls/{id}/dispatch/{action}` | Police/Witness Protection dispatch |

---

## Scripted Demo Flow (8 Windows)

The scripted mode cycles through these edge cases from [`call_script.py`](file:///c:/miniProj/RTSAM/call_script.py):

| # | SVI | Category | Scenario |
|---|-----|----------|----------|
| 1 | 12 | LOW | Baseline intake, DPDP recorded |
| 2 | 24 | LOW | Hesitant initial report |
| 3 | 42 | MODERATE | SC/ST public humiliation |
| 4 | 62 | HIGH | Enforced untouchability & coercion |
| 5 | 76 | HIGH | Stealth whispering, someone outside |
| 6 | 88 | **CRITICAL** | Active mob attack, broken gate |
| 7 | 92 | **CRITICAL** | Cognitive freeze, DTMF tap only |
| 8 | 64 | HIGH | De-escalation, threat passing |
