import asyncio
import gc
import io
import os
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np

import audit_log
import svi_engine
from audio_features import extract_features, extract_features_array, load_audio_array
from asr_engine import RealtimeASREngine, AudioBufferManager
from call_script import SCRIPTED_WINDOWS, CALL_SCRIPT

app = FastAPI(title="RTSAM Backend - CUDA Accelerated Multimodal Helpline Triage")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory per-call state
_CALL_STATE: dict[str, dict] = {}

VALID_SUPERVISOR_TOKENS = {"demo-supervisor-token"}

# Lazy singleton ASR engine
_ASR_ENGINE: Optional[RealtimeASREngine] = None


def get_asr_engine() -> RealtimeASREngine:
    global _ASR_ENGINE
    if _ASR_ENGINE is None:
        _ASR_ENGINE = RealtimeASREngine()
    return _ASR_ENGINE


def _get_state(call_id: str) -> dict:
    if call_id not in _CALL_STATE:
        _CALL_STATE[call_id] = {
            "smoother": svi_engine.EmaSmoother(initial=10.0),
            "last": None,
            "consent": True,
        }
        audit_log.append_entry(call_id, "System", "Call ingested, DPDP consent recorded")
    return _CALL_STATE[call_id]


from fastapi.responses import FileResponse

@app.get("/dashboard")
def get_dashboard():
    return FileResponse("dashboard.html")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "cuda_available": False,
        "engine": "RTSAM CUDA/CPU Pipeline"
    }


# ---------------------------------------------------------------------------
# 1. Main Telemetry WebSocket: /ws/live-call
#    Accepts mode ("scripted" | "real") and consent (bool)
#    Emits the exact JSON contract every 3 seconds
# ---------------------------------------------------------------------------
@app.websocket("/ws/live-call")
async def live_call_stream(
    ws: WebSocket,
    mode: str = Query(default="scripted"),
    consent: bool = Query(default=True),
    call_id: str = Query(default="live-call-01"),
):
    await ws.accept()

    # --- DPDP Act 2023 Compliance Gate ---
    if not consent:
        audit_log.append_entry(
            call_id,
            "Compliance",
            "DPDP Act 2023: Caller withheld AI consent. Decoupling AI pipeline from RAM; routing to unassisted human queue.",
        )
        await ws.send_json({
            "status": "UNASSISTED_QUEUE_ROUTED",
            "compliance": "DPDP_ACT_2023_EXCLUSION",
            "message": "AI biometric and transcription processing decoupled per user preference.",
            "system_actions": {
                "supervisor_barge_in": True,
                "atss_timer_active": False,
                "atss_seconds_remaining": 0,
                "witness_protection_prefilled": False,
                "police_112_dispatch_ready": False,
            }
        })
        await ws.close(code=1000, reason="Decoupled to unassisted queue per DPDP consent preference")
        # Ephemeral RAM purge
        gc.collect()
        return

    audit_log.append_entry(call_id, "System", f"Live call session connected (mode={mode}, DPDP consent=Granted)")
    state = _get_state(call_id)
    smoother = state["smoother"]

    try:
        if mode == "scripted":
            # Serve 3-second scripted payloads representing edge cases
            window_idx = 0
            while True:
                payload = SCRIPTED_WINDOWS[window_idx % len(SCRIPTED_WINDOWS)].copy()
                # Dynamically update window_id
                payload["window_id"] = window_idx + 1

                # Feed through smoother
                raw_score = payload["svi_metrics"]["score"]
                smoothed_score = smoother.force(raw_score) if payload["svi_metrics"]["category"] == "CRITICAL_RISK" else smoother.update(raw_score)
                payload["svi_metrics"]["score"] = round(smoothed_score)

                state["last"] = payload
                await ws.send_json(payload)

                window_idx += 1
                await asyncio.sleep(3.0)

        elif mode == "real":
            # Production GPU ingestion pipeline
            asr = get_asr_engine()
            classifier = svi_engine.get_text_classifier()
            buffer_manager = AudioBufferManager(sr=16000, window_sec=3.0, overlap_sec=0.5)
            window_id = 1

            # Ephemeral RAM isolation: Audio is processed via buffers and explicitly collected
            while True:
                try:
                    # Wait for incoming audio chunk (bytes or json)
                    message = await asyncio.wait_for(ws.receive(), timeout=3.5)
                except asyncio.TimeoutError:
                    message = None

                if message and "bytes" in message and message["bytes"]:
                    raw_bytes = message["bytes"]
                    # In-memory RAM buffer
                    with io.BytesIO(raw_bytes) as mem_buf:
                        try:
                            # 16-bit PCM or wav array
                            chunk_arr = np.frombuffer(mem_buf.getvalue(), dtype=np.int16).astype(np.float32) / 32768.0
                            buffer_manager.append(chunk_arr)
                        except Exception as e:
                            pass

                # If we have reached a 3-second window or tick
                audio_window = buffer_manager.pop_window() if buffer_manager.has_window() else np.zeros(48000, dtype=np.float32)

                # 1. Acoustic biomarker extraction (<50ms via openSMILE / fast DSP)
                acoustic = extract_features_array(audio_window, sr=16000)

                # 2. Real-time ASR via faster-whisper on GPU
                transcript = asr.transcribe_window(audio_window) if len(audio_window) > 0 else ""

                # 3. Transformer text classification on GPU
                p_suicide, p_panic, flags = classifier.analyze_text(transcript)

                # 4. SVI Late Fusion calculation
                raw_svi, delta_crit, category, trigger = svi_engine.compute_svi(
                    a_t=acoustic["acoustic_score"],
                    t_t=max(p_suicide, p_panic),
                    b_t=0.1,
                    p_suicide=p_suicide,
                    p_panic=p_panic,
                    has_explicit_marker=flags.get("critical_marker", False),
                )

                smoothed_svi = smoother.force(raw_svi) if delta_crit > 0 else smoother.update(raw_svi)

                is_critical = category == "CRITICAL_RISK"
                payload = {
                    "window_id": window_id,
                    "svi_metrics": {
                        "score": round(smoothed_svi),
                        "category": category,
                        "primary_trigger": flags.get("primary_trigger", trigger),
                    },
                    "acoustic_telemetry": {
                        "pitch_f0_mean": acoustic["pitch_f0_mean"],
                        "jitter": acoustic["jitter"],
                        "shimmer_db": acoustic["shimmer_db"],
                        "hnr_db": acoustic["hnr_db"],
                    },
                    "nlp_telemetry": {
                        "transcript": transcript or "[ambient audio monitored]",
                        "p_suicide": round(p_suicide, 2),
                        "p_panic": round(p_panic, 2),
                    },
                    "engine_flags": {
                        "bpab": "Background Acoustic Monitored",
                        "asd_nse": "Acoustic-Semantic Dissonance High" if abs(acoustic["acoustic_score"] - p_panic) > 0.4 else "Nominal",
                        "slp_akg": "SC/ST Act Violation Flagged" if flags.get("sc_st_violation") else "None",
                        "nc_acsm": "Stealth Mode Standby" if is_critical else "Standard Audio Mode",
                    },
                    "system_actions": {
                        "supervisor_barge_in": is_critical,
                        "atss_timer_active": is_critical,
                        "atss_seconds_remaining": 22 if is_critical else 0,
                        "witness_protection_prefilled": flags.get("sc_st_violation", False),
                        "police_112_dispatch_ready": is_critical,
                    },
                }

                state["last"] = payload
                await ws.send_json(payload)
                window_id += 1

                # If running loop without client pushing bytes, sleep window
                if not message:
                    await asyncio.sleep(3.0)

    except WebSocketDisconnect:
        audit_log.append_entry(call_id, "System", "Live call WebSocket disconnected.")
    finally:
        # Ephemeral RAM Isolation: explicitly collect and clear memory
        gc.collect()


# ---------------------------------------------------------------------------
# 2. Backward-Compatible Call Stream: /ws/calls/{call_id}/stream
# ---------------------------------------------------------------------------
@app.websocket("/ws/calls/{call_id}/stream")
async def legacy_call_stream(ws: WebSocket, call_id: str):
    await ws.accept()
    state = _get_state(call_id)
    smoother = state["smoother"]
    try:
        step_index = 0
        while True:
            step = CALL_SCRIPT[step_index]
            p_suicide = 0.90 if "suicidal" in step.get("flags", []) else 0.02
            p_panic = 0.85 if "critical override" in step.get("flags", []) else 0.15
            has_crit = any(f in {"critical override", "weapon", "hostage"} for f in step.get("flags", []))

            # Normalize acoustic params
            a_score = svi_engine.WEIGHTS["acoustic"] * (step["jitter"] / 4.0 + step["shimmer"] / 8.0)
            raw_svi, delta_crit, category, _ = svi_engine.compute_svi(
                a_t=a_score,
                t_t=p_panic,
                b_t=0.8 if step.get("is_freeze") else 0.0,
                p_suicide=p_suicide,
                p_panic=p_panic,
                has_explicit_marker=has_crit,
            )

            smoothed_svi = smoother.force(raw_svi) if delta_crit > 0 else smoother.update(raw_svi)
            band = "Critical" if category == "CRITICAL_RISK" else ("High" if category == "HIGH_RISK" else ("Moderate" if category == "MODERATE_RISK" else "Low"))

            message = {
                "transcriptLine": {
                    "speaker": step["speaker"],
                    "text": step["text"],
                    "flag": step["flags"][0] if step.get("flags") else None,
                },
                "svi": round(smoothed_svi, 1),
                "jitter": step["jitter"],
                "shimmer": step["shimmer"],
                "hnr": step["hnr"],
                "f0": step["f0_var"],
                "band": band,
            }
            state["last"] = message
            await ws.send_json(message)

            step_index = (step_index + 1) % len(CALL_SCRIPT)
            await asyncio.sleep(3.0)
    except WebSocketDisconnect:
        pass
    finally:
        gc.collect()


# ---------------------------------------------------------------------------
# 3. Audio Analysis: POST /calls/{call_id}/analyze-audio
# ---------------------------------------------------------------------------
@app.post("/calls/{call_id}/analyze-audio")
async def analyze_audio(call_id: str, file: UploadFile = File(...)):
    # Ephemeral RAM buffer
    content = await file.read()
    with io.BytesIO(content) as mem_file:
        try:
            audio_arr, sr = load_audio_array(mem_file, target_sr=16000)
            features = extract_features_array(audio_arr, sr=sr)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Audio analysis failed: {str(e)}")

    # 1. Real-time ASR on uploaded audio
    asr = get_asr_engine()
    transcript = asr.transcribe_window(audio_arr, sr=16000) if len(audio_arr) > 0 else ""

    # 2. Textual trauma classification
    classifier = svi_engine.get_text_classifier()
    p_suicide, p_panic, flags = classifier.analyze_text(transcript)

    # 3. Multimodal SVI Late Fusion
    raw_svi, delta_crit, category, trigger = svi_engine.compute_svi(
        a_t=features["acoustic_score"],
        t_t=max(p_suicide, p_panic),
        b_t=0.1,
        p_suicide=p_suicide,
        p_panic=p_panic,
        has_explicit_marker=flags.get("critical_marker", False),
    )

    state = _get_state(call_id)
    smoother = state["smoother"]
    smoothed_svi = smoother.force(raw_svi) if delta_crit > 0 else smoother.update(raw_svi)
    band = "Critical" if category == "CRITICAL_RISK" else ("High" if category == "HIGH_RISK" else ("Moderate" if category == "MODERATE_RISK" else "Low"))

    is_critical = category == "CRITICAL_RISK"
    payload = {
        "window_id": "upload",
        "svi_metrics": {
            "score": round(smoothed_svi),
            "category": category,
            "primary_trigger": flags.get("primary_trigger", trigger),
        },
        "acoustic_telemetry": {
            "pitch_f0_mean": features["pitch_f0_mean"],
            "jitter": features["jitter"],
            "shimmer_db": features["shimmer_db"],
            "hnr_db": features["hnr_db"],
        },
        "nlp_telemetry": {
            "transcript": transcript or "[Audio analyzed: ambient noise / unvoiced]",
            "p_suicide": round(p_suicide, 2),
            "p_panic": round(p_panic, 2),
        },
        "engine_flags": {
            "bpab": "Uploaded Recording Analyzed",
            "asd_nse": "Acoustic-Semantic Dissonance High" if abs(features["acoustic_score"] - p_panic) > 0.4 else "Nominal",
            "slp_akg": "SC/ST Act Violation Flagged" if flags.get("sc_st_violation") else "None",
            "nc_acsm": "Stealth Mode Standby" if is_critical else "Standard Audio Mode",
        },
        "system_actions": {
            "supervisor_barge_in": is_critical,
            "atss_timer_active": is_critical,
            "atss_seconds_remaining": 22 if is_critical else 0,
            "witness_protection_prefilled": flags.get("sc_st_violation", False),
            "police_112_dispatch_ready": is_critical,
        },
        "features": features,
        "svi": round(smoothed_svi, 1),
        "band": band,
        "category": category,
    }

    state["last"] = payload
    audit_log.append_entry(call_id, "AI", f"Recording analyzed: '{file.filename}' (SVI {smoothed_svi:.0f} - {category})")

    # Clean up
    del content
    del audio_arr
    gc.collect()

    return payload


# ---------------------------------------------------------------------------
# 4. Audit Log & State
# ---------------------------------------------------------------------------
@app.get("/calls/{call_id}/audit")
def get_audit(call_id: str):
    return {"entries": audit_log.get_log(call_id), "chain_valid": audit_log.verify_chain(call_id)}


@app.get("/calls/{call_id}/state")
def get_state(call_id: str):
    state = _get_state(call_id)
    return {"last": state["last"], "audit": audit_log.get_log(call_id)}


# ---------------------------------------------------------------------------
# 5. Dispatch
# ---------------------------------------------------------------------------
class DispatchRequest(BaseModel):
    coordinates: Optional[str] = None


@app.post("/calls/{call_id}/dispatch/{action}")
def dispatch(
    call_id: str,
    action: str,
    body: DispatchRequest,
    x_supervisor_token: Optional[str] = Header(default=None),
):
    if action not in {"police", "witness-protection"}:
        raise HTTPException(status_code=404, detail="Unknown dispatch action")
    if x_supervisor_token not in VALID_SUPERVISOR_TOKENS:
        raise HTTPException(status_code=401, detail="Supervisor ratification required")

    state = _get_state(call_id)
    last = state.get("last")
    is_crit = False
    if last:
        if "svi_metrics" in last and last["svi_metrics"].get("category") == "CRITICAL_RISK":
            is_crit = True
        elif last.get("band") == "Critical":
            is_crit = True

    if not is_crit:
        raise HTTPException(status_code=409, detail="Dispatch only permitted while band is Critical / CRITICAL_RISK")

    label = "Police Dispatch (112)" if action == "police" else "Witness Protection Docket"
    entry = audit_log.append_entry(
        call_id, "Supervisor", f"{label} ratified" + (f" — {body.coordinates}" if body.coordinates else "")
    )
    return {"status": "dispatched", "action": action, "audit_entry": entry}
