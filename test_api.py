"""
Tests for FastAPI endpoints, DPDP compliance gate, WebSocket telemetry contracts, and dispatch HITL.
"""
import io
import json
import numpy as np
import soundfile as sf
import pytest
from starlette.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_live_call_dpdp_consent_withheld(client):
    """If caller withholds consent, DPDP gate must decouple AI and route to unassisted queue."""
    with client.websocket_connect("/ws/live-call?mode=scripted&consent=false") as ws:
        msg = ws.receive_json()
        assert msg["status"] == "UNASSISTED_QUEUE_ROUTED"
        assert msg["compliance"] == "DPDP_ACT_2023_EXCLUSION"
        assert "decoupled" in msg["message"].lower()


def test_live_call_scripted_contract(client):
    """Scripted live call must emit valid JSON matching the exact contract."""
    with client.websocket_connect("/ws/live-call?mode=scripted&consent=true") as ws:
        msg = ws.receive_json()

        # Check required schema keys
        assert "window_id" in msg
        assert "svi_metrics" in msg
        assert "acoustic_telemetry" in msg
        assert "nlp_telemetry" in msg
        assert "engine_flags" in msg
        assert "system_actions" in msg

        # Metric specifics
        assert isinstance(msg["svi_metrics"]["score"], (int, float))
        assert msg["svi_metrics"]["category"] in ["LOW_RISK", "MODERATE_RISK", "HIGH_RISK", "CRITICAL_RISK"]
        assert isinstance(msg["svi_metrics"]["primary_trigger"], str)

        # Telemetry specifics
        assert "pitch_f0_mean" in msg["acoustic_telemetry"]
        assert "jitter" in msg["acoustic_telemetry"]
        assert "shimmer_db" in msg["acoustic_telemetry"]
        assert "hnr_db" in msg["acoustic_telemetry"]

        # NLP specifics
        assert "transcript" in msg["nlp_telemetry"]
        assert "p_suicide" in msg["nlp_telemetry"]
        assert "p_panic" in msg["nlp_telemetry"]

        # System actions
        assert "supervisor_barge_in" in msg["system_actions"]
        assert "atss_timer_active" in msg["system_actions"]
        assert "witness_protection_prefilled" in msg["system_actions"]
        assert "police_112_dispatch_ready" in msg["system_actions"]


def test_audio_analysis_endpoint(client):
    """Upload a synthetic 16kHz audio buffer and verify acoustic feature extraction."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 220Hz tone with slight noise
    waveform = (0.5 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.randn(sr)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, waveform, sr, format="WAV")
    buf.seek(0)

    response = client.post(
        "/calls/test-audio-call/analyze-audio",
        files={"file": ("test.wav", buf, "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    assert "svi" in data
    assert "band" in data
    assert "category" in data
    assert data["features"]["pitch_f0_mean"] > 0


def test_supervisor_dispatch_gate(client):
    """Dispatch requires valid supervisor token and Critical Risk."""
    # 1. Invalid token -> 401
    res1 = client.post("/calls/call-disp/dispatch/police", json={})
    assert res1.status_code == 401

    # 2. Valid token but not critical -> 409
    res2 = client.post(
        "/calls/call-disp/dispatch/police",
        json={},
        headers={"X-Supervisor-Token": "demo-supervisor-token"},
    )
    assert res2.status_code == 409
