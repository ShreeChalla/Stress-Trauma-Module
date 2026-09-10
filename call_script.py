"""
Pre-recorded scripted 3-second window timeline representing edge cases:
- Monotone 'Freeze' (cognitive paralysis / abnormal silence)
- Stealth Whispering (low HNR, suppressed pitch)
- SC/ST Atrocity Overrides (enforced untouchability, violence threats)
- Active Mob Attack & Weapon Triggers

Provides full telemetry payloads adhering to the expected contract.
"""

SCRIPTED_WINDOWS = [
    {
        "window_id": 1,
        "svi_metrics": {
            "score": 12,
            "category": "LOW_RISK",
            "primary_trigger": "Baseline Intake & DPDP Recorded"
        },
        "acoustic_telemetry": {
            "pitch_f0_mean": 172.0,
            "jitter": 0.012,
            "shimmer_db": 0.45,
            "hnr_db": 21.0
        },
        "nlp_telemetry": {
            "transcript": "NHAA helpline, you're speaking with Divya. You're safe on this line — go ahead.",
            "p_suicide": 0.01,
            "p_panic": 0.04
        },
        "engine_flags": {
            "bpab": "Normal Ambient Baseline",
            "asd_nse": "Nominal Acoustic Alignment",
            "slp_akg": "None",
            "nc_acsm": "Standard Audio Mode"
        },
        "system_actions": {
            "supervisor_barge_in": False,
            "atss_timer_active": False,
            "atss_seconds_remaining": 0,
            "witness_protection_prefilled": False,
            "police_112_dispatch_ready": False
        }
    },
    {
        "window_id": 2,
        "svi_metrics": {
            "score": 24,
            "category": "LOW_RISK",
            "primary_trigger": "Hesitant Initial Report"
        },
        "acoustic_telemetry": {
            "pitch_f0_mean": 175.4,
            "jitter": 0.021,
            "shimmer_db": 0.68,
            "hnr_db": 19.2
        },
        "nlp_telemetry": {
            "transcript": "Hello... yes, I wanted to report something that happened yesterday.",
            "p_suicide": 0.01,
            "p_panic": 0.18
        },
        "engine_flags": {
            "bpab": "Clean Channel",
            "asd_nse": "Mild Vocal Stress",
            "slp_akg": "None",
            "nc_acsm": "Standard Audio Mode"
        },
        "system_actions": {
            "supervisor_barge_in": False,
            "atss_timer_active": False,
            "atss_seconds_remaining": 0,
            "witness_protection_prefilled": False,
            "police_112_dispatch_ready": False
        }
    },
    {
        "window_id": 3,
        "svi_metrics": {
            "score": 42,
            "category": "MODERATE_RISK",
            "primary_trigger": "SC/ST Act Violation (Public Humiliation)"
        },
        "acoustic_telemetry": {
            "pitch_f0_mean": 188.0,
            "jitter": 0.038,
            "shimmer_db": 0.95,
            "hnr_db": 15.4
        },
        "nlp_telemetry": {
            "transcript": "It was the landlord. In front of the whole street. Everyone saw.",
            "p_suicide": 0.02,
            "p_panic": 0.48
        },
        "engine_flags": {
            "bpab": "Crowd Murmur Detected",
            "asd_nse": "Suppressed Tone Dissonance",
            "slp_akg": "SC/ST Act Violation (Public Humiliation)",
            "nc_acsm": "Standard Audio Mode"
        },
        "system_actions": {
            "supervisor_barge_in": False,
            "atss_timer_active": True,
            "atss_seconds_remaining": 45,
            "witness_protection_prefilled": False,
            "police_112_dispatch_ready": False
        }
    },
    {
        "window_id": 4,
        "svi_metrics": {
            "score": 62,
            "category": "HIGH_RISK",
            "primary_trigger": "SC/ST Act Violation (Enforced Untouchability & Coercion)"
        },
        "acoustic_telemetry": {
            "pitch_f0_mean": 204.2,
            "jitter": 0.054,
            "shimmer_db": 1.18,
            "hnr_db": 12.8
        },
        "nlp_telemetry": {
            "transcript": "They said if I complain they will not let us draw water again from the well.",
            "p_suicide": 0.03,
            "p_panic": 0.72
        },
        "engine_flags": {
            "bpab": "Audible Coercive Threat",
            "asd_nse": "Vocal Pitch Spike",
            "slp_akg": "SC/ST Act Violation (Enforced Untouchability)",
            "nc_acsm": "Standard Audio Mode"
        },
        "system_actions": {
            "supervisor_barge_in": False,
            "atss_timer_active": True,
            "atss_seconds_remaining": 35,
            "witness_protection_prefilled": True,
            "police_112_dispatch_ready": False
        }
    },
    {
        "window_id": 5,
        "svi_metrics": {
            "score": 76,
            "category": "HIGH_RISK",
            "primary_trigger": "Acoustic Threat & Stealth Whispering"
        },
        "acoustic_telemetry": {
            "pitch_f0_mean": 142.1,
            "jitter": 0.068,
            "shimmer_db": 1.32,
            "hnr_db": 8.4
        },
        "nlp_telemetry": {
            "transcript": "There— there is someone outside. I have to be quiet a moment.",
            "p_suicide": 0.02,
            "p_panic": 0.79
        },
        "engine_flags": {
            "bpab": "Approaching Footsteps & Distant Shouts",
            "asd_nse": "Acoustic-Semantic Dissonance High",
            "slp_akg": "SC/ST Act Violation (Physical Menace)",
            "nc_acsm": "Stealth Mode Standby"
        },
        "system_actions": {
            "supervisor_barge_in": True,
            "atss_timer_active": True,
            "atss_seconds_remaining": 28,
            "witness_protection_prefilled": True,
            "police_112_dispatch_ready": False
        }
    },
    {
        "window_id": 6,
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
            "supervisor_barge_in": True,
            "atss_timer_active": True,
            "atss_seconds_remaining": 22,
            "witness_protection_prefilled": True,
            "police_112_dispatch_ready": True
        }
    },
    {
        "window_id": 7,
        "svi_metrics": {
            "score": 92,
            "category": "CRITICAL_RISK",
            "primary_trigger": "Cognitive Freeze / Keypad Triage"
        },
        "acoustic_telemetry": {
            "pitch_f0_mean": 110.0,
            "jitter": 0.091,
            "shimmer_db": 1.62,
            "hnr_db": 5.1
        },
        "nlp_telemetry": {
            "transcript": "[breathing, no speech, single tap detected]",
            "p_suicide": 0.04,
            "p_panic": 0.94
        },
        "engine_flags": {
            "bpab": "Heavy Rapid Breathing",
            "asd_nse": "Abnormal Silence (Cognitive Freeze)",
            "slp_akg": "Active Menace Ongoing",
            "nc_acsm": "Stealth Active (DTMF Tap Detected)"
        },
        "system_actions": {
            "supervisor_barge_in": True,
            "atss_timer_active": True,
            "atss_seconds_remaining": 15,
            "witness_protection_prefilled": True,
            "police_112_dispatch_ready": True
        }
    },
    {
        "window_id": 8,
        "svi_metrics": {
            "score": 64,
            "category": "HIGH_RISK",
            "primary_trigger": "Threat Passing / De-escalation"
        },
        "acoustic_telemetry": {
            "pitch_f0_mean": 168.2,
            "jitter": 0.045,
            "shimmer_db": 0.98,
            "hnr_db": 13.5
        },
        "nlp_telemetry": {
            "transcript": "Okay... okay they've gone. I think it's safer now.",
            "p_suicide": 0.02,
            "p_panic": 0.52
        },
        "engine_flags": {
            "bpab": "Ambient Settling",
            "asd_nse": "Post-Stress Recovery",
            "slp_akg": "Under Investigation",
            "nc_acsm": "Standard Audio Mode"
        },
        "system_actions": {
            "supervisor_barge_in": True,
            "atss_timer_active": False,
            "atss_seconds_remaining": 0,
            "witness_protection_prefilled": True,
            "police_112_dispatch_ready": True
        }
    }
]

# Legacy CALL_SCRIPT kept for backward compatibility with older dashboard endpoints
CALL_SCRIPT = [
    {
        "speaker": "counselor",
        "text": "NHAA helpline, you're speaking with Divya. You're safe on this line — go ahead.",
        "jitter": 0.3, "shimmer": 1.1, "hnr": 21, "f0_var": 4, "flags": []
    },
    {
        "speaker": "caller",
        "text": "Hello... yes, I wanted to report something that happened yesterday.",
        "jitter": 0.6, "shimmer": 1.6, "hnr": 19, "f0_var": 7, "flags": []
    },
    {
        "speaker": "caller",
        "text": "It was the landlord. In front of the whole street. Everyone saw.",
        "jitter": 1.4, "shimmer": 3.1, "hnr": 15, "f0_var": 14, "flags": ["sc/st act"]
    },
    {
        "speaker": "caller",
        "text": "They broke our front gate... they won't let us use the well!",
        "jitter": 3.6, "shimmer": 6.7, "hnr": 6, "f0_var": 33, "flags": ["critical override"]
    },
]
