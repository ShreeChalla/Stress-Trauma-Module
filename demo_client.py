"""
Live terminal demo client connecting to RTSAM WebSocket stream.
"""
import asyncio
import json
import sys
import websockets

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def color_category(cat: str) -> str:
    return cat


async def run_demo(max_windows: int = 4):
    uri = "ws://127.0.0.1:8000/ws/live-call?mode=scripted&consent=true"
    print(f"\n>>> Connecting to RTSAM Live Helpline Stream: {uri}\n")

    async with websockets.connect(uri) as ws:
        for _ in range(max_windows):
            data = await ws.recv()
            payload = json.loads(data)

            w_id = payload["window_id"]
            svi = payload["svi_metrics"]
            ac = payload["acoustic_telemetry"]
            nlp = payload["nlp_telemetry"]
            flags = payload["engine_flags"]
            sys_acts = payload["system_actions"]

            print("=" * 68)
            print(f"[WINDOW #{w_id}] SVI Score: {svi['score']} | Category: {svi['category']}")
            print(f"Trigger: {svi['primary_trigger']}")
            print(f"Transcript: \"{nlp['transcript']}\"")
            print(f"NLP: P(suicide)={nlp['p_suicide']:.2f} | P(panic)={nlp['p_panic']:.2f}")
            print(f"Acoustics: F0={ac['pitch_f0_mean']}Hz | Jitter={ac['jitter']} | Shimmer={ac['shimmer_db']}dB | HNR={ac['hnr_db']}dB")
            print(f"Engine Flags: {flags['bpab']} | {flags['asd_nse']} | {flags['slp_akg']}")
            print(f"System Actions: Barge-in={sys_acts['supervisor_barge_in']} | Police 112 Ready={sys_acts['police_112_dispatch_ready']} | Witness Protection={sys_acts['witness_protection_prefilled']}")
            print("=" * 68 + "\n")

if __name__ == "__main__":
    asyncio.run(run_demo(4))
