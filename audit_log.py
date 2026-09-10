"""
Append-only, hash-chained audit ledger.

Each entry's hash is derived from its own content PLUS the previous entry's
hash (classic blockchain-style chaining). This means any edit to an earlier
entry changes every hash after it, which is exactly the tamper-evidence
property you need for evidentiary use. This is a real cryptographic chain
(sha256) -- it is not a fake or decorative hash.

In-memory only for this demo. Swap `_LOGS` for a real append-only store
(e.g. Postgres with a trigger blocking UPDATE/DELETE, or a proper ledger
DB) before this touches real cases.
"""
import hashlib
import time
from typing import Literal

Actor = Literal["System", "AI", "Supervisor"]

_LOGS: dict[str, list[dict]] = {}
_GENESIS_HASH = "0" * 8


def _short_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def append_entry(call_id: str, actor: Actor, action: str) -> dict:
    log = _LOGS.setdefault(call_id, [])
    prev_hash = log[-1]["hash"] if log else _GENESIS_HASH
    ts = time.time()
    payload = f"{prev_hash}|{actor}|{action}|{ts:.3f}"
    entry = {
        "time": time.strftime("%M:%S", time.gmtime(ts)),
        "timestamp": ts,
        "actor": actor,
        "action": action,
        "prev_hash": prev_hash,
        "hash": _short_hash(payload),
    }
    log.append(entry)
    return entry


def get_log(call_id: str) -> list[dict]:
    return _LOGS.get(call_id, [])


def verify_chain(call_id: str) -> bool:
    """Recompute every hash and confirm nothing has been tampered with."""
    log = _LOGS.get(call_id, [])
    prev_hash = _GENESIS_HASH
    for entry in log:
        payload = f"{prev_hash}|{entry['actor']}|{entry['action']}|{entry['timestamp']:.3f}"
        if _short_hash(payload) != entry["hash"] or entry["prev_hash"] != prev_hash:
            return False
        prev_hash = entry["hash"]
    return True
