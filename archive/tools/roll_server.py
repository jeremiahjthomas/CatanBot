# roll_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
import json
from pathlib import Path

app = FastAPI()

STATE_PATH = Path(__file__).resolve().parent / "roll_state.json"
TMP_PATH = STATE_PATH.with_suffix(".tmp.json")

class RollEvent(BaseModel):
    type: str
    player: str
    total: int
    ts: int
    raw: Optional[str] = None

def load_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"last_id": 0, "events": []}

def save_state_atomic(state: Dict[str, Any]) -> None:
    data = json.dumps(state, indent=2)
    TMP_PATH.write_text(data, encoding="utf-8")
    TMP_PATH.replace(STATE_PATH)

@app.get("/health")
def health():
    return {"ok": True, "state_path": str(STATE_PATH), "exists": STATE_PATH.exists()}

@app.get("/state")
def state():
    s = load_state()
    return {"last_id": s.get("last_id", 0), "events": len(s.get("events", []))}

@app.post("/roll")
def roll(evt: RollEvent):
    s = load_state()
    s["last_id"] = int(s.get("last_id", 0)) + 1

    record = {
        "id": s["last_id"],
        "type": evt.type,
        "player": evt.player,
        "total": int(evt.total),
        "ts": int(evt.ts),
        "raw": evt.raw,
        "received_ts": int(time.time() * 1000),
    }

    events = list(s.get("events", []))
    events.append(record)
    s["events"] = events[-1000:]  # keep tail

    save_state_atomic(s)
    return {"ok": True, "id": record["id"], "state_path": str(STATE_PATH)}
