"""Shared experiment format for multi-agent gossip.

Each line in shared_experiments.jsonl is a JSON object:
{
    "ts": "2026-03-09T19:30:00",
    "agent": "ane" | "mlx" | "mps",
    "val_bpb": 1.635,
    "steps": 72000,
    "wall_sec": 17280,
    "status": "keep" | "discard" | "crash" | "baseline" | "pretest" | "partial" | "complete",
    "config": {"lr": 2.5e-4, "seq": 512, ...},
    "description": "v3b: half LR + zero-init + softcap + split LR",
    "lesson": "activation stability requires lower LR than short runs suggest"
}
"""
import json
from datetime import datetime
from pathlib import Path

GOSSIP_DIR = Path.home() / ".cache" / "autoresearch" / "gossip"
GOSSIP_FILE = GOSSIP_DIR / "shared_experiments.jsonl"


def write_experiment(entry: dict) -> None:
    """Append one experiment to shared_experiments.jsonl."""
    GOSSIP_DIR.mkdir(parents=True, exist_ok=True)
    if "ts" not in entry:
        entry["ts"] = datetime.now().isoformat(timespec="seconds")
    with open(GOSSIP_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def read_peer_experiments(my_agent: str, n: int = 20) -> list[dict]:
    """Read last N experiments from OTHER agents."""
    if not GOSSIP_FILE.exists():
        return []
    entries = []
    with open(GOSSIP_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("agent") != my_agent:
                entries.append(entry)
    return entries[-n:]


def read_all_experiments(n: int = 50) -> list[dict]:
    """Read last N experiments from ALL agents."""
    if not GOSSIP_FILE.exists():
        return []
    entries = []
    with open(GOSSIP_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries[-n:]
