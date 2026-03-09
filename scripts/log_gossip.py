"""Log an experiment to the shared gossip file.

Usage:
    python3 scripts/log_gossip.py \
        --agent mlx \
        --val-bpb 1.266 \
        --status keep \
        --description "WARMDOWN_RATIO=0.6 + EMBEDDING_LR=1.3" \
        --lesson "removing softcap lets embedding LR go higher" \
        --steps 1848 \
        --wall-sec 300
"""
import argparse
import json
import sys

sys.path.insert(0, "scripts")
from gossip_format import write_experiment


def main():
    parser = argparse.ArgumentParser(description="Log experiment to gossip")
    parser.add_argument("--agent", required=True, choices=["ane", "mlx", "mps"])
    parser.add_argument("--val-bpb", required=True, type=float)
    parser.add_argument("--status", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--lesson", default="")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--wall-sec", type=int, default=0)
    parser.add_argument("--config-json", default="{}", help="JSON string of config")
    args = parser.parse_args()

    entry = {
        "agent": args.agent,
        "val_bpb": args.val_bpb,
        "steps": args.steps,
        "wall_sec": args.wall_sec,
        "status": args.status,
        "config": json.loads(args.config_json),
        "description": args.description,
        "lesson": args.lesson,
    }
    write_experiment(entry)
    print(f"Logged: [{args.agent}] val_bpb={args.val_bpb} — {args.description}")


if __name__ == "__main__":
    main()
