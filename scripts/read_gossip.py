"""Read peer experiments from shared gossip file.

Usage:
    python3 scripts/read_gossip.py --agent ane --n 20
    python3 scripts/read_gossip.py --agent mlx --n 10
    python3 scripts/read_gossip.py --all --n 30
    python3 scripts/read_gossip.py --best
"""
import argparse
import sys

sys.path.insert(0, "scripts")
from gossip_format import read_peer_experiments, read_all_experiments, GOSSIP_FILE


STATUS_ICONS = {
    "keep": "+", "discard": "-", "crash": "X", "baseline": "=",
    "pretest": "~", "partial": "!", "complete": "*",
}


def main():
    parser = argparse.ArgumentParser(description="Read gossip experiments")
    parser.add_argument("--agent", help="Your agent name (shows peer experiments)")
    parser.add_argument("--all", action="store_true", help="Show all agents")
    parser.add_argument("--best", action="store_true", help="Show best per agent")
    parser.add_argument("--n", type=int, default=20, help="Number of entries")
    args = parser.parse_args()

    if args.best:
        entries = read_all_experiments(n=9999)
        best = {}
        for e in entries:
            agent = e["agent"]
            if agent not in best or e["val_bpb"] < best[agent]["val_bpb"]:
                best[agent] = e
        for agent, e in sorted(best.items()):
            print(f"{agent}: val_bpb={e['val_bpb']:.4f} — {e['description']}")
        return

    if args.agent:
        entries = read_peer_experiments(args.agent, args.n)
        print(f"=== Last {len(entries)} peer experiments (you are {args.agent}) ===")
    else:
        entries = read_all_experiments(args.n)
        print(f"=== Last {len(entries)} experiments (all agents) ===")

    for e in entries:
        icon = STATUS_ICONS.get(e["status"], "?")
        print(f"[{icon}] {e['agent']:4s} val_bpb={e['val_bpb']:.4f} — {e['description']}")
        if e.get("lesson"):
            print(f"      lesson: {e['lesson']}")


if __name__ == "__main__":
    main()
