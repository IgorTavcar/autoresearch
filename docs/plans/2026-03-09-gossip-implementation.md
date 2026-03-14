# Multi-Agent Gossip Infrastructure — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build infrastructure for ANE and MLX agents to share experiment findings via a shared JSONL file, bootstrapped with 255+ historical experiments.

**Architecture:** Thin bridge — repos stay separate. A shared JSONL file at `~/.cache/autoresearch/gossip/shared_experiments.jsonl` is readable/writable by both agents. A one-time conversion script bootstraps all historical results. Each agent's `program.md` gets a cross-pollination protocol section.

**Tech Stack:** Python 3 (stdlib only — json, csv, datetime), shell scripts. No new dependencies.

---

### Task 1: Create gossip directory and shared JSONL format

**Files:**
- Create: `~/.cache/autoresearch/gossip/` (directory)
- Create: `scripts/gossip_format.py` (JSONL schema + write helper)

**Step 1: Create the gossip directory**

```bash
mkdir -p ~/.cache/autoresearch/gossip
```

**Step 2: Write the JSONL helper module**

Create `scripts/gossip_format.py`:

```python
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
```

**Step 3: Verify it works**

```bash
cd /Users/dan/Dev/autoresearch-ANE
python3 -c "
import sys; sys.path.insert(0, 'scripts')
from gossip_format import write_experiment, read_all_experiments
write_experiment({'agent': 'test', 'val_bpb': 1.0, 'description': 'test entry'})
print(read_all_experiments())
"
# Clean up test entry
rm ~/.cache/autoresearch/gossip/shared_experiments.jsonl
```

Expected: Prints list with one test entry.

**Step 4: Commit**

```bash
git add scripts/gossip_format.py
git commit -m "feat: add shared JSONL format for multi-agent gossip"
```

---

### Task 2: Convert ANE historical results to JSONL

**Files:**
- Create: `scripts/convert_ane_results.py`
- Read: `results/ane_karpathy_results.tsv` (55 experiments)

**Step 1: Write the ANE converter**

Create `scripts/convert_ane_results.py`:

```python
"""Convert ANE results.tsv to shared gossip JSONL format."""
import csv
import re
import sys
sys.path.insert(0, "scripts")
from gossip_format import write_experiment

TSV = "results/ane_karpathy_results.tsv"


def parse_ane_config(config_str: str) -> dict:
    """Parse config string like NL6_SEQ512_LR3e-4_ACC2_WU25_B299_MLR005_ELR5."""
    config = {}
    parts = config_str.split("_")
    for p in parts:
        if p.startswith("NL"):
            config["n_layers"] = int(p[2:])
        elif p.startswith("SEQ"):
            config["seq"] = int(p[3:])
        elif p.startswith("LR"):
            config["lr"] = float(p[2:])
        elif p.startswith("ACC"):
            config["accum"] = int(p[3:])
        elif p.startswith("WU"):
            config["warmup"] = int(p[2:])
        elif p.startswith("B2"):
            config["beta2"] = float("0." + p[2:])
        elif p.startswith("MLR"):
            config["matrix_lr_scale"] = float("0." + p[3:])
        elif p.startswith("ELR"):
            config["embed_lr_scale"] = float(p[3:])
        elif p.startswith("CL"):
            config["clip"] = float(p[2:])
        elif p.startswith("WD"):
            config["weight_decay"] = float("0." + p[2:]) if p[2:] != "0" else 0.0
    # Defaults for ANE Karpathy config
    config.setdefault("dim", 768)
    config.setdefault("vocab", 8192)
    config.setdefault("params_m", 48.8)
    return config


def parse_steps_from_config(config_str: str) -> int:
    """Extract step count from config string suffix like _1800 or _72K."""
    match = re.search(r"_(\d+)K?$", config_str)
    if match:
        val = int(match.group(1))
        if config_str.endswith("K"):
            return val * 1000
        return val
    return 0


def estimate_wall_sec(steps: int, ms_per_step: float = 139) -> int:
    """Estimate wall time from step count. ANE SEQ=512 ~139ms/step."""
    return int(steps * ms_per_step / 1000)


def convert():
    count = 0
    with open(TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run = row["run"]
            val_bpb = float(row["val_bpb"])
            config_str = row["config"]
            status = row["status"]
            description = row["description"]

            config = parse_ane_config(config_str)
            steps = parse_steps_from_config(config_str)

            # Long runs have known step counts
            if "72K" in config_str or "72000" in description:
                steps = 72000
            elif "177K" in config_str:
                steps = 177000
            elif "10K" in config_str:
                steps = 10000

            # Estimate wall time
            wall_sec = estimate_wall_sec(steps)
            if steps >= 72000:
                wall_sec = estimate_wall_sec(steps, ms_per_step=139)

            # Generate lesson from description
            lesson = ""
            if "diverge" in description.lower() or "x[" in description:
                lesson = "activation instability at this config"
            elif status == "keep":
                lesson = f"improvement: {description}"
            elif status == "discard":
                lesson = f"no improvement: {description}"
            elif status == "baseline":
                lesson = "baseline measurement"

            entry = {
                "ts": "2026-03-09T00:00:00",  # historical, no real timestamp
                "agent": "ane",
                "run_id": run,
                "val_bpb": val_bpb,
                "steps": steps,
                "wall_sec": wall_sec,
                "status": status,
                "config": config,
                "description": description,
                "lesson": lesson,
            }
            write_experiment(entry)
            count += 1
    print(f"Converted {count} ANE experiments")


if __name__ == "__main__":
    convert()
```

**Step 2: Run the converter**

```bash
cd /Users/dan/Dev/autoresearch-ANE
python3 scripts/convert_ane_results.py
wc -l ~/.cache/autoresearch/gossip/shared_experiments.jsonl
```

Expected: "Converted 54 ANE experiments" (header excluded), file has 54 lines.

**Step 3: Spot-check a few entries**

```bash
head -1 ~/.cache/autoresearch/gossip/shared_experiments.jsonl | python3 -m json.tool
tail -1 ~/.cache/autoresearch/gossip/shared_experiments.jsonl | python3 -m json.tool
```

**Step 4: Commit**

```bash
git add scripts/convert_ane_results.py
git commit -m "feat: add ANE results converter for gossip bootstrap"
```

---

### Task 3: Convert MLX historical results to JSONL

**Files:**
- Create: `scripts/convert_mlx_results.py`
- Read: `~/Dev/autoresearch-mlx/results.tsv` (202 experiments)

**Step 1: Write the MLX converter**

Create `scripts/convert_mlx_results.py`:

```python
"""Convert MLX results.tsv to shared gossip JSONL format."""
import csv
import sys
sys.path.insert(0, "scripts")
from gossip_format import write_experiment

TSV_PATH = "/Users/dan/Dev/autoresearch-mlx/results.tsv"


def convert():
    count = 0
    with open(TSV_PATH) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            commit = row["commit"]
            val_bpb = float(row["val_bpb"])
            memory_gb = float(row["memory_gb"])
            status = row["status"]
            description = row["description"]

            # MLX runs 5-min experiments, ~300 wall sec
            # Extract what we can from description
            config = {
                "memory_gb": memory_gb,
                "framework": "mlx",
                "seq": 1024,
                "params_m": 15.7,
            }

            # Generate lesson
            lesson = ""
            if status == "keep":
                lesson = f"improvement: {description}"
            elif status == "discard":
                lesson = f"no improvement: {description}"
            elif status == "crash":
                lesson = f"crashed: {description}"

            entry = {
                "ts": "2026-03-09T00:00:00",
                "agent": "mlx",
                "run_id": commit[:7],
                "val_bpb": val_bpb,
                "steps": 0,  # not tracked in MLX TSV
                "wall_sec": 300,  # 5-min budget
                "status": status,
                "config": config,
                "description": description,
                "lesson": lesson,
            }
            write_experiment(entry)
            count += 1
    print(f"Converted {count} MLX experiments")


if __name__ == "__main__":
    convert()
```

**Step 2: Run the converter**

```bash
cd /Users/dan/Dev/autoresearch-ANE
python3 scripts/convert_mlx_results.py
wc -l ~/.cache/autoresearch/gossip/shared_experiments.jsonl
```

Expected: ~256 total lines (54 ANE + ~202 MLX).

**Step 3: Verify both agents present**

```bash
python3 -c "
import json
agents = {}
with open('$HOME/.cache/autoresearch/gossip/shared_experiments.jsonl') as f:
    for line in f:
        e = json.loads(line)
        a = e['agent']
        agents[a] = agents.get(a, 0) + 1
print(agents)
"
```

Expected: `{'ane': 54, 'mlx': ~202}`

**Step 4: Commit**

```bash
git add scripts/convert_mlx_results.py
git commit -m "feat: add MLX results converter for gossip bootstrap"
```

---

### Task 4: Write the gossip reader script

A standalone script agents can call to see peer findings before each experiment.

**Files:**
- Create: `scripts/read_gossip.py`

**Step 1: Write the reader**

Create `scripts/read_gossip.py`:

```python
"""Read peer experiments from shared gossip file.

Usage:
    python3 scripts/read_gossip.py --agent ane --n 20
    python3 scripts/read_gossip.py --agent mlx --n 10
    python3 scripts/read_gossip.py --all --n 30
    python3 scripts/read_gossip.py --best --n 10
"""
import argparse
import json
import sys
sys.path.insert(0, "scripts")
from gossip_format import read_peer_experiments, read_all_experiments, GOSSIP_FILE


def main():
    parser = argparse.ArgumentParser(description="Read gossip experiments")
    parser.add_argument("--agent", help="Your agent name (shows peer experiments)")
    parser.add_argument("--all", action="store_true", help="Show all agents")
    parser.add_argument("--best", action="store_true", help="Show best results per agent")
    parser.add_argument("--n", type=int, default=20, help="Number of entries")
    args = parser.parse_args()

    if args.best:
        # Read all, find best per agent
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
        status_icon = {"keep": "+", "discard": "-", "crash": "X", "baseline": "=",
                       "pretest": "~", "partial": "!", "complete": "*"}.get(e["status"], "?")
        print(f"[{status_icon}] {e['agent']:4s} val_bpb={e['val_bpb']:.4f} — {e['description']}")
        if e.get("lesson"):
            print(f"      lesson: {e['lesson']}")


if __name__ == "__main__":
    main()
```

**Step 2: Test it**

```bash
python3 scripts/read_gossip.py --best
python3 scripts/read_gossip.py --agent ane --n 5
python3 scripts/read_gossip.py --agent mlx --n 5
```

**Step 3: Commit**

```bash
git add scripts/read_gossip.py
git commit -m "feat: add gossip reader for cross-agent experiment lookup"
```

---

### Task 5: Write the gossip writer script

A standalone script agents call after each experiment to log findings.

**Files:**
- Create: `scripts/log_gossip.py`

**Step 1: Write the logger**

Create `scripts/log_gossip.py`:

```python
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
```

**Step 2: Test it**

```bash
python3 scripts/log_gossip.py \
    --agent ane \
    --val-bpb 1.5 \
    --status keep \
    --description "test log entry" \
    --lesson "testing the logger"

# Verify
tail -1 ~/.cache/autoresearch/gossip/shared_experiments.jsonl | python3 -m json.tool
```

**Step 3: Commit**

```bash
git add scripts/log_gossip.py
git commit -m "feat: add gossip logger for post-experiment recording"
```

---

### Task 6: Update ANE program with gossip protocol

**Files:**
- Read: `docs/plans/2026-03-09-multi-agent-gossip-design.md` (protocol spec)
- Create: `native/scripts/GOSSIP_PROTOCOL.md` (ANE-specific gossip instructions)

**Step 1: Write ANE gossip protocol**

Create `native/scripts/GOSSIP_PROTOCOL.md`:

```markdown
# Cross-Pollination Protocol — ANE Agent

Before proposing your next experiment, check peer findings:

## Before Each Experiment

1. Run: `python3 /Users/dan/Dev/autoresearch-ANE/scripts/read_gossip.py --agent ane --n 20`
2. Look for MLX findings that might transfer (especially "keep" entries)
3. Reason about WHY the finding worked, not just WHAT changed

### What transfers from MLX → ANE (model-level):
- LR ratios between param groups
- Initialization strategies (zero-init, Kaiming, Xavier)
- Architecture choices (VE, attention patterns, MLP width)
- Schedule shapes (warmdown ratio, warmup ratio)
- Regularization (weight decay, softcapping thresholds)

### What does NOT transfer (framework-level):
- Absolute LR values (different optimizers, different batch sizes)
- Step counts (5-min vs overnight)
- Memory constraints (21GB MLX vs ANE's IOSurface model)
- Specific framework features (Muon optimizer is MLX-only)

## After Each Experiment

Log your result:

```bash
python3 /Users/dan/Dev/autoresearch-ANE/scripts/log_gossip.py \
    --agent ane \
    --val-bpb <VAL_BPB> \
    --status <keep|discard|crash> \
    --description "<what you tried>" \
    --lesson "<why it worked or didn't>" \
    --steps <STEPS> \
    --wall-sec <WALL_SEC>
```

The lesson field is the most important — it's what the other agent learns from.
```

**Step 2: Commit**

```bash
git add native/scripts/GOSSIP_PROTOCOL.md
git commit -m "feat: add ANE cross-pollination protocol"
```

---

### Task 7: Update MLX program.md with gossip protocol

**Files:**
- Modify: `/Users/dan/Dev/autoresearch-mlx/program.md` (append gossip section)

**Step 1: Append cross-pollination section to MLX program.md**

Add to the end of `/Users/dan/Dev/autoresearch-mlx/program.md`:

```markdown

## Cross-Pollination Protocol

Before proposing your next experiment, check what the ANE agent has found:

### Before Each Experiment

1. Run: `python3 /Users/dan/Dev/autoresearch-ANE/scripts/read_gossip.py --agent mlx --n 20`
2. Look for ANE findings with status "keep" — these worked on different hardware
3. Reason about WHY the finding worked. ANE uses a different optimizer (pure Adam) and longer runs. Consider framework-specific constraints.

### What transfers from ANE → MLX (model-level):
- LR ratios between param groups (split LR, embedding LR scale)
- Initialization strategies (zero-init output projections)
- Regularization (logit softcapping thresholds, weight decay values)
- Activation stability patterns (if ANE found x>30 at some config, MLX might too)

### What does NOT transfer (framework-level):
- Absolute LR values (MLX uses Muon+AdamW, ANE uses pure Adam)
- Step counts (5-min budget vs overnight)
- Hardware constraints (ANE has fixed IOSurface layout)

### After Each Experiment

Log your result to the shared gossip file:

```bash
python3 /Users/dan/Dev/autoresearch-ANE/scripts/log_gossip.py \
    --agent mlx \
    --val-bpb <VAL_BPB> \
    --status <keep|discard|crash> \
    --description "<what you tried>" \
    --lesson "<why it worked or didn't>"
```

The `lesson` field is the most important — explain WHY your result happened, not just WHAT you changed.
```

**Step 2: Commit in MLX repo**

```bash
cd /Users/dan/Dev/autoresearch-mlx
git add program.md
git commit -m "feat: add cross-pollination gossip protocol"
cd /Users/dan/Dev/autoresearch-ANE
```

---

### Task 8: Bootstrap — run both converters and verify

**Files:**
- Run: `scripts/convert_ane_results.py`
- Run: `scripts/convert_mlx_results.py`

**Step 1: Clear any test data and run fresh**

```bash
rm -f ~/.cache/autoresearch/gossip/shared_experiments.jsonl
python3 scripts/convert_ane_results.py
python3 scripts/convert_mlx_results.py
```

**Step 2: Verify counts and data quality**

```bash
python3 scripts/read_gossip.py --best
python3 scripts/read_gossip.py --all --n 5
wc -l ~/.cache/autoresearch/gossip/shared_experiments.jsonl
```

Expected output:
```
ane:  val_bpb=1.6347 — STABLE full run, best ever ANE result
mlx:  val_bpb=1.266097 — ...
~256 total lines
```

**Step 3: No commit needed (data lives outside git)**

---

### Task 9: Final integration test

**Step 1: Simulate MLX reading ANE findings**

```bash
python3 scripts/read_gossip.py --agent mlx --n 5
```

Should show ANE experiments only.

**Step 2: Simulate ANE reading MLX findings**

```bash
python3 scripts/read_gossip.py --agent ane --n 5
```

Should show MLX experiments only.

**Step 3: Simulate logging a new experiment**

```bash
python3 scripts/log_gossip.py \
    --agent ane \
    --val-bpb 1.5000 \
    --status keep \
    --description "SEQ=1024 first checkpoint" \
    --lesson "2x context gives better bpb at same step count" \
    --steps 5000 \
    --wall-sec 1660

# Verify MLX can see it
python3 scripts/read_gossip.py --agent mlx --n 3
```

Last entry should show the new ANE experiment.

**Step 4: Clean up test entry**

Remove the test line from the JSONL (it's the last line):

```bash
python3 -c "
p = '$HOME/.cache/autoresearch/gossip/shared_experiments.jsonl'
with open(p) as f: lines = f.readlines()
with open(p, 'w') as f: f.writelines(lines[:-1])
"
```

**Step 5: Commit nothing (integration test only)**

---

### Summary: What Gets Built

| File | Purpose |
|------|---------|
| `scripts/gossip_format.py` | JSONL schema + read/write helpers |
| `scripts/convert_ane_results.py` | Bootstrap 54 ANE experiments |
| `scripts/convert_mlx_results.py` | Bootstrap ~202 MLX experiments |
| `scripts/read_gossip.py` | CLI to read peer findings |
| `scripts/log_gossip.py` | CLI to log new experiments |
| `native/scripts/GOSSIP_PROTOCOL.md` | ANE agent gossip instructions |
| MLX `program.md` (modified) | MLX agent gossip instructions |
| `~/.cache/autoresearch/gossip/shared_experiments.jsonl` | The shared data file |

**After this plan:** Both agents can read peer findings before experiments and log results after. The MLX agent can be restarted with the updated `program.md` and will begin cross-pollinating. The ANE agent's next run can include `GOSSIP_PROTOCOL.md` in its prompt.
