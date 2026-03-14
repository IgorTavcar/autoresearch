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
    for p in config_str.split("_"):
        if p.startswith("NL"):
            config["n_layers"] = int(p[2:])
        elif p.startswith("SEQ"):
            config["seq"] = int(p[3:])
        elif p.startswith("LR") and not p.startswith("LR0"):
            try:
                config["lr"] = float(p[2:])
            except ValueError:
                pass
        elif p.startswith("ACC"):
            try:
                config["accum"] = int(p[3:])
            except ValueError:
                pass
        elif p.startswith("WU"):
            try:
                config["warmup"] = int(p[2:])
            except ValueError:
                pass
        elif p.startswith("B2"):
            config["beta2"] = float("0." + p[2:])
        elif p.startswith("MLR"):
            config["matrix_lr_scale"] = float("0." + p[3:])
        elif p.startswith("ELR"):
            try:
                config["embed_lr_scale"] = float(p[3:])
            except ValueError:
                pass
        elif p.startswith("CL"):
            try:
                config["clip"] = float(p[2:])
            except ValueError:
                pass
        elif p.startswith("WD"):
            config["weight_decay"] = float("0." + p[2:]) if p[2:] != "0" else 0.0
    config.setdefault("dim", 768)
    config.setdefault("vocab", 8192)
    config.setdefault("params_m", 48.8)
    return config


def extract_steps(config_str: str) -> int:
    """Extract step count from config string suffix."""
    # Check for explicit K suffix
    m = re.search(r"_(\d+)K$", config_str)
    if m:
        return int(m.group(1)) * 1000
    # Check for bare number at end (step count)
    m = re.search(r"_(\d{3,})$", config_str)
    if m:
        return int(m.group(1))
    return 0


def estimate_wall_sec(steps: int) -> int:
    """Estimate wall time. ANE SEQ=512 ~139ms/step for 5-min runs, longer for overnight."""
    if steps <= 2000:
        return int(steps * 139 / 1000)  # 5-min sweep runs
    return int(steps * 139 / 1000)


def convert():
    count = 0
    with open(TSV) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            run_id = row["run"]
            val_bpb = float(row["val_bpb"])
            config_str = row["config"]
            status = row["status"]
            description = row["description"]

            config = parse_ane_config(config_str)
            steps = extract_steps(config_str)

            # Override for known long runs
            if "72K" in config_str or "72K" in description or "72000" in description:
                steps = 72000
            elif "177K" in config_str:
                steps = 177000
            elif "10K" in config_str and steps == 0:
                steps = 10000

            wall_sec = estimate_wall_sec(steps)

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
            elif status == "complete":
                lesson = f"completed run: {description}"
            elif status == "partial":
                lesson = f"partial run: {description}"

            entry = {
                "ts": "2026-03-09T00:00:00",
                "agent": "ane",
                "run_id": run_id,
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
