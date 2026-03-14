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

            config = {
                "memory_gb": memory_gb,
                "framework": "mlx",
                "seq": 1024,
                "params_m": 15.7,
            }

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
                "steps": 0,
                "wall_sec": 300,
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
