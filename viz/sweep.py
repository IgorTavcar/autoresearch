"""Sweep DEPTH values and record val_bpb for each. Generates a chart."""
import subprocess
import re
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEPTHS = [2, 6, 10]  # depth 4 and 8 already done
TRAIN_PY = "train.py"

# Pre-existing results
results = {
    4: {"val_bpb": 1.312210, "steps": 368, "params_M": 11.5, "tokens_M": 25.1},
    8: None,  # will be filled from run_big.log
}

def read_result(log_path):
    with open(log_path) as f:
        text = f.read()
    def extract(key):
        m = re.search(rf"^{key}:\s+([\d.]+)", text, re.MULTILINE)
        return float(m.group(1)) if m else None
    return {
        "val_bpb": extract("val_bpb"),
        "steps": int(extract("num_steps")) if extract("num_steps") else None,
        "params_M": extract("num_params_M"),
        "tokens_M": extract("total_tokens_M"),
    }

def set_depth(depth):
    with open(TRAIN_PY) as f:
        code = f.read()
    code = re.sub(r"^DEPTH = \d+.*$", f"DEPTH = {depth}               # sweep test", code, flags=re.MULTILINE)
    with open(TRAIN_PY, "w") as f:
        f.write(code)

def run_experiment(depth):
    print(f"\n{'='*60}")
    print(f"  DEPTH = {depth}")
    print(f"{'='*60}")
    set_depth(depth)
    log_path = f"run_depth{depth}.log"
    t0 = time.time()
    proc = subprocess.run(["uv", "run", "train.py"], capture_output=True, text=True, timeout=700)
    with open(log_path, "w") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
    elapsed = time.time() - t0
    result = read_result(log_path)
    print(f"  val_bpb:  {result['val_bpb']}")
    print(f"  steps:    {result['steps']}")
    print(f"  params:   {result['params_M']}M")
    print(f"  tokens:   {result['tokens_M']}M")
    print(f"  wall time: {elapsed:.0f}s")
    return result

# Check if depth-8 result exists
try:
    r8 = read_result("run_big.log")
    if r8["val_bpb"]:
        results[8] = r8
        print(f"Depth 8 (from run_big.log): val_bpb={r8['val_bpb']}, steps={r8['steps']}")
except:
    print("WARNING: run_big.log not found, will run depth 8 too")
    DEPTHS.insert(1, 8)

# Run the sweep
for depth in DEPTHS:
    results[depth] = run_experiment(depth)

# Restore depth 4
set_depth(4)

# --- Plot ---
BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
ACCENT = "#3fb950"
ACCENT2 = "#58a6ff"
HIGHLIGHT = "#f0883e"

depths_sorted = sorted(results.keys())
bpbs = [results[d]["val_bpb"] for d in depths_sorted]
steps = [results[d]["steps"] for d in depths_sorted]
params = [results[d]["params_M"] for d in depths_sorted]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)

# Left: Depth vs val_bpb (the money chart)
ax1.set_facecolor(CARD)
ax1.plot(depths_sorted, bpbs, 'o-', color=ACCENT, markersize=12, linewidth=2.5,
         markeredgecolor="white", markeredgewidth=1.5, zorder=5)

best_depth = depths_sorted[np.argmin(bpbs)]
best_bpb = min(bpbs)
ax1.scatter([best_depth], [best_bpb], s=200, color=HIGHLIGHT, zorder=6,
            edgecolors="white", linewidths=2)
ax1.annotate(f"OPTIMAL: depth={best_depth}\nval_bpb={best_bpb:.4f}",
             xy=(best_depth, best_bpb), xytext=(best_depth + 1, best_bpb + 0.02),
             fontsize=11, color=HIGHLIGHT, fontweight="bold", fontfamily="monospace",
             arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, lw=2))

# Mark GPU default
gpu_idx = depths_sorted.index(8) if 8 in depths_sorted else None
if gpu_idx is not None:
    ax1.annotate("GPU default", xy=(8, bpbs[gpu_idx]),
                 xytext=(8 + 0.5, bpbs[gpu_idx] - 0.015),
                 fontsize=9, color=ACCENT2, fontfamily="monospace",
                 arrowprops=dict(arrowstyle="->", color=ACCENT2, lw=1.5))

for d, b, p in zip(depths_sorted, bpbs, params):
    ax1.annotate(f"{p:.0f}M params", xy=(d, b), xytext=(0, -20),
                 textcoords="offset points", fontsize=8, color=TEXT_DIM,
                 ha="center", fontfamily="monospace")

ax1.set_xlabel("Depth (layers)", fontsize=13, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("val_bpb (lower is better)", fontsize=13, color=TEXT, fontfamily="monospace")
ax1.set_title("Model Depth vs Performance\non Apple M4 Max", fontsize=15, color=TEXT,
              fontfamily="monospace", pad=15)
ax1.set_xticks(depths_sorted)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)
for spine in ax1.spines.values():
    spine.set_color("#30363d")

# Right: Depth vs steps (shows the tradeoff)
ax2.set_facecolor(CARD)
bars = ax2.bar(depths_sorted, steps, color=ACCENT2, alpha=0.8, width=1.2,
               edgecolor="#30363d", linewidth=1)
for d, s in zip(depths_sorted, steps):
    ax2.text(d, s + 8, f"{s}", ha="center", fontsize=11, fontweight="bold",
             color=TEXT, fontfamily="monospace")

ax2.set_xlabel("Depth (layers)", fontsize=13, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("Training Steps (in 5 min)", fontsize=13, color=TEXT, fontfamily="monospace")
ax2.set_title("The Tradeoff: Bigger Model = Fewer Steps\nin Fixed Time Budget", fontsize=15,
              color=TEXT, fontfamily="monospace", pad=15)
ax2.set_xticks(depths_sorted)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM, axis="y")
for spine in ax2.spines.values():
    spine.set_color("#30363d")

fig.suptitle("autoresearch: Finding the Optimal Model Size for Apple Silicon",
             fontsize=18, fontweight="bold", color=TEXT, fontfamily="monospace", y=1.02)

# Watermark
fig.text(0.98, 0.01, "@danpacary", fontsize=11, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.02, 0.01, "based on karpathy/autoresearch", fontsize=8, color=TEXT_DIM, alpha=0.35,
         ha="left", va="bottom", fontfamily="monospace")

plt.tight_layout()
plt.savefig("depth_sweep.png", dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none", pad_inches=0.3)
print(f"\nSaved to depth_sweep.png")

# Print summary table
print(f"\n{'Depth':>6} {'Params':>8} {'Steps':>7} {'val_bpb':>10}")
print("-" * 35)
for d in depths_sorted:
    r = results[d]
    marker = " <-- BEST" if d == best_depth else ""
    print(f"{d:>6} {r['params_M']:>7.1f}M {r['steps']:>7} {r['val_bpb']:>10.6f}{marker}")
