"""Three Accelerators, One Chip — M4 Max Comparison"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np

# Colors
BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
RED = "#f85149"
PURPLE = "#bc8cff"
CYAN = "#39d2c0"
GOLD = "#e3b341"

# ═══════════════════════════════════════
# Data
# ═══════════════════════════════════════

# MPS experiments (from results.tsv)
mps_experiments = []
with open("/Users/dan/Dev/autoresearch-macos/results.tsv") as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            mps_experiments.append({
                "num": i + 1,
                "val_bpb": float(parts[1]),
                "status": parts[3],
                "desc": parts[4],
            })

mps_kept = [e for e in mps_experiments if e["status"] == "keep"]
mps_best_line = []
current_best = mps_experiments[0]["val_bpb"]
for e in mps_experiments:
    if e["status"] == "keep":
        current_best = e["val_bpb"]
    mps_best_line.append(current_best)

# MLX experiments
mlx_experiments = []
with open("/Users/dan/Dev/autoresearch-mlx/results.tsv") as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            mlx_experiments.append({
                "num": i + 1,
                "val_bpb": float(parts[1]),
                "status": parts[3],
                "desc": parts[4],
            })

# ANE test results (training loss, not val_bpb — different scale)
ane_tests = {
    "E": {"lr": 2e-4, "steps": 10000, "best_loss": 5.72, "ms_step": 109, "status": "STABLE", "x_range": 1.5},
    "F": {"lr": 1e-4, "steps": 30000, "best_loss": 5.70, "ms_step": 170, "status": "STABLE", "x_range": 2.5},
    "v1": {"lr": 3e-4, "steps": 330000, "best_loss": 5.86, "ms_step": 99, "status": "DIVERGED", "x_range": 481},
    "v2": {"lr": 1e-4, "steps": 330000, "best_loss": 5.90, "ms_step": 99, "status": "DIVERGED", "x_range": 227},
}

# ═══════════════════════════════════════
# Figure
# ═══════════════════════════════════════
fig = plt.figure(figsize=(26, 16), facecolor=BG)

fig.text(0.5, 0.97, "Three Accelerators, One Chip", fontsize=32,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.945, "M4 Max — ANE (Neural Engine) + MLX (GPU) + MPS (GPU, retired)",
         fontsize=14, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

gs = gridspec.GridSpec(2, 3, left=0.06, right=0.97, top=0.90, bottom=0.08,
                       hspace=0.35, wspace=0.3)

# ═══════════════════════════════════════
# Panel 1: MPS journey (top left, 2 cols)
# ═══════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :2], facecolor=CARD)
for spine in ax1.spines.values(): spine.set_color("#30363d")

for e in mps_experiments:
    if e["val_bpb"] == 0 or e["val_bpb"] > 1.4:
        continue
    color = GREEN if e["status"] == "keep" else TEXT_DIM
    alpha = 1.0 if e["status"] == "keep" else 0.3
    size = 80 if e["status"] == "keep" else 25
    ax1.scatter(e["num"], e["val_bpb"], c=color, s=size, alpha=alpha, zorder=3,
                edgecolors="white" if e["status"] == "keep" else "none", linewidth=1)

valid_nums = [e["num"] for e in mps_experiments if 0 < e["val_bpb"] < 1.4]
valid_best = [b for e, b in zip(mps_experiments, mps_best_line) if 0 < e["val_bpb"] < 1.4]
ax1.plot(valid_nums, valid_best, color=GOLD, linewidth=2.5, alpha=0.9, label="Best so far", zorder=4)

ax1.axhline(y=1.3157, color=BLUE, linewidth=1, linestyle="--", alpha=0.4)
ax1.text(1, 1.317, "baseline (1.316)", fontsize=7, color=BLUE, fontfamily="monospace")

ax1.set_xlabel("Experiment #", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("val_bpb", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_title("MPS Agent — 79 Experiments (PyTorch, retired)", fontsize=14,
              color=ORANGE, fontfamily="monospace", pad=10)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)
ax1.set_ylim(1.300, 1.340)

from matplotlib.lines import Line2D
leg1 = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN, markersize=8,
           label=f'Kept ({len(mps_kept)})', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TEXT_DIM, markersize=5,
           label=f'Discarded ({len(mps_experiments) - len(mps_kept)})', linestyle='None'),
    Line2D([0], [0], color=GOLD, linewidth=2, label='Best so far'),
]
ax1.legend(handles=leg1, loc="upper right", fontsize=8, facecolor=CARD,
           edgecolor="#30363d", labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 2: Scoreboard (top right)
# ═══════════════════════════════════════
ax_card = fig.add_subplot(gs[0, 2], facecolor=CARD)
for spine in ax_card.spines.values(): spine.set_color("#30363d")
ax_card.set_xlim(0, 10)
ax_card.set_ylim(0, 10)
ax_card.set_xticks([])
ax_card.set_yticks([])

ax_card.text(5, 9.3, "SCOREBOARD", fontsize=18, fontweight="bold",
             color=GOLD, ha="center", fontfamily="monospace")
ax_card.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

rows = [
    ("", "Framework", "Params", "Speed", "#30363d"),
    ("ANE", "Obj-C", "67.6M", "99ms", CYAN),
    ("MLX", "MLX", "~50M", "800ms", GREEN),
    ("MPS", "PyTorch", "11.5M", "764ms", ORANGE),
    ("", "", "", "", ""),
    ("", "Metric", "Best", "Expts", "#30363d"),
    ("ANE*", "train loss", "5.70", "6", CYAN),
    ("MLX", "val_bpb", f"{mlx_experiments[0]['val_bpb']:.3f}" if mlx_experiments else "?", str(len(mlx_experiments)), GREEN),
    ("MPS", "val_bpb", "1.308", "79", ORANGE),
    ("H100", "val_bpb", "0.970", "—", PURPLE),
]

y = 8.2
for col1, col2, col3, col4, color in rows:
    if col1 == "" and col2 == "":
        y -= 0.3
        continue
    if col2 == "Framework" or col2 == "Metric":
        # Header row
        ax_card.text(0.5, y, col1, fontsize=9, color="#30363d", ha="left", fontfamily="monospace", fontweight="bold")
        ax_card.text(2.8, y, col2, fontsize=9, color="#30363d", ha="left", fontfamily="monospace", fontweight="bold")
        ax_card.text(5.8, y, col3, fontsize=9, color="#30363d", ha="left", fontfamily="monospace", fontweight="bold")
        ax_card.text(8.0, y, col4, fontsize=9, color="#30363d", ha="left", fontfamily="monospace", fontweight="bold")
        y -= 0.65
        continue
    ax_card.text(0.5, y, col1, fontsize=12, fontweight="bold", color=color, ha="left", fontfamily="monospace")
    ax_card.text(2.8, y, col2, fontsize=10, color=TEXT_DIM, ha="left", fontfamily="monospace")
    ax_card.text(5.8, y, col3, fontsize=11, fontweight="bold", color=color, ha="left", fontfamily="monospace")
    ax_card.text(8.0, y, col4, fontsize=10, color=TEXT_DIM, ha="left", fontfamily="monospace")
    y -= 0.75

# ANE caveat note
ax_card.text(5, 2.4, "*ANE uses different data/tokenizer/eval", fontsize=7,
             color=RED, ha="center", fontfamily="monospace", alpha=0.7, fontstyle="italic")
ax_card.text(5, 1.9, "  — not directly comparable to val_bpb", fontsize=7,
             color=RED, ha="center", fontfamily="monospace", alpha=0.5, fontstyle="italic")

# Key insight box
ax_card.add_patch(mpatches.FancyBboxPatch((0.5, 0.2), 9, 1.4,
    boxstyle="round,pad=0.3", facecolor=GREEN, alpha=0.06,
    edgecolor=GREEN, linewidth=1, linestyle="--"))
ax_card.text(5, 1.2, "ANE + GPU run simultaneously", fontsize=9,
             color=GREEN, ha="center", fontfamily="monospace", fontweight="bold", alpha=0.8)
ax_card.text(5, 0.6, "zero interference, free compute", fontsize=9,
             color=GREEN, ha="center", fontfamily="monospace", alpha=0.6)

# ═══════════════════════════════════════
# Panel 3: ANE stability map (bottom left)
# ═══════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 0], facecolor=CARD)
for spine in ax2.spines.values(): spine.set_color("#30363d")

configs = [
    (330000, 3e-4, "DIVERGED", "v1", RED),
    (330000, 1e-4, "DIVERGED", "v2", ORANGE),
    (3000,   3e-4, "STABLE",  "A", GREEN),
    (10000,  3e-4, "DIVERGED", "B", RED),
    (30000,  3e-4, "DIVERGED", "C", ORANGE),
    (10000,  1e-4, "STABLE",  "D", GREEN),
    (10000,  2e-4, "STABLE",  "E", GREEN),
    (30000,  1e-4, "STABLE",  "F", GREEN),
    (10000,  5e-5, "SLOW",    "1", TEXT_DIM),
    (5000,   3e-4, "QUEUED",  "G", CYAN),
]

for ts, lr, outcome, label, color in configs:
    marker = "o" if outcome == "STABLE" else ("X" if outcome == "DIVERGED" else ("v" if outcome == "SLOW" else "D"))
    size = 200 if outcome == "STABLE" else 140
    ax2.scatter(ts, lr, c=color, s=size, marker=marker, zorder=5, linewidth=2)
    ax2.annotate(label, xy=(ts, lr), xytext=(ts * 1.2, lr * 1.15),
                 fontsize=9, color=color, fontfamily="monospace", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=color, alpha=0.3, lw=0.5))

# Sweet spot
from matplotlib.patches import FancyBboxPatch
sweet = FancyBboxPatch((5000, 8e-5), 15000, 1.5e-4,
    boxstyle="round,pad=1500", facecolor=GREEN, alpha=0.06,
    edgecolor=GREEN, linewidth=1.5, linestyle="--")
ax2.add_patch(sweet)
ax2.text(12000, 1.5e-4, "SWEET\nSPOT", fontsize=11, color=GREEN, alpha=0.3,
         fontfamily="monospace", fontweight="bold", ha="center", va="center")

ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel("total_steps", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("Learning Rate", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_title("ANE — Stability Search Space (different data*)", fontsize=13,
              color=CYAN, fontfamily="monospace", pad=10)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM)

leg2 = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN, markersize=9, label='Stable', linestyle='None'),
    Line2D([0], [0], marker='X', color=RED, markersize=9, label='Diverged', linestyle='None'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=CYAN, markersize=8, label='Queued', linestyle='None'),
]
ax2.legend(handles=leg2, loc="lower left", fontsize=8, facecolor=CARD,
           edgecolor="#30363d", labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 4: Hardware comparison bars (bottom center)
# ═══════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 1], facecolor=CARD)
for spine in ax3.spines.values(): spine.set_color("#30363d")

# Speed comparison
accel = ["ANE\n(Neural Engine)", "MLX\n(GPU)", "MPS\n(GPU)"]
ms_per_step = [99, 800, 764]
params_m = [67.6, 50, 11.5]
colors_bar = [CYAN, GREEN, ORANGE]

x = np.arange(len(accel))
width = 0.35

bars1 = ax3.bar(x - width/2, ms_per_step, width, color=colors_bar, alpha=0.7, label="ms/step")

ax3_twin = ax3.twinx()
bars2 = ax3_twin.bar(x + width/2, params_m, width, color=colors_bar, alpha=0.35,
                     edgecolor=colors_bar, linewidth=2, linestyle="--", label="Params (M)")

for bar, val in zip(bars1, ms_per_step):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
             f"{val}ms", fontsize=10, color=TEXT, fontfamily="monospace",
             ha="center", fontweight="bold")

for bar, val in zip(bars2, params_m):
    ax3_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                  f"{val}M", fontsize=10, color=TEXT_DIM, fontfamily="monospace",
                  ha="center")

ax3.set_xticks(x)
ax3.set_xticklabels(accel, fontsize=10, color=TEXT, fontfamily="monospace")
ax3.set_ylabel("ms/step (lower = faster)", fontsize=10, color=TEXT, fontfamily="monospace")
ax3_twin.set_ylabel("Parameters (M)", fontsize=10, color=TEXT_DIM, fontfamily="monospace")
ax3.set_title("Speed vs Model Size", fontsize=14,
              color=TEXT, fontfamily="monospace", pad=10)
ax3.tick_params(colors=TEXT_DIM)
ax3_twin.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM, axis="y")

# ═══════════════════════════════════════
# Panel 5: Architecture comparison (bottom right)
# ═══════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 2], facecolor=CARD)
for spine in ax4.spines.values(): spine.set_color("#30363d")
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_xticks([])
ax4.set_yticks([])

ax4.text(5, 9.3, "ARCHITECTURE COMPARISON", fontsize=13, fontweight="bold",
         color=TEXT, ha="center", fontfamily="monospace")
ax4.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

features = [
    ("Feature", "ANE", "MLX", "MPS"),
    ("ReluSquared", "—", "Y", "Y"),
    ("Value Embeds", "—", "Y", "Y"),
    ("Resid Lambdas", "—", "Y", "Y"),
    ("QK Norm", "—", "Y", "Y"),
    ("Logit Softcap", "—", "Y", "—"),
    ("Sliding Window", "—", "Y", "Y"),
    ("Separate LRs", "—", "4", "4"),
    ("bf16", "fp32", "Y", "SLOW"),
    ("Muon Optimizer", "—", "—", "Y"),
    ("Karpathy Data", "—", "Y", "Y"),
]

y = 8.3
for feat, ane, mlx, mps in features:
    if feat == "Feature":
        ax4.text(0.3, y, feat, fontsize=9, color="#30363d", fontfamily="monospace", fontweight="bold")
        ax4.text(4.5, y, "ANE", fontsize=9, color=CYAN, fontfamily="monospace", fontweight="bold", ha="center")
        ax4.text(6.5, y, "MLX", fontsize=9, color=GREEN, fontfamily="monospace", fontweight="bold", ha="center")
        ax4.text(8.5, y, "MPS", fontsize=9, color=ORANGE, fontfamily="monospace", fontweight="bold", ha="center")
        y -= 0.7
        continue

    ax4.text(0.3, y, feat, fontsize=9, color=TEXT_DIM, fontfamily="monospace")

    for val, xpos in [(ane, 4.5), (mlx, 6.5), (mps, 8.5)]:
        if val == "Y":
            color = GREEN
        elif val == "—":
            color = "#30363d"
        elif val == "SLOW":
            color = RED
        else:
            color = TEXT_DIM
        ax4.text(xpos, y, val, fontsize=9, color=color, fontfamily="monospace",
                 ha="center", fontweight="bold" if val == "Y" else "normal")
    y -= 0.68

# Center watermark
fig.text(0.5, 0.50, "@danpacary", fontsize=60, color=TEXT_DIM, alpha=0.04,
         ha="center", va="center", fontfamily="monospace", fontweight="bold",
         rotation=25, zorder=0)

# Bottom watermarks
fig.text(0.97, 0.005, "@danpacary", fontsize=10, color=TEXT_DIM, alpha=0.4,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch-ANE · M4 Max · three accelerators, one chip",
         fontsize=8, color=TEXT_DIM, alpha=0.3, ha="left", va="bottom", fontfamily="monospace")

out = "/Users/dan/Dev/autoresearch-ANE/viz/three_accelerator_comparison.png"
plt.savefig(out, dpi=180, facecolor=BG, edgecolor="none", pad_inches=0.3)
print(f"Saved to {out}")
