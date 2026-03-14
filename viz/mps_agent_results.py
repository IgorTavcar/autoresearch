"""MPS Autoresearch Agent — 73 Experiments Overnight"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Parse results.tsv
experiments = []
with open("/Users/dan/Dev/autoresearch-macos/results.tsv") as f:
    next(f)  # skip header
    for i, line in enumerate(f):
        parts = line.strip().split('\t')
        if len(parts) >= 5:
            experiments.append({
                "num": i + 1,
                "commit": parts[0],
                "val_bpb": float(parts[1]),
                "status": parts[3],
                "desc": parts[4],
            })

# Track the "kept" improvements
kept = [e for e in experiments if e["status"] == "keep"]
discarded = [e for e in experiments if e["status"] == "discard"]
crashed = [e for e in experiments if e["status"] == "crash"]

# Running best line
best_so_far = []
current_best = experiments[0]["val_bpb"]
for e in experiments:
    if e["status"] == "keep":
        current_best = e["val_bpb"]
    best_so_far.append(current_best)

fig = plt.figure(figsize=(24, 14), facecolor=BG)

fig.text(0.5, 0.97, "MPS Autoresearch Agent — Overnight Results", fontsize=28,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.942, "73 experiments, fully autonomous, M4 Max GPU (Metal/MPS)",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

gs = gridspec.GridSpec(2, 2, left=0.06, right=0.97, top=0.91, bottom=0.08,
                       hspace=0.35, wspace=0.25, height_ratios=[1, 1])

# ═══════════════════════════════════════
# Panel 1: All experiments scatter + best line
# ═══════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :], facecolor=CARD)
for spine in ax1.spines.values(): spine.set_color("#30363d")

# Plot all experiments
for e in experiments:
    if e["val_bpb"] == 0 or e["val_bpb"] > 1.35:
        continue  # skip crashes and extreme outliers
    color = GREEN if e["status"] == "keep" else (RED if e["status"] == "crash" else TEXT_DIM)
    alpha = 1.0 if e["status"] == "keep" else 0.4
    size = 100 if e["status"] == "keep" else 35
    ax1.scatter(e["num"], e["val_bpb"], c=color, s=size, alpha=alpha, zorder=3,
                edgecolors="white" if e["status"] == "keep" else "none", linewidth=1.5)

# Best-so-far line
valid_nums = [e["num"] for e in experiments if e["val_bpb"] > 0]
valid_best = [b for e, b in zip(experiments, best_so_far) if e["val_bpb"] > 0]
ax1.plot(valid_nums, valid_best, color=GOLD, linewidth=2.5, alpha=0.9, label="Best so far", zorder=4)

# Annotate kept experiments
for e in kept:
    short = e["desc"].split("(")[0].strip() if "(" in e["desc"] else e["desc"][:25]
    ax1.annotate(short, xy=(e["num"], e["val_bpb"]),
                 xytext=(e["num"] + 2, e["val_bpb"] - 0.003),
                 fontsize=6.5, color=GREEN, fontfamily="monospace",
                 arrowprops=dict(arrowstyle="-", color=GREEN, alpha=0.3, lw=0.5))

# Baseline and H100 reference lines
ax1.axhline(y=1.3157, color=BLUE, linewidth=1, linestyle="--", alpha=0.4)
ax1.text(1, 1.3165, "MPS baseline (1.316)", fontsize=8, color=BLUE, fontfamily="monospace")
ax1.axhline(y=0.998, color=PURPLE, linewidth=1, linestyle="--", alpha=0.3)
ax1.text(1, 0.999, "H100 target (0.998)", fontsize=8, color=PURPLE, fontfamily="monospace")

ax1.set_xlabel("Experiment #", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("val_bpb", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_title("73 Experiments — Every Dot is a 5-min Training Run", fontsize=14,
              color=TEXT, fontfamily="monospace", pad=10)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)
ax1.set_ylim(1.300, 1.345)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN, markersize=8, label=f'Kept ({len(kept)})', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=TEXT_DIM, markersize=6, label=f'Discarded ({len(discarded)})', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=RED, markersize=6, label=f'Crashed ({len(crashed)})', linestyle='None'),
    Line2D([0], [0], color=GOLD, linewidth=2, label='Best so far'),
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=9, facecolor=CARD,
           edgecolor="#30363d", labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 2: Improvement waterfall
# ═══════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 0], facecolor=CARD)
for spine in ax2.spines.values(): spine.set_color("#30363d")

labels = ["Baseline"]
values = [1.3157]
deltas = [0]
colors_bar = [BLUE]

for e in kept[1:]:  # skip baseline
    short = e["desc"].split("(")[0].strip() if "(" in e["desc"] else e["desc"][:20]
    labels.append(short)
    values.append(e["val_bpb"])
    deltas.append(e["val_bpb"] - values[-2] if len(values) > 1 else 0)
    colors_bar.append(GREEN)

y_pos = np.arange(len(labels))
ax2.barh(y_pos, values, color=colors_bar, alpha=0.8, height=0.6, edgecolor="#30363d")

# Add value labels
for i, (v, l) in enumerate(zip(values, labels)):
    ax2.text(v + 0.0002, i, f"{v:.4f}", fontsize=9, color=TEXT, fontfamily="monospace", va="center")
    if i > 0:
        delta = v - values[0]
        ax2.text(values[0] - 0.001, i, f"{delta:+.4f}", fontsize=8, color=GREEN,
                 fontfamily="monospace", va="center", ha="right")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(labels, fontsize=8, color=TEXT, fontfamily="monospace")
ax2.set_xlabel("val_bpb", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_title("Improvement Waterfall — Each Kept Change", fontsize=14,
              color=TEXT, fontfamily="monospace", pad=10)
ax2.tick_params(colors=TEXT_DIM)
ax2.set_xlim(1.305, 1.320)
ax2.invert_yaxis()
ax2.grid(True, alpha=0.15, color=TEXT_DIM, axis="x")

# ═══════════════════════════════════════
# Panel 3: Category analysis
# ═══════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 1], facecolor=CARD)
for spine in ax3.spines.values(): spine.set_color("#30363d")

# Categorize experiments
categories = {
    "Learning rates": [],
    "Optimizer": [],
    "Architecture": [],
    "Schedule": [],
    "Other": [],
}

for e in experiments:
    if e["val_bpb"] == 0:
        continue
    desc = e["desc"].lower()
    if any(w in desc for w in ["lr", "learning rate", "embedding_lr", "unembedding", "scalar_lr", "matrix_lr"]):
        categories["Learning rates"].append(e)
    elif any(w in desc for w in ["adam", "muon", "beta", "momentum", "weight_decay", "weight decay"]):
        categories["Optimizer"].append(e)
    elif any(w in desc for w in ["depth", "aspect", "head_dim", "swiglu", "tied", "window", "rope", "ve ", "softcap", "qk norm", "mlp"]):
        categories["Architecture"].append(e)
    elif any(w in desc for w in ["warmdown", "warmup", "batch", "cosine", "final_lr"]):
        categories["Schedule"].append(e)
    else:
        categories["Other"].append(e)

cat_names = list(categories.keys())
cat_counts = [len(v) for v in categories.values()]
cat_kept = [sum(1 for e in v if e["status"] == "keep") for v in categories.values()]
cat_colors = [BLUE, CYAN, PURPLE, GOLD, TEXT_DIM]

x_pos = np.arange(len(cat_names))
bars1 = ax3.bar(x_pos - 0.15, cat_counts, 0.3, color=cat_colors, alpha=0.4, label="Total tried")
bars2 = ax3.bar(x_pos + 0.15, cat_kept, 0.3, color=cat_colors, alpha=0.9, label="Kept",
                edgecolor="white", linewidth=1)

# Labels on bars
for bar, count in zip(bars1, cat_counts):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             str(count), fontsize=10, color=TEXT_DIM, fontfamily="monospace", ha="center")
for bar, count in zip(bars2, cat_kept):
    if count > 0:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 str(count), fontsize=10, color=GREEN, fontfamily="monospace", ha="center", fontweight="bold")

ax3.set_xticks(x_pos)
ax3.set_xticklabels(cat_names, fontsize=9, color=TEXT, fontfamily="monospace")
ax3.set_ylabel("Experiments", fontsize=11, color=TEXT, fontfamily="monospace")
ax3.set_title("Where Did Improvements Come From?", fontsize=14,
              color=TEXT, fontfamily="monospace", pad=10)
ax3.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM, axis="y")
ax3.legend(loc="upper right", fontsize=9, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=10, color=TEXT_DIM, alpha=0.4,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch-ANE · Karpathy protocol · fully autonomous overnight",
         fontsize=8, color=TEXT_DIM, alpha=0.3, ha="left", va="bottom", fontfamily="monospace")

out = "/Users/dan/Dev/autoresearch-ANE/viz/mps_agent_results.png"
plt.savefig(out, dpi=180, facecolor=BG, edgecolor="none", pad_inches=0.3)
print(f"Saved to {out}")
