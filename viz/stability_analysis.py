"""ANE Training Stability Analysis — Finding the Sweet Spot"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re, os

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

def parse_log(path):
    steps, losses, lrs, xmins, xmaxs = [], [], [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                steps.append(int(parts[0]))
                losses.append(float(parts[1]))
                lrs.append(float(parts[2]))
                xmins.append(float(parts[3]))
                xmaxs.append(float(parts[4]))
    return np.array(steps), np.array(losses), np.array(lrs), np.array(xmins), np.array(xmaxs)

def smooth(arr, window=50):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='valid')

# Load all runs
runs = {}

# Overnight v1: LR=3e-4, steps=330K
s, l, lr, xn, xx = parse_log("/tmp/overnight_v1_data.tsv")
runs["v1: LR=3e-4, 330K steps"] = {"s": s, "l": l, "lr": lr, "xmin": xn, "xmax": xx,
    "color": RED, "desc": "Overnight v1\nLR=3e-4, steps=330K\nDIVERGED at ~15K"}

# Overnight v2: LR=1e-4, steps=330K
s, l, lr, xn, xx = parse_log("/tmp/overnight_v2_data.tsv")
runs["v2: LR=1e-4, 330K steps"] = {"s": s, "l": l, "lr": lr, "xmin": xn, "xmax": xx,
    "color": ORANGE, "desc": "Overnight v2\nLR=1e-4, steps=330K\nDIVERGED at ~15K"}

# Test D: LR=1e-4, steps=10K (current test)
if os.path.exists("/tmp/test_d_data.tsv"):
    s, l, lr, xn, xx = parse_log("/tmp/test_d_data.tsv")
    runs["test D: LR=1e-4, 10K steps"] = {"s": s, "l": l, "lr": lr, "xmin": xn, "xmax": xx,
        "color": GREEN, "desc": "Test D\nLR=1e-4, steps=10K\nProper cosine decay"}

# ═══════════════════════════════════════
# Figure: 2×2 grid
# ═══════════════════════════════════════
fig = plt.figure(figsize=(22, 16), facecolor=BG)

fig.text(0.5, 0.97, "ANE Training Stability Analysis", fontsize=28,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.942, "Finding the right LR + cosine schedule for stable overnight training",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

gs = gridspec.GridSpec(2, 2, left=0.07, right=0.97, top=0.91, bottom=0.08,
                       hspace=0.3, wspace=0.25)

# ═══════════════════════════════════════
# Panel 1: Loss curves (all runs)
# ═══════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0], facecolor=CARD)
for spine in ax1.spines.values(): spine.set_color("#30363d")

for name, r in runs.items():
    s, l = r["s"], smooth(r["l"])
    s_plot = s[:len(l)]
    ax1.plot(s_plot, l, color=r["color"], linewidth=2, alpha=0.85, label=name)

ax1.set_xlabel("Step", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("Training Loss", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_title("Loss Curves — All Runs", fontsize=14, color=TEXT, fontfamily="monospace", pad=10)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)
ax1.legend(loc="upper right", fontsize=8, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)
ax1.set_ylim(3.5, 9.5)

# ═══════════════════════════════════════
# Panel 2: Activation magnitude (the instability signal)
# ═══════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1], facecolor=CARD)
for spine in ax2.spines.values(): spine.set_color("#30363d")

for name, r in runs.items():
    s = r["s"]
    x_range = np.maximum(np.abs(r["xmin"]), np.abs(r["xmax"]))
    x_smooth = smooth(x_range, window=30)
    s_plot = s[:len(x_smooth)]
    ax2.plot(s_plot, x_smooth, color=r["color"], linewidth=2, alpha=0.85, label=name)

# Danger zone
ax2.axhline(y=50, color=RED, linewidth=1.5, linestyle="--", alpha=0.5)
ax2.text(100, 55, "DANGER ZONE", fontsize=10, color=RED, fontfamily="monospace", alpha=0.7)
ax2.axhline(y=10, color=GOLD, linewidth=1, linestyle="--", alpha=0.3)
ax2.text(100, 12, "WARNING", fontsize=9, color=GOLD, fontfamily="monospace", alpha=0.5)

ax2.set_xlabel("Step", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("|x| max activation", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_title("Activation Magnitude — Stability Signal", fontsize=14, color=TEXT, fontfamily="monospace", pad=10)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM)
ax2.set_yscale("log")
ax2.legend(loc="upper left", fontsize=8, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 3: LR schedule comparison
# ═══════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0], facecolor=CARD)
for spine in ax3.spines.values(): spine.set_color("#30363d")

for name, r in runs.items():
    ax3.plot(r["s"], r["lr"], color=r["color"], linewidth=2, alpha=0.85, label=name)

ax3.set_xlabel("Step", fontsize=11, color=TEXT, fontfamily="monospace")
ax3.set_ylabel("Learning Rate", fontsize=11, color=TEXT, fontfamily="monospace")
ax3.set_title("LR Schedules — The Root Cause", fontsize=14, color=TEXT, fontfamily="monospace", pad=10)
ax3.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM)
ax3.legend(loc="upper right", fontsize=8, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)

# Annotate the problem
ax3.annotate("LR stays HIGH\ntoo long = BOOM",
             xy=(15000, 2.9e-4), fontsize=10, color=RED,
             fontfamily="monospace", fontweight="bold",
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.5),
             xytext=(50000, 2.5e-4))

# ═══════════════════════════════════════
# Panel 4: The search space — 3D plane concept
# ═══════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1], facecolor=CARD)
for spine in ax4.spines.values(): spine.set_color("#30363d")

# Show the search space as a heatmap/scatter
# LR (y-axis) vs total_steps (x-axis), color = outcome
configs = [
    # (total_steps, lr, outcome, label, color)
    (330000, 3e-4, "DIVERGED",   "v1: boom at 15K", RED),
    (330000, 1e-4, "DIVERGED",   "v2: boom at 15K", ORANGE),
    (3000,   3e-4, "STABLE",     "Test A: stable, loss=6.3", GREEN),
    (10000,  3e-4, "DIVERGED",   "Test B: boom at 5K", RED),
    (30000,  3e-4, "DIVERGED",   "Test C: boom at 3K", ORANGE),
    (10000,  1e-4, "TESTING",    "Test D: running...", CYAN),
    (10000,  5e-5, "TOO SLOW",   "Test 1: barely learned", TEXT_DIM),
]

for ts, lr, outcome, label, color in configs:
    marker = "o" if outcome == "STABLE" else ("x" if "DIVERGED" in outcome else "s")
    size = 200 if outcome == "STABLE" else (150 if outcome == "TESTING" else 120)
    ax4.scatter(ts, lr, c=color, s=size, marker=marker, zorder=5, edgecolors="white", linewidth=1.5)
    # Label offset
    xoff = ts * 0.1
    yoff = lr * 0.15
    ax4.annotate(label, xy=(ts, lr), xytext=(ts + xoff, lr + yoff),
                 fontsize=7.5, color=color, fontfamily="monospace",
                 arrowprops=dict(arrowstyle="-", color=color, alpha=0.3, lw=0.5))

# Shade the "sweet spot" region
from matplotlib.patches import FancyBboxPatch
sweet = FancyBboxPatch((2000, 5e-5), 12000, 1.5e-4,
                        boxstyle="round,pad=1000", facecolor=GREEN, alpha=0.08,
                        edgecolor=GREEN, linewidth=1.5, linestyle="--")
ax4.add_patch(sweet)
ax4.text(8000, 1.2e-4, "SWEET\nSPOT?", fontsize=14, color=GREEN, alpha=0.4,
         fontfamily="monospace", fontweight="bold", ha="center", va="center")

ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlabel("total_steps (cosine schedule length)", fontsize=11, color=TEXT, fontfamily="monospace")
ax4.set_ylabel("Learning Rate", fontsize=11, color=TEXT, fontfamily="monospace")
ax4.set_title("Search Space — LR vs Schedule Length", fontsize=14, color=TEXT, fontfamily="monospace", pad=10)
ax4.tick_params(colors=TEXT_DIM)
ax4.grid(True, alpha=0.15, color=TEXT_DIM)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN, markersize=10, label='Stable'),
    Line2D([0], [0], marker='x', color=RED, markersize=10, label='Diverged', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=CYAN, markersize=10, label='Testing'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor=TEXT_DIM, markersize=10, label='Too slow'),
]
ax4.legend(handles=legend_elements, loc="lower right", fontsize=9, facecolor=CARD,
           edgecolor="#30363d", labelcolor=TEXT, framealpha=0.9)

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=10, color=TEXT_DIM, alpha=0.4,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")

out = "/Users/dan/Dev/autoresearch-ANE/viz/stability_analysis.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG, edgecolor="none", pad_inches=0.3)
print(f"Saved to {out}")
