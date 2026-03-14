"""ANE Training — Stability Dashboard: All Runs + Search Space"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
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

def parse_tsv(path):
    steps, losses, lrs, xmins, xmaxs = [], [], [], [], []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 5:
                try:
                    steps.append(int(parts[0]))
                    losses.append(float(parts[1]))
                    lrs.append(float(parts[2]))
                    xmins.append(float(parts[3]))
                    xmaxs.append(float(parts[4]))
                except ValueError:
                    pass
    return np.array(steps), np.array(losses), np.array(lrs), np.array(xmins), np.array(xmaxs)

def parse_log(path):
    steps, losses, lrs, xmins, xmaxs = [], [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('step '):
                parts = line.split()
                try:
                    step = int(parts[1])
                    loss = float(parts[2].replace('loss=',''))
                    lr = float(parts[3].replace('lr=',''))
                    m = re.search(r'x\[([^,]+),([^\]]+)\]', line)
                    if m:
                        steps.append(step)
                        losses.append(loss)
                        lrs.append(lr)
                        xmins.append(float(m.group(1)))
                        xmaxs.append(float(m.group(2)))
                except (ValueError, IndexError):
                    pass
    return np.array(steps), np.array(losses), np.array(lrs), np.array(xmins), np.array(xmaxs)

def smooth(arr, window=30):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window)/window, mode='valid')

# ═══════════════════════════════════════
# Load all runs
# ═══════════════════════════════════════
runs = {}

# Overnight v1
s, l, lr, xn, xx = parse_log("/Users/dan/Dev/autoresearch-ANE/results/overnight_ane_nl6_s512.log")
runs["v1"] = {"s": s, "l": l, "lr": lr, "xmin": xn, "xmax": xx,
    "color": RED, "label": "v1: LR=3e-4, 330K steps",
    "status": "DIVERGED", "lr_val": 3e-4, "total_steps": 330000,
    "best_loss": min(l) if len(l) > 0 else 99, "final_x": max(abs(xx[-1]), abs(xn[-1])) if len(xx)>0 else 0}

# Overnight v2
s, l, lr, xn, xx = parse_log("/Users/dan/Dev/autoresearch-ANE/results/overnight_ane_nl6_s512_v2.log")
runs["v2"] = {"s": s, "l": l, "lr": lr, "xmin": xn, "xmax": xx,
    "color": ORANGE, "label": "v2: LR=1e-4, 330K steps",
    "status": "DIVERGED", "lr_val": 1e-4, "total_steps": 330000,
    "best_loss": min(l) if len(l) > 0 else 99, "final_x": max(abs(xx[-1]), abs(xn[-1])) if len(xx)>0 else 0}

# Test D (completed)
if os.path.exists("/tmp/test_d_data.tsv"):
    s, l, lr, xn, xx = parse_tsv("/tmp/test_d_data.tsv")
    if len(s) > 0:
        runs["D"] = {"s": s, "l": l, "lr": lr, "xmin": xn, "xmax": xx,
            "color": GREEN, "label": "D: LR=1e-4, 10K steps",
            "status": "STABLE", "lr_val": 1e-4, "total_steps": 10000,
            "best_loss": min(l), "final_x": max(abs(xx[-1]), abs(xn[-1]))}

# ═══════════════════════════════════════
# Figure
# ═══════════════════════════════════════
fig = plt.figure(figsize=(24, 14), facecolor=BG)

fig.text(0.5, 0.97, "ANE Training — Stability Dashboard", fontsize=28,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.942, "67.6M param GPT · NL=6 SEQ=512 · M4 Max Apple Neural Engine",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

gs = gridspec.GridSpec(2, 3, left=0.06, right=0.97, top=0.90, bottom=0.08,
                       hspace=0.35, wspace=0.3)

# ═══════════════════════════════════════
# Panel 1: Loss curves (top left, spans 2 cols)
# ═══════════════════════════════════════
ax1 = fig.add_subplot(gs[0, :2], facecolor=CARD)
for spine in ax1.spines.values(): spine.set_color("#30363d")

for key in ["v1", "v2", "D"]:
    if key not in runs: continue
    r = runs[key]
    s, l = r["s"], smooth(r["l"], window=50)
    s_plot = s[:len(l)]
    lw = 3 if key == "D" else 2
    ax1.plot(s_plot, l, color=r["color"], linewidth=lw, alpha=0.85, label=r["label"])

# Mark divergence points
for key in ["v1", "v2"]:
    if key not in runs: continue
    r = runs[key]
    x_range = np.maximum(np.abs(r["xmin"]), np.abs(r["xmax"]))
    # Find where x > 50
    diverge_idx = np.where(x_range > 50)[0]
    if len(diverge_idx) > 0:
        div_step = r["s"][diverge_idx[0]]
        div_loss = r["l"][diverge_idx[0]]
        ax1.axvline(x=div_step, color=r["color"], linewidth=1, linestyle=":", alpha=0.5)
        ax1.annotate(f"DIVERGED\nstep {div_step:,}", xy=(div_step, div_loss),
                     xytext=(div_step + 5000, div_loss - 0.5),
                     fontsize=8, color=r["color"], fontfamily="monospace",
                     arrowprops=dict(arrowstyle="->", color=r["color"], lw=1))

ax1.set_xlabel("Step", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("Training Loss", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_title("Loss Curves — Overnight Runs vs Short Tests", fontsize=14,
              color=TEXT, fontfamily="monospace", pad=10)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)
ax1.legend(loc="upper right", fontsize=9, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)
ax1.set_ylim(3.5, 9.5)

# ═══════════════════════════════════════
# Panel 2: Key numbers card (top right)
# ═══════════════════════════════════════
ax_card = fig.add_subplot(gs[0, 2], facecolor=CARD)
for spine in ax_card.spines.values(): spine.set_color("#30363d")
ax_card.set_xlim(0, 10)
ax_card.set_ylim(0, 10)
ax_card.set_xticks([])
ax_card.set_yticks([])

ax_card.text(5, 9.3, "KEY NUMBERS", fontsize=16, fontweight="bold",
             color=GOLD, ha="center", fontfamily="monospace")
ax_card.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

stats = [
    ("67.6M", "parameters", TEXT),
    ("99 ms", "per step (peak)", GREEN),
    ("6 layers", "DIM=768, SEQ=512", TEXT_DIM),
    ("", "", TEXT_DIM),
    ("v1 best", "loss = {:.2f}".format(runs["v1"]["best_loss"]) if "v1" in runs else "?", RED),
    ("v2 best", "loss = {:.2f}".format(runs["v2"]["best_loss"]) if "v2" in runs else "?", ORANGE),
    ("test D", "loss = {:.2f}".format(runs["D"]["best_loss"]) if "D" in runs else "?", GREEN),
    ("", "", TEXT_DIM),
    ("Problem", "activations explode >15K", RED),
    ("Fix", "match --steps to run length", GREEN),
]

y = 8.2
for val, label, color in stats:
    if val == "":
        y -= 0.4
        continue
    ax_card.text(1.0, y, val, fontsize=13, fontweight="bold",
                 color=color, ha="left", fontfamily="monospace")
    ax_card.text(5.0, y, label, fontsize=10,
                 color=TEXT_DIM, ha="left", fontfamily="monospace", va="center")
    y -= 0.75

# ═══════════════════════════════════════
# Panel 3: Activation magnitude (bottom left)
# ═══════════════════════════════════════
ax2 = fig.add_subplot(gs[1, 0], facecolor=CARD)
for spine in ax2.spines.values(): spine.set_color("#30363d")

for key in ["v1", "v2", "D"]:
    if key not in runs: continue
    r = runs[key]
    x_range = np.maximum(np.abs(r["xmin"]), np.abs(r["xmax"]))
    x_smooth = smooth(x_range, window=20)
    s_plot = r["s"][:len(x_smooth)]
    ax2.plot(s_plot, x_smooth, color=r["color"], linewidth=2, alpha=0.85, label=r["label"])

ax2.axhline(y=50, color=RED, linewidth=1.5, linestyle="--", alpha=0.5)
ax2.text(200, 70, "DANGER (>50)", fontsize=9, color=RED, fontfamily="monospace", alpha=0.7)
ax2.axhline(y=10, color=GOLD, linewidth=1, linestyle="--", alpha=0.3)
ax2.text(200, 13, "WARNING (>10)", fontsize=8, color=GOLD, fontfamily="monospace", alpha=0.5)

ax2.set_xlabel("Step", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("|x| max", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_title("Activation Magnitude", fontsize=13, color=TEXT, fontfamily="monospace", pad=10)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM)
ax2.set_yscale("log")
ax2.legend(loc="upper left", fontsize=8, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 4: LR schedules (bottom center)
# ═══════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 1], facecolor=CARD)
for spine in ax3.spines.values(): spine.set_color("#30363d")

for key in ["v1", "v2", "D"]:
    if key not in runs: continue
    r = runs[key]
    ax3.plot(r["s"], r["lr"], color=r["color"], linewidth=2, alpha=0.85, label=r["label"])

ax3.set_xlabel("Step", fontsize=11, color=TEXT, fontfamily="monospace")
ax3.set_ylabel("Learning Rate", fontsize=11, color=TEXT, fontfamily="monospace")
ax3.set_title("LR Schedules — Root Cause", fontsize=13, color=TEXT, fontfamily="monospace", pad=10)
ax3.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM)
ax3.legend(loc="upper right", fontsize=8, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 5: Search space (bottom right)
# ═══════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 2], facecolor=CARD)
for spine in ax4.spines.values(): spine.set_color("#30363d")

configs = [
    (330000, 3e-4, "DIVERGED",  "v1", RED),
    (330000, 1e-4, "DIVERGED",  "v2", ORANGE),
    (3000,   3e-4, "STABLE",    "A: loss=6.3", GREEN),
    (10000,  3e-4, "DIVERGED",  "B: boom 5K", RED),
    (30000,  3e-4, "DIVERGED",  "C: boom 3K", ORANGE),
    (10000,  1e-4, "STABLE",    "D: loss=6.0", GREEN),
    (10000,  5e-5, "TOO SLOW",  "1: barely learned", TEXT_DIM),
    (10000,  2e-4, "TESTING",   "E: running...", CYAN),
]

for ts, lr, outcome, label, color in configs:
    marker = "o" if outcome == "STABLE" else ("D" if outcome == "TESTING" else ("v" if outcome == "TOO SLOW" else "X"))
    size = 250 if outcome in ["STABLE", "TESTING"] else 180
    ax4.scatter(ts, lr, c=color, s=size, marker=marker, zorder=5, linewidth=2)
    xoff = ts * 0.15
    yoff = lr * 0.2
    ax4.annotate(label, xy=(ts, lr), xytext=(ts * 1.15, lr * 1.2),
                 fontsize=8, color=color, fontfamily="monospace", fontweight="bold",
                 arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, lw=0.5))

# Sweet spot region
from matplotlib.patches import FancyBboxPatch
sweet = FancyBboxPatch((5000, 8e-5), 15000, 1.5e-4,
                        boxstyle="round,pad=1500", facecolor=GREEN, alpha=0.06,
                        edgecolor=GREEN, linewidth=1.5, linestyle="--")
ax4.add_patch(sweet)
ax4.text(12000, 1.5e-4, "SWEET\nSPOT?", fontsize=13, color=GREEN, alpha=0.3,
         fontfamily="monospace", fontweight="bold", ha="center", va="center")

ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlabel("total_steps", fontsize=11, color=TEXT, fontfamily="monospace")
ax4.set_ylabel("Learning Rate", fontsize=11, color=TEXT, fontfamily="monospace")
ax4.set_title("Search Space — Where to Look Next", fontsize=13, color=TEXT, fontfamily="monospace", pad=10)
ax4.tick_params(colors=TEXT_DIM)
ax4.grid(True, alpha=0.15, color=TEXT_DIM)

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=GREEN, markersize=10, label='Stable', linestyle='None'),
    Line2D([0], [0], marker='X', color=RED, markersize=10, label='Diverged', linestyle='None'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor=CYAN, markersize=10, label='Testing', linestyle='None'),
    Line2D([0], [0], marker='v', color='w', markerfacecolor=TEXT_DIM, markersize=10, label='Too slow', linestyle='None'),
]
ax4.legend(handles=legend_elements, loc="lower left", fontsize=8, facecolor=CARD,
           edgecolor="#30363d", labelcolor=TEXT, framealpha=0.9)

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=10, color=TEXT_DIM, alpha=0.4,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")

out = "/Users/dan/Dev/autoresearch-ANE/viz/ane_stability_dashboard.png"
plt.savefig(out, dpi=180, facecolor=BG, edgecolor="none", pad_inches=0.3)
print(f"Saved to {out}")
