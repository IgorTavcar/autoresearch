"""Clean, easy-to-read results dashboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
RED = "#f85149"

fig = plt.figure(figsize=(18, 16), facecolor=BG)

# Title
fig.text(0.5, 0.975, "Optimizing autoresearch for M4 Max (Metal/MPS)",
         fontsize=24, fontweight="bold", color=TEXT, ha="center", va="top",
         fontfamily="monospace")

# ============================================================
# ROW 1: Depth sweep — horizontal bar chart (easy to read)
# ============================================================
ax1 = fig.add_axes([0.06, 0.72, 0.88, 0.22])
ax1.set_facecolor(CARD)
for s in ax1.spines.values(): s.set_color("#30363d")

depths = [2, 4, 6, 8]
depth_bpb = [1.391, 1.312, 1.418, 1.576]
depth_steps = [1002, 368, 198, 107]
depth_params = ["3.5M", "11.5M", "26.3M", "50.3M"]
d_colors = [BLUE, GREEN, BLUE, RED]
labels = [f"Depth {d}  ({p})" for d, p in zip(depths, depth_params)]

y_pos = range(len(depths))
bars = ax1.barh(y_pos, depth_bpb, color=d_colors, alpha=0.85, height=0.55,
                edgecolor="#30363d")

# Value labels
for i, (b, s) in enumerate(zip(depth_bpb, depth_steps)):
    ax1.text(b + 0.005, i, f" {b:.3f}  ({s} steps)", va="center", fontsize=12,
             fontweight="bold", color=TEXT, fontfamily="monospace")

ax1.set_yticks(y_pos)
ax1.set_yticklabels(labels, fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_xlabel("val_bpb (lower = better)", fontsize=11, color=TEXT_DIM, fontfamily="monospace")
ax1.set_xlim(1.25, 1.7)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.1, color=TEXT_DIM, axis="x")
ax1.invert_yaxis()

# Winner annotation
ax1.text(1.26, 1, "← WINNER", fontsize=11, fontweight="bold", color=GREEN,
         va="center", fontfamily="monospace")

ax1.set_title("TEST 1: Model Depth  —  How many layers?",
              fontsize=14, color=GREEN, fontfamily="monospace", pad=12, loc="left")

# ============================================================
# ROW 2: Batch sweep — horizontal bar chart
# ============================================================
ax2 = fig.add_axes([0.06, 0.38, 0.88, 0.28])
ax2.set_facecolor(CARD)
for s in ax2.spines.values(): s.set_color("#30363d")

b_labels = [
    "Batch 16  / 65K total",
    "Batch 32  / 65K total",
    "Batch 64  / 128K total",
    "Batch 128 / 256K total",
    "Batch 32  / 256K total",
    "Batch 32  / 512K total",
    "Batch 256 / 512K total",
]
b_bpb =   [1.312, 1.309, 1.403, 1.672, 1.706, 1.832, 1.95]
b_steps = [368,   393,   205,   89,    81,    48,    None]
b_colors = [BLUE, GREEN, BLUE, BLUE, BLUE, BLUE, RED]
b_is_oom = [False, False, False, False, False, False, True]

y_pos2 = range(len(b_labels))
bars2 = ax2.barh(y_pos2, b_bpb, color=b_colors, alpha=0.85, height=0.55,
                 edgecolor="#30363d")

for i, (b, s, oom) in enumerate(zip(b_bpb, b_steps, b_is_oom)):
    if oom:
        ax2.text(b + 0.005, i, " OOM! (141GB used)", va="center", fontsize=11,
                 fontweight="bold", color=RED, fontfamily="monospace")
    else:
        ax2.text(b + 0.005, i, f" {b:.3f}  ({s} steps)", va="center", fontsize=11,
                 fontweight="bold", color=TEXT, fontfamily="monospace")

ax2.set_yticks(y_pos2)
ax2.set_yticklabels(b_labels, fontsize=10, color=TEXT, fontfamily="monospace")
ax2.set_xlabel("val_bpb (lower = better)", fontsize=11, color=TEXT_DIM, fontfamily="monospace")
ax2.set_xlim(1.25, 2.05)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.1, color=TEXT_DIM, axis="x")
ax2.invert_yaxis()

ax2.text(1.26, 1, "← WINNER", fontsize=11, fontweight="bold", color=GREEN,
         va="center", fontfamily="monospace")

ax2.set_title("TEST 2: Batch Size  —  How much data per step?",
              fontsize=14, color=BLUE, fontfamily="monospace", pad=12, loc="left")

# ============================================================
# ROW 3: Summary boxes
# ============================================================
boxes = [
    ("BEST CONFIG FOUND",
     "Depth 4 · Batch 32 · Total 65K\n"
     "11.5M params · 393 steps\n"
     "val_bpb = 1.309",
     GREEN),
    ("WHY SMALL WINS",
     "Fixed 5-min budget means:\n"
     "more steps > bigger model.\n"
     "393 steps @ 11.5M beats\n"
     "107 steps @ 50.3M.",
     ORANGE),
    ("STILL TO TEST",
     "• bf16 autocast (2x speed?)\n"
     "• Depth 3 and 5\n"
     "• Aspect ratio tuning\n"
     "• Overnight agent run",
     BLUE),
    ("vs GPU DEFAULT",
     "GPU config on this Mac:\n"
     "val_bpb = 1.576 (20% worse)\n"
     "Only 107 steps in 5 min.\n"
     "Fork wasn't optimized.",
     RED),
]

for i, (title, body, accent) in enumerate(boxes):
    ax = fig.add_axes([0.03 + i * 0.245, 0.03, 0.225, 0.28])
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_color(accent)
        s.set_linewidth(1.5)
        s.set_alpha(0.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(5, 8.5, title, fontsize=11, fontweight="bold", color=accent,
            ha="center", va="center", fontfamily="monospace")
    ax.plot([1, 9], [7.3, 7.3], color="#30363d", linewidth=1)
    ax.text(5, 3.8, body, fontsize=9.5, color=TEXT_DIM, ha="center", va="center",
            fontfamily="monospace", linespacing=1.8)

fig.text(0.98, 0.005, "@danpacary", fontsize=12, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.02, 0.005, "based on karpathy/autoresearch", fontsize=8, color=TEXT_DIM,
         alpha=0.35, ha="left", va="bottom", fontfamily="monospace")

plt.savefig("viz/full_results.png", dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to viz/full_results.png")
