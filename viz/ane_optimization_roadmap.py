"""ANE Optimization Roadmap + Config Selection — M4 Max"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

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

# ═══════════════════════════════════════
# Full sweep data
# ═══════════════════════════════════════
data = {
    (2, 256): 46.0, (2, 512): 105.6, (2, 1024): 283.8,
    (4, 256): 76.7, (4, 512): 162.4, (4, 1024): 406.6,
    (8, 256): 123.7, (8, 512): 269.0, (8, 1024): 592.1,
    (16, 256): 251.4, (16, 512): 521.1, (16, 1024): 971.6,
    (24, 256): 321.1, (24, 512): 637.9, (24, 1024): 1343.6,
}
params = {2: 39.3, 4: 53.5, 8: 81.8, 16: 138.4, 24: 195.1}

# Timing breakdown (NL=8, SEQ=256)
timing_categories = {
    "ANE compute":   {"items": {"ANE fwd": 10.4, "ANE bwd": 15.4}, "color": GREEN},
    "IO / copies":   {"items": {"IO fwd": 3.9, "IO bwd": 12.3, "dw_copy": 7.3}, "color": ORANGE},
    "CPU compute":   {"items": {"classifier": 16.7, "SiLU": 5.8}, "color": RED},
    "CPU math":      {"items": {"RMSNorm": 4.3, "RMS bwd": 2.8}, "color": PURPLE},
}

fig = plt.figure(figsize=(24, 16), facecolor=BG)

# Title
fig.text(0.5, 0.975, "ANE Training — Optimization Roadmap", fontsize=28,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.950, "Where time is spent · What to optimize · Which config to pick",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

# ═══════════════════════════════════════
# Top-left: Current bottleneck breakdown (stacked bar)
# ═══════════════════════════════════════
ax1 = fig.add_axes([0.04, 0.58, 0.44, 0.33])
ax1.set_facecolor(CARD)
for spine in ax1.spines.values():
    spine.set_color("#30363d")

categories = list(timing_categories.keys())
cat_totals = []
cat_colors = []
for cat in categories:
    total = sum(timing_categories[cat]["items"].values())
    cat_totals.append(total)
    cat_colors.append(timing_categories[cat]["color"])

total_time = sum(cat_totals)

# Horizontal stacked bar
left = 0
for i, (cat, total, color) in enumerate(zip(categories, cat_totals, cat_colors)):
    pct = 100 * total / total_time
    bar = ax1.barh(0, total, left=left, color=color, alpha=0.85, height=0.5, edgecolor="#30363d")
    if total > 5:
        ax1.text(left + total/2, 0, f"{cat}\n{total:.0f}ms ({pct:.0f}%)",
                 ha="center", va="center", fontsize=9, fontweight="bold",
                 color="white" if color != PURPLE else TEXT, fontfamily="monospace")
    left += total

# Sub-items below
y_offset = -0.8
for cat in categories:
    color = timing_categories[cat]["color"]
    for name, ms in timing_categories[cat]["items"].items():
        pct = 100 * ms / total_time
        ax1.barh(y_offset, ms, left=0, color=color, alpha=0.5, height=0.3, edgecolor="#30363d")
        ax1.text(ms + 0.5, y_offset, f"{name}: {ms:.1f}ms ({pct:.0f}%)",
                 va="center", fontsize=8, color=TEXT_DIM, fontfamily="monospace")
        y_offset -= 0.4

ax1.set_xlim(0, total_time * 1.4)
ax1.set_ylim(y_offset - 0.3, 0.6)
ax1.set_yticks([])
ax1.set_xlabel("Time (ms)", fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_title("Where Time Goes (NL=8, SEQ=256, per step)", fontsize=14,
              color=TEXT, fontfamily="monospace", pad=10)
ax1.tick_params(colors=TEXT_DIM)

# ═══════════════════════════════════════
# Top-right: Optimization waterfall
# ═══════════════════════════════════════
ax2 = fig.add_axes([0.54, 0.58, 0.42, 0.33])
ax2.set_facecolor(CARD)
for spine in ax2.spines.values():
    spine.set_color("#30363d")

# Waterfall: current → optimizations → target
steps_data = [
    ("Current\n(NL=8 S=256)", 91.4, TEXT_DIM, ""),
    ("Move classifier\nto ANE", -17, RED, "–17ms\n(22%)"),
    ("Batch IO\ncopies", -12, ORANGE, "–12ms\n(15%)"),
    ("Fuse SiLU\ninto ANE", -6, PURPLE, "–6ms\n(8%)"),
    ("Optimized\ntarget", 56.4, GREEN, ""),
]

running = 0
x_positions = []
bar_bottoms = []
bar_heights = []
bar_colors = []

for i, (label, val, color, ann) in enumerate(steps_data):
    if i == 0 or i == len(steps_data) - 1:
        # Total bars
        bar_bottoms.append(0)
        bar_heights.append(val)
        bar_colors.append(color if i > 0 else TEXT_DIM)
        running = val if i == 0 else running
    else:
        # Reduction bars
        bar_bottoms.append(running + val)
        bar_heights.append(-val)
        bar_colors.append(color)
        running += val
    x_positions.append(i)

bars = ax2.bar(x_positions, bar_heights, bottom=bar_bottoms, color=bar_colors,
               alpha=0.8, width=0.65, edgecolor="#30363d", linewidth=1.5)

for i, (label, val, color, ann) in enumerate(steps_data):
    y_top = bar_bottoms[i] + bar_heights[i]
    ax2.text(i, -6, label, ha="center", va="top", fontsize=9, color=TEXT,
             fontfamily="monospace", fontweight="bold")
    if ann:
        ax2.text(i, bar_bottoms[i] + bar_heights[i]/2, ann, ha="center", va="center",
                 fontsize=9, color="white", fontfamily="monospace", fontweight="bold")
    if i == 0:
        ax2.text(i, y_top + 2, f"{val:.0f}ms", ha="center", fontsize=12,
                 color=TEXT, fontfamily="monospace", fontweight="bold")
    elif i == len(steps_data) - 1:
        ax2.text(i, y_top + 2, f"~{val:.0f}ms", ha="center", fontsize=12,
                 color=GREEN, fontfamily="monospace", fontweight="bold")

# Connector lines
for i in range(len(steps_data) - 1):
    y = bar_bottoms[i] + bar_heights[i] if bar_heights[i] > 0 else bar_bottoms[i]
    if i == 0:
        y = bar_heights[0]
    else:
        y = bar_bottoms[i]
    ax2.plot([i + 0.35, i + 0.65], [y, y], '--', color=TEXT_DIM, alpha=0.4, linewidth=1)

# Speedup annotation
ax2.annotate("~1.6x faster\n~40% less time", xy=(4, 70), fontsize=12,
             color=GREEN, fontweight="bold", ha="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD, edgecolor=GREEN, alpha=0.8))

ax2.set_ylim(-12, 105)
ax2.set_ylabel("ms / step", fontsize=11, color=TEXT, fontfamily="monospace")
ax2.set_title("Optimization Waterfall", fontsize=14, color=TEXT, fontfamily="monospace", pad=10)
ax2.set_xticks([])
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.1, color=TEXT_DIM, axis="y")

# ═══════════════════════════════════════
# Bottom-left: Config candidates (bubble chart)
# ═══════════════════════════════════════
ax3 = fig.add_axes([0.04, 0.06, 0.44, 0.42])
ax3.set_facecolor(CARD)
for spine in ax3.spines.values():
    spine.set_color("#30363d")

# Candidates: (NL, SEQ, ms, params, label, color, recommended)
candidates = [
    (2, 1024, 283.8, 39.3, "A", BLUE, False),
    (4, 512, 162.4, 53.5, "B", GREEN, True),     # recommended
    (4, 1024, 406.6, 53.5, "C", BLUE, False),
    (8, 256, 123.7, 81.8, "D", ORANGE, False),
    (8, 512, 269.0, 81.8, "E", ORANGE, False),
]

# Also plot non-candidate configs as dim dots
NLs = [2, 4, 8, 16, 24]
SEQs = [256, 512, 1024]
for nl in NLs:
    for s in SEQs:
        ms = data[(nl, s)]
        # Skip candidates
        is_candidate = any(c[0] == nl and c[1] == s for c in candidates)
        if not is_candidate:
            # tokens in 5 min
            tokens_5m = (300000 / ms) * s * 10 / 1e6
            ax3.scatter(ms, params[nl], s=60, color=TEXT_DIM, alpha=0.2, zorder=1)

for nl, seq, ms, par, label, color, rec in candidates:
    tokens_5m = (300000 / ms) * seq * 10 / 1e6
    size = tokens_5m * 8  # scale bubble by throughput
    edge = "white" if rec else color
    lw = 3 if rec else 1.5
    ax3.scatter(ms, par, s=size, color=color, alpha=0.7, edgecolors=edge,
                linewidths=lw, zorder=5)
    ax3.annotate(f"{label}: NL={nl} S={seq}\n{tokens_5m:.0f}M tok/5min",
                 xy=(ms, par), xytext=(15, 15 if not rec else -25),
                 textcoords="offset points", fontsize=9, color=color,
                 fontfamily="monospace", fontweight="bold" if rec else "normal",
                 arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

# Highlight recommendation
ax3.annotate("RECOMMENDED", xy=(162.4, 53.5), xytext=(40, -45),
             textcoords="offset points", fontsize=13, color=GREEN,
             fontfamily="monospace", fontweight="bold",
             arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=2.5),
             bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD, edgecolor=GREEN))

ax3.set_xlabel("ms / step (faster →)", fontsize=11, color=TEXT, fontfamily="monospace")
ax3.set_ylabel("Params (M)", fontsize=11, color=TEXT, fontfamily="monospace")
ax3.set_title("Config Candidates (bubble = data throughput)", fontsize=14,
              color=TEXT, fontfamily="monospace", pad=10)
ax3.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM)
ax3.invert_xaxis()

# ═══════════════════════════════════════
# Bottom-right: Roadmap timeline
# ═══════════════════════════════════════
ax4 = fig.add_axes([0.54, 0.06, 0.42, 0.42])
ax4.set_facecolor(CARD)
for spine in ax4.spines.values():
    spine.set_color(GREEN)
    spine.set_linewidth(1.5)
    spine.set_alpha(0.5)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_xticks([])
ax4.set_yticks([])

ax4.text(5, 9.5, "OPTIMIZATION ROADMAP", fontsize=16, fontweight="bold",
         color=GREEN, ha="center", fontfamily="monospace")
ax4.plot([0.5, 9.5], [9.0, 9.0], color="#30363d", linewidth=1)

phases = [
    ("PHASE 1: Validate (now)", GREEN, [
        "Generate real training data",
        "Run 5 configs × 5 min each",
        "Pick compute-optimal config",
        "Run overnight ANE training",
    ]),
    ("PHASE 2: Optimize (~1.6x)", ORANGE, [
        "Move classifier to ANE kernel",
        "Batch IOSurface copies",
        "Fuse SiLU into FFN kernel",
        "Target: 56ms/step (from 91ms)",
    ]),
    ("PHASE 3: Scale (SEQ=2048)", BLUE, [
        "Tiled attention (FlashAttention)",
        "Break SRAM wall at SEQ=1024",
        "1:1 comparison with MPS",
        "Target: true Chinchilla-optimal",
    ]),
]

y = 8.4
for phase_name, phase_color, items in phases:
    ax4.text(0.5, y, phase_name, fontsize=11, fontweight="bold",
             color=phase_color, fontfamily="monospace")
    y -= 0.1
    # Phase line
    ax4.plot([0.5, 5.5], [y, y], color=phase_color, linewidth=1, alpha=0.3)
    y -= 0.35
    for item in items:
        ax4.text(1.0, y, f"→ {item}", fontsize=9.5, color=TEXT_DIM,
                 fontfamily="monospace")
        y -= 0.4
    y -= 0.2

# Bottom note
ax4.text(5, 0.3, "6-8% ANE utilization → target 15-20% after Phase 2",
         fontsize=10, color=CYAN, ha="center", fontfamily="monospace", fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD, edgecolor=CYAN, alpha=0.8))

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=11, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch-ANE · Apple Neural Engine training", fontsize=8,
         color=TEXT_DIM, alpha=0.35, ha="left", va="bottom", fontfamily="monospace")

plt.savefig("/Users/dan/Dev/autoresearch-ANE/viz/ane_optimization_roadmap.png",
            dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to viz/ane_optimization_roadmap.png")
