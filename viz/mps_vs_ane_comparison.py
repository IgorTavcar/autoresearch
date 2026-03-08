"""MPS Pre-test Results + ANE Benchmark Comparison — M4 Max (128GB)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
# MPS Data
# ═══════════════════════════════════════
mps_depth = {
    2: {"params": 3.5, "steps": 1002, "val_bpb": 1.391},
    4: {"params": 11.5, "steps": 368, "val_bpb": 1.312},
    6: {"params": 26.3, "steps": 198, "val_bpb": 1.418},
    8: {"params": 50.3, "steps": 107, "val_bpb": 1.576},
}

mps_batch = [
    (16, "65K", 368, 1.312, "Fork default"),
    (32, "65K", 393, 1.309, "BEST"),
    (64, "128K", 205, 1.403, ""),
    (128, "256K", 89, 1.672, ""),
    (256, "512K", 0, None, "OOM (141GB)"),
    (32, "256K", 81, 1.706, "Big total"),
    (32, "512K", 48, 1.832, "GPU-level total"),
]

# ANE Data
ane_data = {
    (2, 256): 46.0, (2, 512): 105.6, (2, 1024): 283.8,
    (4, 256): 76.7, (4, 512): 162.4, (4, 1024): 406.6,
    (8, 256): 123.7, (8, 512): 269.0, (8, 1024): 592.1,
    (16, 256): 251.4, (16, 512): 521.1, (16, 1024): 971.6,
    (24, 256): 321.1, (24, 512): 637.9, (24, 1024): 1343.6,
}
ane_params = {2: 39.3, 4: 53.5, 8: 81.8, 16: 138.4, 24: 195.1}

# H100 reference
h100 = {"val_bpb": 0.998, "steps": 953, "tokens_M": 499.6, "params": 50.3}

fig = plt.figure(figsize=(26, 18), facecolor=BG)

# Title
fig.text(0.5, 0.975, "M4 Max Training Profile — MPS vs ANE vs H100", fontsize=28,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.952, "Pre-test sweep results · Same hardware, different accelerators · 5-minute budget",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

# ═══════════════════════════════════════
# Row 1 Left: MPS Depth Sweep
# ═══════════════════════════════════════
ax1 = fig.add_axes([0.04, 0.66, 0.28, 0.25])
ax1.set_facecolor(CARD)
for spine in ax1.spines.values():
    spine.set_color("#30363d")

depths = sorted(mps_depth.keys())
bpbs = [mps_depth[d]["val_bpb"] for d in depths]
steps = [mps_depth[d]["steps"] for d in depths]

ax1.plot(depths, bpbs, 'o-', color=BLUE, markersize=12, linewidth=2.5,
         markeredgecolor="white", markeredgewidth=1.5, zorder=5)

best_idx = np.argmin(bpbs)
ax1.scatter([depths[best_idx]], [bpbs[best_idx]], s=200, color=GREEN, zorder=6,
            edgecolors="white", linewidths=2)
ax1.annotate(f"Best: D={depths[best_idx]}\n{bpbs[best_idx]:.3f}",
             xy=(depths[best_idx], bpbs[best_idx]),
             xytext=(depths[best_idx]+1.2, bpbs[best_idx]-0.02),
             fontsize=9, color=GREEN, fontweight="bold", fontfamily="monospace",
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5))

for d, b, s in zip(depths, bpbs, steps):
    ax1.annotate(f"{s} steps", xy=(d, b), xytext=(0, 10),
                 textcoords="offset points", fontsize=7, color=TEXT_DIM,
                 ha="center", fontfamily="monospace")

ax1.set_xlabel("Depth", fontsize=10, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("val_bpb", fontsize=10, color=TEXT, fontfamily="monospace")
ax1.set_title("MPS Depth Sweep", fontsize=13, color=BLUE, fontfamily="monospace", pad=8)
ax1.set_xticks(depths)
ax1.tick_params(colors=TEXT_DIM, labelsize=9)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)

# ═══════════════════════════════════════
# Row 1 Center: MPS Batch Sweep
# ═══════════════════════════════════════
ax2 = fig.add_axes([0.36, 0.66, 0.28, 0.25])
ax2.set_facecolor(CARD)
for spine in ax2.spines.values():
    spine.set_color("#30363d")

valid_batch = [(b, t, s, v, n) for b, t, s, v, n in mps_batch if v is not None]
x_labels = [f"B{b}\n{t}" for b, t, s, v, n in valid_batch]
batch_bpbs = [v for _, _, _, v, _ in valid_batch]
batch_steps = [s for _, _, s, _, _ in valid_batch]

colors_batch = [GREEN if n == "BEST" else BLUE for _, _, _, _, n in valid_batch]
bars = ax2.bar(range(len(valid_batch)), batch_bpbs, color=colors_batch, alpha=0.8,
               width=0.65, edgecolor="#30363d")

for i, (b, t, s, v, n) in enumerate(valid_batch):
    ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold",
             color=TEXT, fontfamily="monospace")
    ax2.text(i, v - 0.015, f"{s}st", ha="center", fontsize=7,
             color=TEXT_DIM, fontfamily="monospace")

# OOM bar
ax2.bar(len(valid_batch), 0.05, bottom=1.85, color=RED, alpha=0.5, width=0.65)
ax2.text(len(valid_batch), 1.88, "OOM\nB256", ha="center", fontsize=8,
         color=RED, fontfamily="monospace", fontweight="bold")

ax2.set_xticks(range(len(valid_batch)))
ax2.set_xticklabels(x_labels, fontsize=7, color=TEXT_DIM, fontfamily="monospace")
ax2.set_ylabel("val_bpb", fontsize=10, color=TEXT, fontfamily="monospace")
ax2.set_title("MPS Batch Sweep (Depth=4)", fontsize=13, color=BLUE, fontfamily="monospace", pad=8)
ax2.tick_params(colors=TEXT_DIM, labelsize=9)
ax2.grid(True, alpha=0.15, color=TEXT_DIM, axis="y")

# ═══════════════════════════════════════
# Row 1 Right: H100 vs MPS vs ANE summary
# ═══════════════════════════════════════
ax3 = fig.add_axes([0.68, 0.66, 0.28, 0.25])
ax3.set_facecolor(CARD)
for spine in ax3.spines.values():
    spine.set_color(GOLD)
    spine.set_linewidth(1.5)
    spine.set_alpha(0.5)
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_xticks([])
ax3.set_yticks([])

ax3.text(5, 9.3, "3-WAY COMPARISON", fontsize=14, fontweight="bold",
         color=GOLD, ha="center", fontfamily="monospace")
ax3.plot([0.5, 9.5], [8.7, 8.7], color="#30363d", linewidth=1)

rows = [
    ("",            "H100",    "MPS",     "ANE"),
    ("val_bpb",     "0.998",   "1.309",   "TBD"),
    ("Steps/5min",  "953",     "393",     "~3,900*"),
    ("Tokens/run",  "499.6M",  "25.8M",   "~10M*"),
    ("Params",      "50.3M",   "11.5M",   "53.5M*"),
    ("Depth",       "8",       "4",       "4*"),
    ("SEQ",         "2048",    "2048",    "512*"),
    ("Batch",       "128",     "32",      "1×10"),
]

header_colors = [TEXT_DIM, RED, BLUE, GREEN]
for i, row in enumerate(rows):
    y = 8.0 - i * 0.85
    for j, (val, col) in enumerate(zip(row, header_colors)):
        x = 0.5 + j * 2.4
        weight = "bold" if i == 0 else "normal"
        color = col if i == 0 else (TEXT if j == 0 else col)
        ax3.text(x, y, val, fontsize=10 if i == 0 else 9, color=color,
                 fontfamily="monospace", fontweight=weight)

ax3.text(5, 0.4, "* ANE estimated (NL=4, SEQ=512, 162ms/step)",
         fontsize=8, color=TEXT_DIM, ha="center", fontfamily="monospace")

# ═══════════════════════════════════════
# Row 2 Left: ANE Heatmap
# ═══════════════════════════════════════
ax4 = fig.add_axes([0.04, 0.34, 0.28, 0.25])
ax4.set_facecolor(CARD)

NLs = [2, 4, 8, 16, 24]
SEQs = [256, 512, 1024]
matrix = np.array([[ane_data[(nl, s)] for s in SEQs] for nl in NLs])
im = ax4.imshow(matrix, cmap="RdYlGn_r", aspect="auto", interpolation="nearest")

for i, nl in enumerate(NLs):
    for j, s in enumerate(SEQs):
        ms = ane_data[(nl, s)]
        color = "white" if ms > 500 else TEXT
        ax4.text(j, i, f"{ms:.0f}", ha="center", va="center",
                 fontsize=10, fontweight="bold", color=color, fontfamily="monospace")

# Highlight recommended
rect = plt.Rectangle((0.55, 0.55), 0.9, 0.9, linewidth=3, edgecolor=GREEN,
                      facecolor="none", linestyle="-")
ax4.add_patch(rect)

ax4.set_xticks(range(len(SEQs)))
ax4.set_xticklabels([f"S={s}" for s in SEQs], fontsize=9, color=TEXT, fontfamily="monospace")
ax4.set_yticks(range(len(NLs)))
ax4.set_yticklabels([f"NL={nl}" for nl in NLs], fontsize=9, color=TEXT, fontfamily="monospace")
ax4.set_title("ANE ms/step (all 15 configs)", fontsize=13, color=GREEN,
              fontfamily="monospace", pad=8)

# ═══════════════════════════════════════
# Row 2 Center: Tokens throughput comparison
# ═══════════════════════════════════════
ax5 = fig.add_axes([0.36, 0.34, 0.28, 0.25])
ax5.set_facecolor(CARD)
for spine in ax5.spines.values():
    spine.set_color("#30363d")

# Tokens in 5 min for each system
systems = ["H100\n(D8 B128)", "MPS\n(D4 B32)", "ANE NL4\n(S256)", "ANE NL4\n(S512)", "ANE NL4\n(S1024)"]
tokens_5min = [
    499.6,  # H100
    25.8,   # MPS
    (300000/76.7) * 256 * 10 / 1e6,   # ANE NL4 S256
    (300000/162.4) * 512 * 10 / 1e6,  # ANE NL4 S512
    (300000/406.6) * 1024 * 10 / 1e6, # ANE NL4 S1024
]
sys_colors = [RED, BLUE, GREEN, GREEN, GREEN]

bars = ax5.bar(range(len(systems)), tokens_5min, color=sys_colors, alpha=0.8,
               width=0.6, edgecolor="#30363d")
for i, (s, t) in enumerate(zip(systems, tokens_5min)):
    ax5.text(i, t + 5, f"{t:.1f}M", ha="center", fontsize=9, fontweight="bold",
             color=TEXT, fontfamily="monospace")

ax5.set_xticks(range(len(systems)))
ax5.set_xticklabels(systems, fontsize=8, color=TEXT_DIM, fontfamily="monospace")
ax5.set_ylabel("Tokens in 5 min (M)", fontsize=10, color=TEXT, fontfamily="monospace")
ax5.set_title("Data Throughput (5-min budget)", fontsize=13, color=TEXT,
              fontfamily="monospace", pad=8)
ax5.tick_params(colors=TEXT_DIM, labelsize=9)
ax5.grid(True, alpha=0.15, color=TEXT_DIM, axis="y")

# ═══════════════════════════════════════
# Row 2 Right: Steps/min comparison
# ═══════════════════════════════════════
ax6 = fig.add_axes([0.68, 0.34, 0.28, 0.25])
ax6.set_facecolor(CARD)
for spine in ax6.spines.values():
    spine.set_color("#30363d")

sys2 = ["H100", "MPS\n(best)", "ANE\nNL=2 S=256", "ANE\nNL=4 S=256", "ANE\nNL=4 S=512", "ANE\nNL=8 S=256"]
steps_min = [
    953/5,       # H100
    393/5,       # MPS
    60000/46,    # ANE NL2 S256
    60000/76.7,  # ANE NL4 S256
    60000/162.4, # ANE NL4 S512
    60000/123.7, # ANE NL8 S256
]
sys2_colors = [RED, BLUE, GREEN, GREEN, GREEN, GREEN]

bars = ax6.barh(range(len(sys2)), steps_min, color=sys2_colors, alpha=0.8,
                height=0.55, edgecolor="#30363d")
for i, (s, v) in enumerate(zip(sys2, steps_min)):
    ax6.text(v + 10, i, f"{v:.0f}", va="center", fontsize=9, fontweight="bold",
             color=TEXT, fontfamily="monospace")

ax6.set_yticks(range(len(sys2)))
ax6.set_yticklabels(sys2, fontsize=8, color=TEXT_DIM, fontfamily="monospace")
ax6.set_xlabel("Steps / minute", fontsize=10, color=TEXT, fontfamily="monospace")
ax6.set_title("Training Speed", fontsize=13, color=TEXT, fontfamily="monospace", pad=8)
ax6.tick_params(colors=TEXT_DIM, labelsize=9)
ax6.grid(True, alpha=0.15, color=TEXT_DIM, axis="x")
ax6.invert_yaxis()

# ═══════════════════════════════════════
# Row 3: Key insights
# ═══════════════════════════════════════
ax7 = fig.add_axes([0.04, 0.03, 0.92, 0.24])
ax7.set_facecolor(CARD)
for spine in ax7.spines.values():
    spine.set_color(GOLD)
    spine.set_linewidth(1.5)
    spine.set_alpha(0.4)
ax7.set_xlim(0, 10)
ax7.set_ylim(0, 10)
ax7.set_xticks([])
ax7.set_yticks([])

ax7.text(5, 9.3, "KEY INSIGHTS FROM PRE-TESTING", fontsize=16, fontweight="bold",
         color=GOLD, ha="center", fontfamily="monospace")
ax7.plot([0.3, 9.7], [8.6, 8.6], color="#30363d", linewidth=1)

col1 = [
    ("MPS FINDINGS", BLUE),
    ("Depth 4 wins (1.309 val_bpb)", TEXT),
    ("Batch 32 optimal (not 128)", TEXT),
    ("More steps > bigger batches on Metal", TEXT),
    ("Batch 256 OOMs at 141GB", RED),
    ("GPU defaults are 20% worse on Mac", ORANGE),
]

col2 = [
    ("ANE FINDINGS", GREEN),
    ("All 15 configs compile & run", TEXT),
    ("SRAM wall at SEQ=1024", ORANGE),
    ("49ms - 1344ms/step range", TEXT),
    ("6-8% ANE utilization (lots of headroom)", TEXT),
    ("Depth scales linearly", TEXT),
]

col3 = [
    ("NEXT STEPS", GOLD),
    ("Generate real training data", TEXT),
    ("Run ANE 5-min sweep (5 configs)", TEXT),
    ("Pick compute-optimal ANE config", TEXT),
    ("Compare MPS 1.309 vs ANE val_bpb", TEXT),
    ("Overnight run: MPS + ANE in parallel!", CYAN),
]

for col_idx, col_data in enumerate([col1, col2, col3]):
    x_base = 0.3 + col_idx * 3.3
    for i, (text, color) in enumerate(col_data):
        y = 7.8 - i * 1.2
        weight = "bold" if i == 0 else "normal"
        size = 11 if i == 0 else 9.5
        prefix = "" if i == 0 else "→ "
        ax7.text(x_base, y, f"{prefix}{text}", fontsize=size, color=color,
                 fontfamily="monospace", fontweight=weight)

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=11, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch · M4 Max · MPS + ANE training benchmark", fontsize=8,
         color=TEXT_DIM, alpha=0.35, ha="left", va="bottom", fontfamily="monospace")

plt.savefig("/Users/dan/Dev/autoresearch-ANE/viz/mps_vs_ane_comparison.png",
            dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to viz/mps_vs_ane_comparison.png")
