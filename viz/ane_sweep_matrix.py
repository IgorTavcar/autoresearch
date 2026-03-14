"""ANE Full Sweep Matrix — M4 Max (128GB)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Full 5×3 sweep results
data = {
    (2, 256): 46.0, (2, 512): 105.6, (2, 1024): 283.8,
    (4, 256): 76.7, (4, 512): 162.4, (4, 1024): 406.6,
    (8, 256): 123.7, (8, 512): 269.0, (8, 1024): 592.1,
    (16, 256): 251.4, (16, 512): 521.1, (16, 1024): 971.6,
    (24, 256): 321.1, (24, 512): 637.9, (24, 1024): 1343.6,
}

params = {2: 39.3, 4: 53.5, 8: 81.8, 16: 138.4, 24: 195.1}
flops = {
    (2, 256): 21.7, (2, 512): 43.5, (2, 1024): 87.0,
    (4, 256): 43.5, (4, 512): 87.0, (4, 1024): 173.9,
    (8, 256): 87.0, (8, 512): 173.9, (8, 1024): 347.9,
    (16, 256): 173.9, (16, 512): 347.9, (16, 1024): 695.8,
    (24, 256): 260.9, (24, 512): 521.8, (24, 1024): 1043.7,
}

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

NLs = [2, 4, 8, 16, 24]
SEQs = [256, 512, 1024]

fig = plt.figure(figsize=(22, 14), facecolor=BG)

# Title
fig.text(0.5, 0.97, "ANE Training Sweep — Full Hardware Profile", fontsize=26,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.945, "M4 Max · 16 ANE cores · DIM=768 · HEADS=6 · HIDDEN=2048 · All configs compile & run",
         fontsize=12, color=GREEN, ha="center", va="top", fontfamily="monospace")

# ═══════════════════════════════════════
# Top-left: Heatmap of ms/step
# ═══════════════════════════════════════
ax1 = fig.add_axes([0.06, 0.52, 0.42, 0.38])
ax1.set_facecolor(CARD)

matrix = np.array([[data[(nl, s)] for s in SEQs] for nl in NLs])
im = ax1.imshow(matrix, cmap="RdYlGn_r", aspect="auto", interpolation="nearest")

for i, nl in enumerate(NLs):
    for j, s in enumerate(SEQs):
        ms = data[(nl, s)]
        color = "white" if ms > 500 else TEXT
        ax1.text(j, i, f"{ms:.0f}ms", ha="center", va="center",
                 fontsize=12, fontweight="bold", color=color, fontfamily="monospace")

ax1.set_xticks(range(len(SEQs)))
ax1.set_xticklabels([f"SEQ={s}" for s in SEQs], fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_yticks(range(len(NLs)))
ax1.set_yticklabels([f"NL={nl}" for nl in NLs], fontsize=11, color=TEXT, fontfamily="monospace")
ax1.set_title("ms / step (lower = faster)", fontsize=15, color=TEXT, fontfamily="monospace", pad=12)

cbar = plt.colorbar(im, ax=ax1, shrink=0.8, pad=0.02)
cbar.ax.tick_params(colors=TEXT_DIM)
cbar.set_label("ms/step", color=TEXT_DIM, fontfamily="monospace")

# ═══════════════════════════════════════
# Top-right: GFLOP/s efficiency
# ═══════════════════════════════════════
ax2 = fig.add_axes([0.56, 0.52, 0.40, 0.38])
ax2.set_facecolor(CARD)
for spine in ax2.spines.values():
    spine.set_color("#30363d")

# Compute GFLOP/s for each config
for s_idx, s in enumerate(SEQs):
    gflops_list = []
    for nl in NLs:
        ms = data[(nl, s)]
        gf = flops[(nl, s)] * 1000 / ms  # GFLOP/s
        gflops_list.append(gf)
    color = [GREEN, BLUE, ORANGE][s_idx]
    ax2.plot(NLs, gflops_list, 'o-', color=color, markersize=10, linewidth=2.5,
             markeredgecolor="white", markeredgewidth=1.5, label=f"SEQ={s}", zorder=5)

# ANE theoretical peak
ax2.axhline(y=10500, color=RED, linewidth=1.5, linestyle="--", alpha=0.5)
ax2.text(1.5, 10800, "ANE peak: 10.5 TFLOP/s", fontsize=9, color=RED,
         fontfamily="monospace", alpha=0.7)

ax2.set_xlabel("NLAYERS", fontsize=12, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("GFLOP/s (effective)", fontsize=12, color=TEXT, fontfamily="monospace")
ax2.set_title("Compute Efficiency", fontsize=15, color=TEXT, fontfamily="monospace", pad=12)
ax2.set_xticks(NLs)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM)
ax2.legend(loc="upper left", fontsize=10, facecolor=CARD, edgecolor="#30363d", labelcolor=TEXT)

# ═══════════════════════════════════════
# Bottom-left: tokens/sec for 5-min budget
# ═══════════════════════════════════════
ax3 = fig.add_axes([0.06, 0.06, 0.42, 0.38])
ax3.set_facecolor(CARD)
for spine in ax3.spines.values():
    spine.set_color("#30363d")

# In 5 minutes, how many tokens does each config process?
budget_s = 300  # 5 minutes
for s_idx, s in enumerate(SEQs):
    tokens_5min = []
    for nl in NLs:
        ms = data[(nl, s)]
        steps_5min = budget_s * 1000 / ms
        # effective tokens = steps * SEQ * accum_steps(10)
        total_tok = steps_5min * s * 10
        tokens_5min.append(total_tok / 1e6)  # millions
    color = [GREEN, BLUE, ORANGE][s_idx]
    ax3.plot(NLs, tokens_5min, 's-', color=color, markersize=10, linewidth=2.5,
             markeredgecolor="white", markeredgewidth=1.5, label=f"SEQ={s}", zorder=5)

ax3.set_xlabel("NLAYERS", fontsize=12, color=TEXT, fontfamily="monospace")
ax3.set_ylabel("Tokens processed (M) in 5 min", fontsize=12, color=TEXT, fontfamily="monospace")
ax3.set_title("Data Throughput (5-min training budget)", fontsize=15, color=TEXT,
              fontfamily="monospace", pad=12)
ax3.set_xticks(NLs)
ax3.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM)
ax3.legend(loc="upper right", fontsize=10, facecolor=CARD, edgecolor="#30363d", labelcolor=TEXT)

# ═══════════════════════════════════════
# Bottom-right: key insights
# ═══════════════════════════════════════
ax4 = fig.add_axes([0.56, 0.06, 0.40, 0.38])
ax4.set_facecolor(CARD)
for spine in ax4.spines.values():
    spine.set_color(GREEN)
    spine.set_linewidth(1.5)
    spine.set_alpha(0.5)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_xticks([])
ax4.set_yticks([])

ax4.text(5, 9.3, "HARDWARE PROFILE SUMMARY", fontsize=14, fontweight="bold",
         color=GREEN, ha="center", fontfamily="monospace")
ax4.plot([0.5, 9.5], [8.7, 8.7], color="#30363d", linewidth=1)

findings = [
    ("SRAM Wall", ""),
    (f"  Max monolithic SEQ:    1024", GREEN),
    (f"  Tiled attention for:   2048+", ORANGE),
    ("", ""),
    ("Config Space (all work!)", ""),
    (f"  Fastest:  NL=2 SEQ=256   46ms/step", GREEN),
    (f"  Biggest:  NL=24 SEQ=1024 1.3s/step", BLUE),
    (f"  195M params trained on ANE!", BLUE),
    ("", ""),
    ("Scaling Properties", ""),
    (f"  Depth:    ~linear (good)", GREEN),
    (f"  SEQ:      ~2.5x per doubling", ORANGE),
    (f"  GFLOP/s:  600-800 effective", BLUE),
    (f"  Peak ANE: 10,500 (6-8% util)", RED),
]

for i, (text, color) in enumerate(findings):
    if not color:
        color = TEXT
        text = f"\n{text}"
    ax4.text(0.5, 8.2 - i * 0.56, text, fontsize=10, color=color,
             fontfamily="monospace", va="center",
             fontweight="bold" if not text.startswith("  ") else "normal")

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=11, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch-ANE · Apple Neural Engine training benchmark", fontsize=8,
         color=TEXT_DIM, alpha=0.35, ha="left", va="bottom", fontfamily="monospace")

plt.savefig("/Users/dan/Dev/autoresearch-ANE/viz/ane_sweep_matrix.png",
            dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to viz/ane_sweep_matrix.png")
