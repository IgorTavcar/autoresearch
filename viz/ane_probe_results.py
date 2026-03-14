"""ANE Hardware Probe Results — M4 Max (128GB)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Probe results ---
seq_probe = {
    256:  {"ms_step": 91.4, "status": "OK",   "flops_B": 87.0},
    512:  {"ms_step": 171.7, "status": "OK",  "flops_B": 174.0},
    1024: {"ms_step": 389.0, "status": "OK",  "flops_B": 347.9},
    1152: {"ms_step": None,  "status": "FAIL", "flops_B": None},
    1536: {"ms_step": None,  "status": "FAIL", "flops_B": None},
    2048: {"ms_step": None,  "status": "FAIL", "flops_B": None},
}

depth_probe = {
    4:  {"ms_step": 49.2,  "params_M": 53.5,  "flops_B": 43.5},
    8:  {"ms_step": 91.4,  "params_M": 81.8,  "flops_B": 87.0},
    16: {"ms_step": 146.5, "params_M": 138.4, "flops_B": 174.0},
}

# Timing breakdown (NL=8, SEQ=256 baseline, steady-state avg)
timing = {
    "ANE fwd":    10.4,
    "ANE bwd":    15.4,
    "IO fwd":      3.9,
    "IO bwd":     12.3,
    "classifier": 16.7,
    "dw_copy":     7.3,
    "SiLU":        5.8,
    "RMSNorm":     4.3,
    "RMS bwd":     2.8,
}

# --- Colors ---
BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
ORANGE = "#f0883e"
RED = "#f85149"
PURPLE = "#bc8cff"

fig = plt.figure(figsize=(20, 13), facecolor=BG)

# --- Title ---
fig.text(0.5, 0.97, "Apple Neural Engine — Training Hardware Profile", fontsize=26,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.945, "M4 Max · 16 ANE cores · 128GB unified memory · monolithic SDPA kernels",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

# ═══════════════════════════════════════
# Top-left: Sequence Length Probe
# ═══════════════════════════════════════
ax1 = fig.add_axes([0.06, 0.52, 0.40, 0.38])
ax1.set_facecolor(CARD)
for spine in ax1.spines.values():
    spine.set_color("#30363d")

seqs = sorted(seq_probe.keys())
ok_seqs = [s for s in seqs if seq_probe[s]["status"] == "OK"]
fail_seqs = [s for s in seqs if seq_probe[s]["status"] == "FAIL"]
ok_ms = [seq_probe[s]["ms_step"] for s in ok_seqs]

ax1.bar(range(len(ok_seqs)), ok_ms, color=GREEN, alpha=0.85, width=0.6, edgecolor="#30363d")
for i, (s, ms) in enumerate(zip(ok_seqs, ok_ms)):
    ax1.text(i, ms + 8, f"{ms:.0f}ms", ha="center", fontsize=11, fontweight="bold",
             color=TEXT, fontfamily="monospace")
    ax1.text(i, -18, f"SEQ={s}", ha="center", fontsize=10, color=TEXT_DIM, fontfamily="monospace")

fail_start = len(ok_seqs)
for i, s in enumerate(fail_seqs):
    idx = fail_start + i
    ax1.bar(idx, 50, color=RED, alpha=0.3, width=0.6, edgecolor=RED, linewidth=1.5, linestyle="--")
    ax1.text(idx, 55, "SRAM\nFAIL", ha="center", fontsize=9, fontweight="bold",
             color=RED, fontfamily="monospace")
    ax1.text(idx, -18, f"SEQ={s}", ha="center", fontsize=10, color=TEXT_DIM, fontfamily="monospace")

# Wall marker
ax1.axvline(x=len(ok_seqs) - 0.5, color=RED, linewidth=2, linestyle="--", alpha=0.7)
ax1.text(len(ok_seqs) - 0.3, ax1.get_ylim()[1] * 0.85, "SRAM WALL",
         fontsize=11, fontweight="bold", color=RED, fontfamily="monospace", rotation=90, va="top")

ax1.set_xticks([])
ax1.set_ylabel("ms / step", fontsize=12, color=TEXT, fontfamily="monospace")
ax1.set_title("Sequence Length Probe", fontsize=15, color=TEXT, fontfamily="monospace", pad=12)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.15, color=TEXT_DIM, axis="y")
ax1.set_xlim(-0.5, len(seqs) - 0.5)

# ═══════════════════════════════════════
# Top-right: Depth Scaling
# ═══════════════════════════════════════
ax2 = fig.add_axes([0.54, 0.52, 0.40, 0.38])
ax2.set_facecolor(CARD)
for spine in ax2.spines.values():
    spine.set_color("#30363d")

nl_depths = sorted(depth_probe.keys())
nl_ms = [depth_probe[d]["ms_step"] for d in nl_depths]
nl_params = [depth_probe[d]["params_M"] for d in nl_depths]

ax2.plot(nl_depths, nl_ms, 'o-', color=BLUE, markersize=14, linewidth=2.5,
         markeredgecolor="white", markeredgewidth=1.5, zorder=5)

for d, ms, p in zip(nl_depths, nl_ms, nl_params):
    ax2.annotate(f"{ms:.0f}ms\n{p:.0f}M params", xy=(d, ms), xytext=(0, 16),
                 textcoords="offset points", fontsize=10, color=TEXT,
                 ha="center", fontfamily="monospace", fontweight="bold")

# Show linear scaling line
x_fit = np.array(nl_depths)
slope = (nl_ms[-1] - nl_ms[0]) / (nl_depths[-1] - nl_depths[0])
y_fit = nl_ms[0] + slope * (x_fit - nl_depths[0])
ax2.plot(x_fit, y_fit, '--', color=TEXT_DIM, alpha=0.5, linewidth=1.5, label="Linear scaling")

ax2.set_xlabel("NLAYERS", fontsize=12, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("ms / step", fontsize=12, color=TEXT, fontfamily="monospace")
ax2.set_title("Depth Scaling (SEQ=256)", fontsize=15, color=TEXT, fontfamily="monospace", pad=12)
ax2.set_xticks(nl_depths)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM)
ax2.legend(loc="upper left", fontsize=10, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT_DIM)

# ═══════════════════════════════════════
# Bottom-left: Timing Breakdown (pie-ish)
# ═══════════════════════════════════════
ax3 = fig.add_axes([0.06, 0.06, 0.40, 0.38])
ax3.set_facecolor(CARD)
for spine in ax3.spines.values():
    spine.set_color("#30363d")

labels = list(timing.keys())
values = list(timing.values())
total = sum(values)
sorted_idx = np.argsort(values)[::-1]
labels = [labels[i] for i in sorted_idx]
values = [values[i] for i in sorted_idx]

colors_bar = []
for l in labels:
    if "ANE" in l:
        colors_bar.append(GREEN)
    elif "IO" in l or "copy" in l:
        colors_bar.append(ORANGE)
    elif "class" in l:
        colors_bar.append(RED)
    else:
        colors_bar.append(PURPLE)

bars = ax3.barh(range(len(labels)), values, color=colors_bar, alpha=0.85,
                height=0.65, edgecolor="#30363d")
for i, (l, v) in enumerate(zip(labels, values)):
    pct = 100 * v / total
    ax3.text(v + 0.3, i, f"{v:.1f}ms ({pct:.0f}%)", va="center", fontsize=10,
             color=TEXT, fontfamily="monospace")

ax3.set_yticks(range(len(labels)))
ax3.set_yticklabels(labels, fontsize=11, fontfamily="monospace")
ax3.invert_yaxis()
ax3.set_xlabel("Time (ms)", fontsize=12, color=TEXT, fontfamily="monospace")
ax3.set_title("Per-Step Timing Breakdown (NL=8, SEQ=256)", fontsize=15,
              color=TEXT, fontfamily="monospace", pad=12)
ax3.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM, axis="x")

# Legend for colors
legend_items = [
    ("ANE compute", GREEN),
    ("IO / copies", ORANGE),
    ("CPU compute", RED),
    ("CPU math", PURPLE),
]
for i, (label, color) in enumerate(legend_items):
    ax3.text(total * 0.65, len(labels) - 1.5 + i * 0.45, f"● {label}",
             fontsize=9, color=color, fontfamily="monospace", fontweight="bold")

# ═══════════════════════════════════════
# Bottom-right: Summary stats
# ═══════════════════════════════════════
ax4 = fig.add_axes([0.54, 0.06, 0.40, 0.38])
ax4.set_facecolor(CARD)
for spine in ax4.spines.values():
    spine.set_color(GREEN)
    spine.set_linewidth(1.5)
    spine.set_alpha(0.5)
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_xticks([])
ax4.set_yticks([])

ax4.text(5, 9.2, "KEY FINDINGS", fontsize=16, fontweight="bold", color=GREEN,
         ha="center", fontfamily="monospace")
ax4.plot([1, 9], [8.5, 8.5], color="#30363d", linewidth=1)

findings = [
    (f"Max SEQ (monolithic):  1024", TEXT),
    (f"SRAM wall at:          SEQ > 1024", RED),
    (f"Tiled attention for:   SEQ = 2048", ORANGE),
    ("", TEXT),
    (f"Fastest config:        NL=4 @ 49ms/step", GREEN),
    (f"Baseline config:       NL=8 @ 91ms/step", BLUE),
    (f"Deep config:           NL=16 @ 147ms/step", BLUE),
    ("", TEXT),
    (f"ANE compute:           33% of step time", GREEN),
    (f"IO overhead:           30% of step time", ORANGE),
    (f"CPU bottleneck:        37% of step time", RED),
    ("", TEXT),
    (f"Depth scales:          ~linearly (good!)", GREEN),
    (f"10K steps @ NL=8:      ~15 minutes", GREEN),
]

for i, (text, color) in enumerate(findings):
    ax4.text(0.5, 7.8 - i * 0.55, text, fontsize=10, color=color,
             fontfamily="monospace", va="center")

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=11, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch-ANE · Apple Neural Engine training", fontsize=8,
         color=TEXT_DIM, alpha=0.35, ha="left", va="bottom", fontfamily="monospace")

plt.savefig("/Users/dan/Dev/autoresearch-ANE/viz/ane_probe_results.png",
            dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to viz/ane_probe_results.png")
