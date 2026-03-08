"""Chart the depth sweep results + explain why they're misleading."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Results so far ---
results = {
    2:  {"val_bpb": 1.3908, "steps": 1002, "params_M": 3.5,  "tokens_M": 25.1},
    4:  {"val_bpb": 1.3122, "steps": 368,  "params_M": 11.5, "tokens_M": 25.1},
    6:  {"val_bpb": 1.4182, "steps": 198,  "params_M": 26.3, "tokens_M": 25.1},
    8:  {"val_bpb": 1.5756, "steps": 107,  "params_M": 50.3, "tokens_M": 7.0},
}

# --- Colors ---
BG = "#0d1117"
CARD = "#161b22"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
ACCENT = "#3fb950"
ACCENT2 = "#58a6ff"
HIGHLIGHT = "#f0883e"
WARNING = "#f85149"

depths = sorted(results.keys())
bpbs = [results[d]["val_bpb"] for d in depths]
steps = [results[d]["steps"] for d in depths]
params = [results[d]["params_M"] for d in depths]

fig = plt.figure(figsize=(18, 11), facecolor=BG)

# --- Title ---
fig.text(0.5, 0.96, "Depth Sweep on M4 Max (128GB)", fontsize=26, fontweight="bold",
         color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.925, "...but these results are misleading. Here's why.",
         fontsize=14, color=WARNING, ha="center", va="top", fontfamily="monospace")

# --- Left chart: the U-curve ---
ax1 = fig.add_axes([0.06, 0.42, 0.40, 0.45])
ax1.set_facecolor(CARD)
for spine in ax1.spines.values():
    spine.set_color("#30363d")

ax1.plot(depths, bpbs, 'o-', color=ACCENT, markersize=14, linewidth=2.5,
         markeredgecolor="white", markeredgewidth=1.5, zorder=5)

best_idx = np.argmin(bpbs)
best_d = depths[best_idx]
best_b = bpbs[best_idx]
ax1.scatter([best_d], [best_b], s=220, color=HIGHLIGHT, zorder=6,
            edgecolors="white", linewidths=2)
ax1.annotate(f"Current best: depth {best_d}\nval_bpb = {best_b:.4f}",
             xy=(best_d, best_b), xytext=(best_d + 1.5, best_b - 0.02),
             fontsize=11, color=HIGHLIGHT, fontweight="bold", fontfamily="monospace",
             arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, lw=2))

for d, b, p in zip(depths, bpbs, params):
    if d != best_d:
        ax1.annotate(f"{p:.0f}M", xy=(d, b), xytext=(0, 12),
                     textcoords="offset points", fontsize=9, color=TEXT_DIM,
                     ha="center", fontfamily="monospace")

ax1.set_xlabel("Depth (layers)", fontsize=13, color=TEXT, fontfamily="monospace")
ax1.set_ylabel("val_bpb (lower = better)", fontsize=13, color=TEXT, fontfamily="monospace")
ax1.set_title("Model Depth vs Performance", fontsize=15, color=TEXT,
              fontfamily="monospace", pad=12)
ax1.set_xticks(depths)
ax1.tick_params(colors=TEXT_DIM)
ax1.grid(True, alpha=0.15, color=TEXT_DIM)

# --- Right chart: steps tradeoff ---
ax2 = fig.add_axes([0.54, 0.42, 0.40, 0.45])
ax2.set_facecolor(CARD)
for spine in ax2.spines.values():
    spine.set_color("#30363d")

colors = [HIGHLIGHT if d == best_d else ACCENT2 for d in depths]
bars = ax2.bar(depths, steps, color=colors, alpha=0.85, width=1.2, edgecolor="#30363d")
for d, s in zip(depths, steps):
    ax2.text(d, s + 15, f"{s}", ha="center", fontsize=12, fontweight="bold",
             color=TEXT, fontfamily="monospace")

ax2.set_xlabel("Depth (layers)", fontsize=13, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("Training Steps in 5 min", fontsize=13, color=TEXT, fontfamily="monospace")
ax2.set_title("Bigger Model = Fewer Steps", fontsize=15, color=TEXT,
              fontfamily="monospace", pad=12)
ax2.set_xticks(depths)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM, axis="y")

# --- Bottom: the "why this is wrong" section ---
problems = [
    ("BATCH SIZE = 16",
     "We have 128GB RAM but the fork\n"
     "was tuned for 8-16GB Macs.\n"
     "We're using <1% of our memory.\n"
     "Bigger batches = more steps/sec."),
    ("NO AUTOCAST (bf16)",
     "Running float32 math when bf16\n"
     "could work on MPS. That's 2x\n"
     "the memory bandwidth wasted.\n"
     "Every matmul is half speed."),
    ("BATCH TOKENS = 65K",
     "GPU uses 524K tokens/batch.\n"
     "We're at 65K (8x smaller).\n"
     "With 128GB we could go much\n"
     "bigger for better gradients."),
    ("WINDOW = 'L' (all long)",
     "GPU used 'SSSL' (mostly short\n"
     "sliding windows). Short windows\n"
     "= cheaper attention = faster\n"
     "steps. Never tested on Mac."),
]

box_width = 0.20
for i, (title, body) in enumerate(problems):
    x = 0.04 + i * 0.245
    ax = fig.add_axes([x, 0.04, 0.22, 0.28])
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(WARNING)
        spine.set_linewidth(1.5)
        spine.set_alpha(0.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(5, 8.5, title, fontsize=10.5, fontweight="bold", color=WARNING,
            ha="center", va="center", fontfamily="monospace")
    ax.plot([1, 9], [7.5, 7.5], color="#30363d", linewidth=1)
    ax.text(5, 4.0, body, fontsize=9, color=TEXT_DIM,
            ha="center", va="center", fontfamily="monospace", linespacing=1.6)

# Label for bottom section
fig.text(0.5, 0.335, "WHY THESE RESULTS ARE WRONG — the macOS fork wasn't optimized for M4 Max (128GB)",
         fontsize=12, fontweight="bold", color=WARNING, ha="center", va="center",
         fontfamily="monospace")

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=11, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "based on karpathy/autoresearch", fontsize=8, color=TEXT_DIM, alpha=0.35,
         ha="left", va="bottom", fontfamily="monospace")

plt.savefig("depth_sweep.png", dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to depth_sweep.png")
