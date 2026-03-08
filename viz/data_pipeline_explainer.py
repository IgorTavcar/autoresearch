"""Why ANE Results Don't Compare — Data Pipeline Explainer"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

fig = plt.figure(figsize=(22, 12), facecolor=BG)

fig.text(0.5, 0.96, "Why ANE Results Don't Directly Compare", fontsize=26,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.925, "Three accelerators, two completely different data pipelines",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

gs = gridspec.GridSpec(2, 2, left=0.06, right=0.97, top=0.88, bottom=0.08,
                       hspace=0.4, wspace=0.3)

# ═══════════════════════════════════════
# Panel 1: MLX/MPS pipeline (top left)
# ═══════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0], facecolor=CARD)
for spine in ax1.spines.values(): spine.set_color("#30363d")
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 10)
ax1.set_xticks([])
ax1.set_yticks([])

ax1.text(5, 9.3, "MLX / MPS Pipeline", fontsize=16, fontweight="bold",
         color=GREEN, ha="center", fontfamily="monospace")
ax1.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

# Pipeline boxes
boxes = [
    (1, 7.5, "Karpathy climbmix-400B", BLUE, "400B tokens, curated web text"),
    (1, 6.0, "rustbpe tokenizer", GREEN, "vocab = 8,192 (MLX) or tiktoken 32K (MPS)"),
    (1, 4.5, "BOS-aligned packing", GREEN, "SEQ = 2,048, best-fit packing"),
    (1, 3.0, "5-min training", GREEN, "~375 steps (MLX) or ~400 steps (MPS)"),
    (1, 1.5, "evaluate_bpb()", GREEN, "bits per byte, vocab-independent"),
]

for x, y, label, color, desc in boxes:
    ax1.add_patch(mpatches.FancyBboxPatch((x, y - 0.45), 8, 0.9,
        boxstyle="round,pad=0.15", facecolor=color, alpha=0.08,
        edgecolor=color, linewidth=1.5))
    ax1.text(x + 0.3, y + 0.05, label, fontsize=10, fontweight="bold",
             color=color, fontfamily="monospace", va="center")
    ax1.text(x + 0.3, y - 0.3, desc, fontsize=7.5, color=TEXT_DIM,
             fontfamily="monospace", va="center")

# Arrows
for y_start, y_end in [(7.05, 6.45), (5.55, 4.95), (4.05, 3.45), (2.55, 1.95)]:
    ax1.annotate("", xy=(5, y_end), xytext=(5, y_start),
                 arrowprops=dict(arrowstyle="->", color=TEXT_DIM, lw=1.5))

# ═══════════════════════════════════════
# Panel 2: ANE pipeline (top right)
# ═══════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1], facecolor=CARD)
for spine in ax2.spines.values(): spine.set_color("#30363d")
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 10)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.text(5, 9.3, "ANE Pipeline (native Obj-C)", fontsize=16, fontweight="bold",
         color=CYAN, ha="center", fontfamily="monospace")
ax2.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

boxes_ane = [
    (1, 7.5, "Custom binary data", RED, "pre-tokenized train.bin / val.bin"),
    (1, 6.0, "Custom tokenizer", RED, "different vocab size, different encoding"),
    (1, 4.5, "Simple sequential read", ORANGE, "SEQ = 512, no packing"),
    (1, 3.0, "Variable-length training", ORANGE, "10K-330K steps, 99ms/step"),
    (1, 1.5, "Training cross-entropy", RED, "raw loss, NOT val_bpb"),
]

for x, y, label, color, desc in boxes_ane:
    ax2.add_patch(mpatches.FancyBboxPatch((x, y - 0.45), 8, 0.9,
        boxstyle="round,pad=0.15", facecolor=color, alpha=0.08,
        edgecolor=color, linewidth=1.5))
    ax2.text(x + 0.3, y + 0.05, label, fontsize=10, fontweight="bold",
             color=color, fontfamily="monospace", va="center")
    ax2.text(x + 0.3, y - 0.3, desc, fontsize=7.5, color=TEXT_DIM,
             fontfamily="monospace", va="center")

for y_start, y_end in [(7.05, 6.45), (5.55, 4.95), (4.05, 3.45), (2.55, 1.95)]:
    ax2.annotate("", xy=(5, y_end), xytext=(5, y_start),
                 arrowprops=dict(arrowstyle="->", color=TEXT_DIM, lw=1.5))

# ═══════════════════════════════════════
# Panel 3: What's different (bottom left)
# ═══════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0], facecolor=CARD)
for spine in ax3.spines.values(): spine.set_color("#30363d")
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 10)
ax3.set_xticks([])
ax3.set_yticks([])

ax3.text(5, 9.3, "What's Different", fontsize=16, fontweight="bold",
         color=RED, ha="center", fontfamily="monospace")
ax3.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

diffs = [
    ("Dataset", "climbmix-400B (curated)", "custom binary (unknown)", RED),
    ("Tokenizer", "rustbpe / tiktoken", "custom", RED),
    ("Vocab size", "8,192 / 32,768", "different", ORANGE),
    ("Seq length", "2,048", "512 (4x shorter)", RED),
    ("Batch size", "65K tokens (grad accum)", "small (SRAM limited)", ORANGE),
    ("Evaluation", "val_bpb (standardized)", "training loss (raw)", RED),
    ("Optimizer", "AdamW (4 LR groups)", "Basic Adam", ORANGE),
    ("Architecture", "VE, lambdas, softcap...", "Basic attn + FFN", ORANGE),
]

y = 8.2
ax3.text(0.5, y, "Component", fontsize=8, color="#30363d", fontfamily="monospace", fontweight="bold")
ax3.text(3.5, y, "MLX / MPS", fontsize=8, color=GREEN, fontfamily="monospace", fontweight="bold")
ax3.text(7.0, y, "ANE", fontsize=8, color=CYAN, fontfamily="monospace", fontweight="bold")
y -= 0.7

for component, mlx_val, ane_val, severity in diffs:
    ax3.text(0.5, y, component, fontsize=9, color=TEXT_DIM, fontfamily="monospace")
    ax3.text(3.5, y, mlx_val, fontsize=8, color=TEXT, fontfamily="monospace")
    ax3.text(7.0, y, ane_val, fontsize=8, color=severity, fontfamily="monospace")
    y -= 0.85

# ═══════════════════════════════════════
# Panel 4: What ANE results DO tell us (bottom right)
# ═══════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1], facecolor=CARD)
for spine in ax4.spines.values(): spine.set_color("#30363d")
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)
ax4.set_xticks([])
ax4.set_yticks([])

ax4.text(5, 9.3, "What ANE Results DO Tell Us", fontsize=16, fontweight="bold",
         color=GREEN, ha="center", fontfamily="monospace")
ax4.plot([0.5, 9.5], [8.8, 8.8], color="#30363d", linewidth=1)

findings = [
    ("The Neural Engine can train LLMs", "Nobody else has demonstrated this", GREEN),
    ("99ms/step — 8x faster than GPU", "67.6M params vs 11.5M on MPS", CYAN),
    ("ANE + GPU = zero interference", "Both run simultaneously, free compute", GREEN),
    ("SRAM wall at SEQ=1024", "Hardware constraint discovery", GOLD),
    ("Depth U-curve: NL=6 optimal", "Architecture search within ANE", CYAN),
    ("Cosine schedule must match run", "Stability finding (transfers to any hw)", GREEN),
    ("Activation explosion patterns", "Early warning signals identified", ORANGE),
    ("Dynamic weight pipeline works", "memcpy updates, no recompile", CYAN),
]

y = 8.0
for title, desc, color in findings:
    ax4.text(0.5, y + 0.1, title, fontsize=10, fontweight="bold",
             color=color, fontfamily="monospace")
    ax4.text(0.5, y - 0.3, desc, fontsize=8, color=TEXT_DIM, fontfamily="monospace")
    y -= 0.95

# Center watermark
fig.text(0.5, 0.50, "@danpacary", fontsize=60, color=TEXT_DIM, alpha=0.04,
         ha="center", va="center", fontfamily="monospace", fontweight="bold",
         rotation=25, zorder=0)

# Bottom
fig.text(0.97, 0.005, "@danpacary", fontsize=10, color=TEXT_DIM, alpha=0.4,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "To make ANE comparable: bridge Karpathy data -> binary, match tokenizer + eval",
         fontsize=8, color=TEXT_DIM, alpha=0.4, ha="left", va="bottom", fontfamily="monospace")

out = "/Users/dan/Dev/autoresearch-ANE/viz/data_pipeline_explainer.png"
plt.savefig(out, dpi=180, facecolor=BG, edgecolor="none", pad_inches=0.3)
print(f"Saved to {out}")
