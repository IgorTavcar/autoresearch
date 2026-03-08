"""Hero Summary — One Chip, Two Brains — M4 Max Dual-Accelerator Training"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import os

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

# Parse ANE loss curves from logs
log_dir = os.path.join(os.path.dirname(__file__), "..", "results", "sweep_5min")
ane_configs = {
    "B":  {"nl": 4,  "seq": 512,  "ms": 75.9,  "color": BLUE},
    "D":  {"nl": 8,  "seq": 256,  "ms": 63.9,  "color": CYAN},
    "E":  {"nl": 8,  "seq": 512,  "ms": 123.4, "color": GREEN},
    "F1": {"nl": 6,  "seq": 512,  "ms": 99.2,  "color": GOLD},
    "F2": {"nl": 10, "seq": 512,  "ms": 146.9, "color": PURPLE},
}
loss_curves = {}
for key, cfg in ane_configs.items():
    log_file = os.path.join(log_dir, f"{key}_nl{cfg['nl']}_s{cfg['seq']}.log")
    steps, losses = [], []
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                if line.startswith("step "):
                    parts = line.split()
                    try:
                        steps.append(int(parts[1]))
                        losses.append(float(parts[2].replace("loss=", "")))
                    except (ValueError, IndexError):
                        pass
    loss_curves[key] = (steps, losses)

fig = plt.figure(figsize=(24, 14), facecolor=BG)

# ═══════════════════════════════════════
# Title
# ═══════════════════════════════════════
fig.text(0.5, 0.97, "One Chip, Two Brains", fontsize=36,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.935, "Training GPT models on Apple Neural Engine + Metal GPU simultaneously",
         fontsize=14, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.912, "M4 Max · 128GB · macOS · Native Objective-C + Python",
         fontsize=11, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace", alpha=0.6)

# ═══════════════════════════════════════
# Left: The comparison (the headline)
# ═══════════════════════════════════════
ax_comp = fig.add_axes([0.03, 0.42, 0.44, 0.46])
ax_comp.set_facecolor(CARD)
for spine in ax_comp.spines.values():
    spine.set_color("#30363d")
ax_comp.set_xlim(0, 10)
ax_comp.set_ylim(0, 10)
ax_comp.set_xticks([])
ax_comp.set_yticks([])

# ANE side
ax_comp.text(2.5, 9.3, "NEURAL ENGINE", fontsize=16, fontweight="bold",
             color=GOLD, ha="center", fontfamily="monospace")
ax_comp.text(2.5, 8.65, "16 ANE cores", fontsize=10,
             color=TEXT_DIM, ha="center", fontfamily="monospace")
ax_comp.plot([0.3, 4.7], [8.3, 8.3], color="#30363d", linewidth=1)

ane_stats = [
    ("67.6M", "params", GOLD),
    ("99 ms", "per step", GOLD),
    ("6 layers", "DIM=768", TEXT_DIM),
    ("3,000 steps", "in 5 min", GREEN),
    ("327K steps", "in 9 hours", GREEN),
]
y = 7.7
for val, label, color in ane_stats:
    ax_comp.text(1.2, y, val, fontsize=15, fontweight="bold",
                 color=color, ha="right", fontfamily="monospace")
    ax_comp.text(1.4, y, label, fontsize=10,
                 color=TEXT_DIM, ha="left", fontfamily="monospace", va="center")
    y -= 0.9

# Divider
ax_comp.plot([5, 5], [1.5, 9.5], color="#30363d", linewidth=2)
ax_comp.text(5, 5.0, "vs", fontsize=14, color=TEXT_DIM, ha="center", va="center",
             fontfamily="monospace", fontstyle="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD, edgecolor="#30363d"))

# MPS side
ax_comp.text(7.5, 9.3, "METAL GPU", fontsize=16, fontweight="bold",
             color=BLUE, ha="center", fontfamily="monospace")
ax_comp.text(7.5, 8.65, "40 GPU cores", fontsize=10,
             color=TEXT_DIM, ha="center", fontfamily="monospace")
ax_comp.plot([5.3, 9.7], [8.3, 8.3], color="#30363d", linewidth=1)

mps_stats = [
    ("11.5M", "params", BLUE),
    ("764 ms", "per step", BLUE),
    ("4 layers", "DIM=small", TEXT_DIM),
    ("393 steps", "in 5 min", TEXT_DIM),
    ("42K steps", "in 9 hours", TEXT_DIM),
]
y = 7.7
for val, label, color in mps_stats:
    ax_comp.text(6.2, y, val, fontsize=15, fontweight="bold",
                 color=color, ha="right", fontfamily="monospace")
    ax_comp.text(6.4, y, label, fontsize=10,
                 color=TEXT_DIM, ha="left", fontfamily="monospace", va="center")
    y -= 0.9

# Headline callout
ax_comp.text(5, 1.6, "ANE: 6x bigger model, 8x faster", fontsize=14,
             fontweight="bold", color=GREEN, ha="center", fontfamily="monospace")
ax_comp.text(5, 1.0, "Same chip. Zero interference. Both run simultaneously.",
             fontsize=10, color=TEXT_DIM, ha="center", fontfamily="monospace")

# ═══════════════════════════════════════
# Right: ANE loss curves (proof it learns)
# ═══════════════════════════════════════
ax_loss = fig.add_axes([0.53, 0.42, 0.44, 0.46])
ax_loss.set_facecolor(CARD)
for spine in ax_loss.spines.values():
    spine.set_color("#30363d")

# Plot top 5 configs, smoothed, vs wall time
for key in ["D", "B", "F2", "E", "F1"]:  # worst to best
    cfg = ane_configs[key]
    s, l = loss_curves[key]
    if s:
        t = [step * cfg["ms"] / 1000 for step in s]
        # Smooth
        window = max(1, len(l) // 40)
        if len(l) > window:
            l_smooth = np.convolve(l, np.ones(window)/window, mode="valid")
            t_smooth = t[:len(l_smooth)]
        else:
            l_smooth, t_smooth = l, t
        is_win = key == "F1"
        lw = 3.5 if is_win else 1.5
        alpha = 1.0 if is_win else 0.5
        label = f"NL={cfg['nl']} S={cfg['seq']}"
        if is_win:
            label += "  (WINNER)"
        ax_loss.plot(t_smooth, l_smooth, color=cfg["color"], linewidth=lw,
                     alpha=alpha, label=label)

ax_loss.axvline(x=300, color=RED, linewidth=1.5, linestyle="--", alpha=0.4)
ax_loss.text(290, 6.0, "5 min ", fontsize=9, color=RED, fontfamily="monospace",
             ha="right")

ax_loss.set_xlabel("Wall Time (seconds)", fontsize=12, color=TEXT, fontfamily="monospace")
ax_loss.set_ylabel("Training Loss", fontsize=12, color=TEXT, fontfamily="monospace")
ax_loss.set_title("ANE Training — 5 configs, real data (climbmix-400b)",
                  fontsize=14, color=TEXT, fontfamily="monospace", pad=10)
ax_loss.tick_params(colors=TEXT_DIM)
ax_loss.grid(True, alpha=0.15, color=TEXT_DIM)
ax_loss.legend(loc="upper right", fontsize=9.5, facecolor=CARD, edgecolor="#30363d",
               labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Bottom: Key facts strip
# ═══════════════════════════════════════
ax_facts = fig.add_axes([0.03, 0.04, 0.94, 0.32])
ax_facts.set_facecolor(CARD)
for spine in ax_facts.spines.values():
    spine.set_color("#30363d")
ax_facts.set_xlim(0, 20)
ax_facts.set_ylim(0, 6)
ax_facts.set_xticks([])
ax_facts.set_yticks([])

# Four columns of facts
facts = [
    {
        "title": "THE HARDWARE",
        "color": GOLD,
        "items": [
            ("Apple M4 Max", TEXT),
            ("128 GB unified memory", TEXT_DIM),
            ("16 ANE cores (10.5 TFLOP/s)", TEXT_DIM),
            ("40 GPU cores (Metal/MPS)", TEXT_DIM),
            ("Shared memory, separate silicon", TEXT_DIM),
        ]
    },
    {
        "title": "WHAT WE FOUND",
        "color": GREEN,
        "items": [
            ("ANE sweet spot: NL=6, SEQ=512", TEXT),
            ("67.6M params at 99ms/step", GREEN),
            ("Loss: 9.05 -> 6.34 in 5 min", GREEN),
            ("9 configs tested, 2 rounds", TEXT_DIM),
            ("SRAM wall at SEQ=1024", TEXT_DIM),
        ]
    },
    {
        "title": "THE WILD PART",
        "color": CYAN,
        "items": [
            ("ANE invisible to Activity Monitor", TEXT),
            ("No CPU usage, no GPU usage shown", TEXT_DIM),
            ("Machine stays cool & silent", TEXT_DIM),
            ("Both accelerators run at once", CYAN),
            ("No blog post or paper exists for this", TEXT_DIM),
        ]
    },
    {
        "title": "HOW WE BUILT IT",
        "color": PURPLE,
        "items": [
            ("Native Obj-C, not CoreML/PyTorch", TEXT),
            ("Direct ANE hardware access", TEXT_DIM),
            ("Custom MIL kernels for fwd+bwd", TEXT_DIM),
            ("Vocab compaction: 32K -> 8K tokens", TEXT_DIM),
            ("Auto-sweep to find optimal config", TEXT_DIM),
        ]
    },
]

for i, section in enumerate(facts):
    x = 0.5 + i * 5
    ax_facts.text(x, 5.4, section["title"], fontsize=13, fontweight="bold",
                  color=section["color"], fontfamily="monospace")
    ax_facts.plot([x, x + 4.2], [5.0, 5.0], color="#30363d", linewidth=1)
    y = 4.5
    for text, color in section["items"]:
        ax_facts.text(x, y, text, fontsize=9.5, color=color, fontfamily="monospace")
        y -= 0.85

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=12, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch-ANE · github.com/ncdrone/autoresearch-ANE",
         fontsize=8, color=TEXT_DIM, alpha=0.35, ha="left", va="bottom", fontfamily="monospace")

plt.savefig("/Users/dan/Dev/autoresearch-ANE/viz/hero_summary.png",
            dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to viz/hero_summary.png")
