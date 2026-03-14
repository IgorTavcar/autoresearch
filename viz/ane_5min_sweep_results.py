"""ANE 5-Minute Sweep Results — All 9 Configs — M4 Max"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
PINK = "#f778ba"
LIME = "#a5d6a7"

# All 9 configs — Round 1 (A-E) + Round 2 (F1-F4)
configs = {
    "A":  {"nl": 2,  "seq": 1024, "params": 39.3,  "ms": 140.6, "steps": 1060, "color": ORANGE},
    "B":  {"nl": 4,  "seq": 512,  "params": 53.5,  "ms": 75.9,  "steps": 1850, "color": BLUE},
    "C":  {"nl": 4,  "seq": 1024, "params": 53.5,  "ms": 191.6, "steps": 740,  "color": PURPLE},
    "D":  {"nl": 8,  "seq": 256,  "params": 81.8,  "ms": 63.9,  "steps": 2420, "color": CYAN},
    "E":  {"nl": 8,  "seq": 512,  "params": 81.8,  "ms": 123.4, "steps": 1120, "color": GREEN},
    "F1": {"nl": 6,  "seq": 512,  "params": 67.6,  "ms": 99.2,  "steps": 3000, "color": GOLD},
    "F2": {"nl": 10, "seq": 512,  "params": 96.0,  "ms": 146.9, "steps": 2000, "color": PINK},
    "F3": {"nl": 12, "seq": 512,  "params": 110.2, "ms": 169.1, "steps": 1670, "color": RED},
    "F4": {"nl": 8,  "seq": 768,  "params": 81.8,  "ms": 180.5, "steps": 1760, "color": LIME},
}

# Parse loss curves from logs
log_dir = os.path.join(os.path.dirname(__file__), "..", "results", "sweep_5min")
loss_curves = {}
smoothed_final = {}

for key, cfg in configs.items():
    log_file = os.path.join(log_dir, f"{key}_nl{cfg['nl']}_s{cfg['seq']}.log")
    steps_list = []
    losses = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            for line in f:
                if line.startswith("step "):
                    parts = line.split()
                    try:
                        step = int(parts[1])
                        loss = float(parts[2].replace("loss=", ""))
                        steps_list.append(step)
                        losses.append(loss)
                    except (ValueError, IndexError):
                        pass
    loss_curves[key] = (steps_list, losses)
    if len(losses) >= 20:
        smoothed_final[key] = np.mean(losses[-20:])
    elif losses:
        smoothed_final[key] = np.mean(losses[-5:])
    else:
        smoothed_final[key] = float("inf")

ranked = sorted(configs.keys(), key=lambda k: smoothed_final.get(k, 99))
winner = ranked[0]

# ═══════════════════════════════════════
# Figure layout: 3 rows × 2 cols
# ═══════════════════════════════════════
fig = plt.figure(figsize=(26, 22), facecolor=BG)
gs = gridspec.GridSpec(3, 2, hspace=0.32, wspace=0.25,
                       left=0.06, right=0.96, top=0.92, bottom=0.04)

# Title
fig.text(0.5, 0.975, "ANE 5-Minute Training Sweep — Complete Results", fontsize=30,
         fontweight="bold", color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.945, "climbmix-400b  ·  631M tokens  ·  Apple Neural Engine  ·  M4 Max  ·  9 configs over 2 rounds",
         fontsize=13, color=TEXT_DIM, ha="center", va="top", fontfamily="monospace")

# ═══════════════════════════════════════
# Panel 1 (top-left): NL × SEQ test matrix
# ═══════════════════════════════════════
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(CARD)
for spine in ax1.spines.values():
    spine.set_color("#30363d")

# Build the matrix
all_nls = sorted(set(c["nl"] for c in configs.values()))   # [2,4,6,8,10,12]
all_seqs = sorted(set(c["seq"] for c in configs.values()))  # [256,512,768,1024]

matrix = np.full((len(all_nls), len(all_seqs)), np.nan)
label_matrix = [[None]*len(all_seqs) for _ in range(len(all_nls))]

for key, cfg in configs.items():
    r = all_nls.index(cfg["nl"])
    c = all_seqs.index(cfg["seq"])
    matrix[r, c] = smoothed_final[key]
    label_matrix[r][c] = key

# Custom colormap: green (good) to red (bad)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("loss", ["#2ea043", "#e3b341", "#f85149"])
vmin, vmax = 6.2, 7.4

im = ax1.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax,
                origin="upper", interpolation="nearest")

# Annotate each cell
for r in range(len(all_nls)):
    for c in range(len(all_seqs)):
        val = matrix[r, c]
        key = label_matrix[r][c]
        if key is not None:
            is_win = key == winner
            # Cell text
            ax1.text(c, r - 0.18, f"{key}", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if is_win else TEXT,
                     fontfamily="monospace")
            ax1.text(c, r + 0.12, f"{val:.3f}", ha="center", va="center",
                     fontsize=13, fontweight="bold",
                     color="white" if is_win else TEXT,
                     fontfamily="monospace")
            cfg = configs[key]
            ax1.text(c, r + 0.38, f"{cfg['steps']}st · {cfg['ms']:.0f}ms", ha="center",
                     va="center", fontsize=7.5, color=TEXT_DIM, fontfamily="monospace")
            if is_win:
                # Gold border around winner cell
                from matplotlib.patches import Rectangle
                rect = Rectangle((c - 0.48, r - 0.48), 0.96, 0.96,
                                 linewidth=3, edgecolor=GOLD, facecolor="none")
                ax1.add_patch(rect)
        else:
            ax1.text(c, r, "—", ha="center", va="center",
                     fontsize=14, color="#30363d", fontfamily="monospace")

ax1.set_xticks(range(len(all_seqs)))
ax1.set_xticklabels([str(s) for s in all_seqs], fontsize=12, color=TEXT, fontfamily="monospace")
ax1.set_yticks(range(len(all_nls)))
ax1.set_yticklabels([str(n) for n in all_nls], fontsize=12, color=TEXT, fontfamily="monospace")
ax1.set_xlabel("Sequence Length (SEQ)", fontsize=13, color=TEXT, fontfamily="monospace", labelpad=8)
ax1.set_ylabel("Number of Layers (NL)", fontsize=13, color=TEXT, fontfamily="monospace", labelpad=8)
ax1.set_title("Test Matrix — Smoothed Loss (green = better)", fontsize=15, color=TEXT,
              fontfamily="monospace", pad=12)
ax1.tick_params(colors=TEXT_DIM)

# ═══════════════════════════════════════
# Panel 2 (top-right): Loss curves vs wall time
# ═══════════════════════════════════════
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(CARD)
for spine in ax2.spines.values():
    spine.set_color("#30363d")

for key in reversed(ranked):
    cfg = configs[key]
    s, l = loss_curves[key]
    if s:
        t = [step * cfg["ms"] / 1000 for step in s]
        # Rolling average for smoother curves
        window = max(1, len(l) // 50)
        if len(l) > window:
            l_smooth = np.convolve(l, np.ones(window)/window, mode="valid")
            t_smooth = t[:len(l_smooth)]
        else:
            l_smooth, t_smooth = l, t
        is_win = key == winner
        lw = 3.5 if is_win else 1.3
        alpha = 1.0 if is_win else 0.5
        label = f"{key}: NL={cfg['nl']} S={cfg['seq']} → {smoothed_final[key]:.3f}"
        ax2.plot(t_smooth, l_smooth, color=cfg["color"], linewidth=lw, alpha=alpha, label=label)

ax2.axvline(x=300, color=RED, linewidth=2, linestyle="--", alpha=0.4)
ax2.text(290, ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 6.0, "5 min ",
         fontsize=10, color=RED, fontfamily="monospace", ha="right", va="bottom")
ax2.set_xlabel("Wall Time (seconds)", fontsize=13, color=TEXT, fontfamily="monospace")
ax2.set_ylabel("Training Loss (smoothed)", fontsize=13, color=TEXT, fontfamily="monospace")
ax2.set_title("Loss vs Wall Time — All 9 Configs", fontsize=15, color=TEXT,
              fontfamily="monospace", pad=12)
ax2.tick_params(colors=TEXT_DIM)
ax2.grid(True, alpha=0.15, color=TEXT_DIM)
ax2.legend(loc="upper right", fontsize=8.5, facecolor=CARD, edgecolor="#30363d",
           labelcolor=TEXT, framealpha=0.9)

# ═══════════════════════════════════════
# Panel 3 (mid-left): Depth curve at SEQ=512
# ═══════════════════════════════════════
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(CARD)
for spine in ax3.spines.values():
    spine.set_color("#30363d")

seq512 = {k: v for k, v in configs.items() if v["seq"] == 512}
seq512_sorted = sorted(seq512.keys(), key=lambda k: seq512[k]["nl"])
nls = [seq512[k]["nl"] for k in seq512_sorted]
losses_512 = [smoothed_final[k] for k in seq512_sorted]
colors_512 = [seq512[k]["color"] for k in seq512_sorted]
steps_512 = [seq512[k]["steps"] for k in seq512_sorted]

ax3.fill_between(nls, losses_512, alpha=0.08, color=GREEN)
ax3.plot(nls, losses_512, color=TEXT_DIM, linewidth=2.5, alpha=0.6, zorder=1, marker="o",
         markersize=0)

for i, key in enumerate(seq512_sorted):
    is_win = key == winner
    sz = 250 if is_win else 120
    ec = GOLD if is_win else "#30363d"
    lw_pt = 3.5 if is_win else 1.5
    ax3.scatter(nls[i], losses_512[i], color=colors_512[i], s=sz,
                edgecolors=ec, linewidths=lw_pt, zorder=3)
    # Label above or below depending on position
    va = "bottom"
    offset = 0.04
    ax3.text(nls[i], losses_512[i] + offset,
             f"  {key}\n  {losses_512[i]:.3f}\n  {steps_512[i]} steps",
             ha="center", va=va, fontsize=9.5,
             color=GOLD if is_win else colors_512[i],
             fontfamily="monospace", fontweight="bold" if is_win else "normal",
             linespacing=1.3)

ax3.set_xlabel("Number of Layers (NL)", fontsize=13, color=TEXT, fontfamily="monospace")
ax3.set_ylabel("Smoothed Loss", fontsize=13, color=TEXT, fontfamily="monospace")
ax3.set_title("Depth Scaling at SEQ=512 — Clear Minimum at NL=6", fontsize=15, color=TEXT,
              fontfamily="monospace", pad=12)
ax3.set_xticks(nls)
ax3.tick_params(colors=TEXT_DIM)
ax3.grid(True, alpha=0.15, color=TEXT_DIM)

# ═══════════════════════════════════════
# Panel 4 (mid-right): Ranked bar chart (zoomed)
# ═══════════════════════════════════════
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(CARD)
for spine in ax4.spines.values():
    spine.set_color("#30363d")

bar_losses = [smoothed_final[k] for k in ranked]
bar_colors = [configs[k]["color"] for k in ranked]

bars = ax4.barh(range(len(ranked)), bar_losses, color=bar_colors, alpha=0.85,
                height=0.65, edgecolor="#30363d", linewidth=1.5)
bars[0].set_edgecolor(GOLD)
bars[0].set_linewidth(3)

# Zoom to relevant range
loss_min = min(bar_losses) - 0.15
loss_max = max(bar_losses) + 0.15
ax4.set_xlim(loss_min, loss_max)

for i, key in enumerate(ranked):
    cfg = configs[key]
    loss = bar_losses[i]
    # Label on the bar
    ax4.text(loss - 0.02, i,
             f" {loss:.3f} ", ha="right", va="center", fontsize=12,
             fontweight="bold", color="white", fontfamily="monospace")
    # Config info on left
    label = f"{key}  NL={cfg['nl']:>2d}  S={cfg['seq']:>4d}  {cfg['steps']:>4d}st  {cfg['ms']:>5.0f}ms"
    ax4.text(loss_min + 0.01, i, label, ha="left", va="center", fontsize=9,
             color=TEXT_DIM, fontfamily="monospace")

ax4.set_yticks(range(len(ranked)))
ax4.set_yticklabels(["" for _ in ranked])
ax4.set_xlabel("Smoothed Loss (lower = better)", fontsize=13, color=TEXT, fontfamily="monospace")
ax4.set_title("All 9 Configs Ranked", fontsize=15, color=TEXT,
              fontfamily="monospace", pad=12)
ax4.tick_params(colors=TEXT_DIM)
ax4.grid(True, alpha=0.15, color=TEXT_DIM, axis="x")
ax4.invert_yaxis()

# ═══════════════════════════════════════
# Panel 5+6 (bottom): Key findings spanning full width
# ═══════════════════════════════════════
ax5 = fig.add_subplot(gs[2, :])
ax5.set_facecolor(CARD)
for spine in ax5.spines.values():
    spine.set_color(GOLD)
    spine.set_linewidth(1.5)
    spine.set_alpha(0.4)
ax5.set_xlim(0, 20)
ax5.set_ylim(0, 4)
ax5.set_xticks([])
ax5.set_yticks([])

wcfg = configs[winner]

# Winner banner
ax5.text(10, 3.65, f"WINNER:  Config {winner}  —  NL={wcfg['nl']}  SEQ={wcfg['seq']}  "
         f"{wcfg['ms']:.0f}ms/step  {wcfg['steps']} steps  →  loss {smoothed_final[winner]:.3f}",
         fontsize=18, fontweight="bold", color=GOLD, ha="center", fontfamily="monospace")
ax5.plot([0.3, 19.7], [3.35, 3.35], color="#30363d", linewidth=1)

# Three columns of findings
col1_x, col2_x, col3_x = 0.5, 7.0, 13.5

# Column 1: What we learned
findings_1 = [
    ("DEPTH", GOLD, True, 14),
    ("Sweet spot = NL=6", TEXT, False, 11),
    ("NL=4: 6.74  NL=6: 6.34  NL=8: 6.94", TEXT_DIM, False, 10),
    ("Too shallow = underfits", TEXT_DIM, False, 10),
    ("Too deep = too few steps in 5 min", TEXT_DIM, False, 10),
]

# Column 2: Sequence findings
findings_2 = [
    ("SEQUENCE LENGTH", GOLD, True, 14),
    ("Sweet spot = SEQ=512", TEXT, False, 11),
    ("SEQ=256: fast but too little context", TEXT_DIM, False, 10),
    ("SEQ=768: 180ms/step, 40% slower", TEXT_DIM, False, 10),
    ("SEQ=1024: hits SRAM wall, too slow", TEXT_DIM, False, 10),
]

# Column 3: The headline
findings_3 = [
    ("THE ANE IS LEARNING", GREEN, True, 14),
    (f"Loss: 9.05 -> {smoothed_final[winner]:.2f} in 5 min", GREEN, False, 11),
    ("No GPU, no CPU, cool & silent", GREEN, False, 10),
    ("Same finding as MPS: more steps wins", TEXT_DIM, False, 10),
    (f"Next: overnight @ NL={wcfg['nl']} SEQ={wcfg['seq']}", CYAN, False, 11),
]

for col_x, findings in [(col1_x, findings_1), (col2_x, findings_2), (col3_x, findings_3)]:
    y = 2.9
    for text, color, bold, size in findings:
        ax5.text(col_x, y, text, fontsize=size, color=color,
                 fontfamily="monospace", fontweight="bold" if bold else "normal")
        y -= 0.52

# Watermark
fig.text(0.97, 0.005, "@danpacary", fontsize=11, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")
fig.text(0.03, 0.005, "autoresearch-ANE · Apple Neural Engine · climbmix training · 9 configs",
         fontsize=8, color=TEXT_DIM, alpha=0.35, ha="left", va="bottom", fontfamily="monospace")

plt.savefig("/Users/dan/Dev/autoresearch-ANE/viz/ane_5min_sweep_results.png",
            dpi=200, bbox_inches="tight", facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to viz/ane_5min_sweep_results.png")
