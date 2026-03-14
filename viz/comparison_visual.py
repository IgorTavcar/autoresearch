"""Side-by-side comparison: H100 GPU vs M4 Max Mac for autoresearch."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Colors ---
BG = "#0d1117"
CARD = "#161b22"
ACCENT_GPU = "#58a6ff"
ACCENT_MAC = "#3fb950"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
BORDER_GPU = "#1f4068"
BORDER_MAC = "#1a4028"
HIGHLIGHT = "#f0883e"

fig = plt.figure(figsize=(16, 11), facecolor=BG)

# Title
fig.text(0.5, 0.96, "autoresearch: GPU vs Mac", fontsize=30, fontweight="bold",
         color=TEXT, ha="center", va="top", fontfamily="monospace")
fig.text(0.5, 0.92, "Same experiment. Different hardware. Different optimal model?",
         fontsize=14, color=HIGHLIGHT, ha="center", va="top", fontfamily="monospace")

# --- Bar chart comparisons ---
categories = [
    ("Training Steps\n(per 5 min)", 953, 368, "More steps = more learning"),
    ("Tokens Processed\n(millions)", 499.6, 25.1, "20x fewer tokens per run"),
    ("Model Size\n(M params)", 50.3, 11.5, "Smaller model fits the budget"),
    ("Depth\n(layers)", 8, 4, "Half the layers, same time"),
]

# Create 4 bar charts in a row
for i, (label, gpu_val, mac_val, note) in enumerate(categories):
    ax = fig.add_axes([0.06 + i * 0.235, 0.52, 0.19, 0.32])
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
        spine.set_linewidth(1)

    bars = ax.bar([0, 1], [gpu_val, mac_val], color=[ACCENT_GPU, ACCENT_MAC],
                  width=0.6, edgecolor="none", alpha=0.85)

    # Value labels on bars
    for bar, val in zip(bars, [gpu_val, mac_val]):
        fmt = f"{val:,.0f}" if val > 100 else f"{val:.1f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + gpu_val * 0.03,
                fmt, ha="center", va="bottom", fontsize=13, fontweight="bold",
                color=TEXT, fontfamily="monospace")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H100", "M4 Max"], fontsize=10, color=TEXT_DIM, fontfamily="monospace")
    ax.set_yticks([])
    ax.set_title(label, fontsize=10.5, color=TEXT, fontfamily="monospace", pad=10, linespacing=1.4)
    ax.text(0.5, -0.18, note, fontsize=8.5, color=TEXT_DIM, ha="center",
            transform=ax.transAxes, fontfamily="monospace", fontstyle="italic")
    ax.set_ylim(0, gpu_val * 1.2)

# --- The interesting part: key insight boxes ---
def insight_box(fig, x, y, w, h, title, body, accent):
    ax = fig.add_axes([x, y, w, h])
    ax.set_facecolor(CARD)
    for spine in ax.spines.values():
        spine.set_color(accent)
        spine.set_linewidth(1.5)
        spine.set_alpha(0.5)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(5, 7.5, title, fontsize=12, fontweight="bold", color=accent,
            ha="center", va="center", fontfamily="monospace")
    ax.text(5, 4.0, body, fontsize=9.5, color=TEXT_DIM, ha="center", va="center",
            fontfamily="monospace", linespacing=1.8, wrap=True,
            bbox=dict(facecolor="none", edgecolor="none", pad=10))

insight_box(fig, 0.04, 0.18, 0.28, 0.25,
            "THE QUESTION",
            "The H100 trains a 50M param\nmodel. The Mac trains 11.5M.\nDoes the AI agent find\ncompletely different wins\non each platform?",
            HIGHLIGHT)

insight_box(fig, 0.36, 0.18, 0.28, 0.25,
            "THE SETUP",
            "AI agent autonomously edits\nthe training code, runs 5 min,\nkeeps improvements, discards\nfailures. ~70 experiments\novernight. Zero human input.",
            TEXT_DIM)

insight_box(fig, 0.68, 0.18, 0.28, 0.25,
            "THE COST",
            "H100 cloud: ~$3/hr\nM4 Max: already on my desk\n\n8 hours of autonomous\nresearch for $0",
            ACCENT_MAC)

# Bottom description
fig.text(0.5, 0.1, "Running @karpathy's autoresearch on Apple Silicon overnight. Full results tomorrow.",
         fontsize=11, color=TEXT_DIM, ha="center", va="center",
         fontfamily="monospace")

# Watermark
fig.text(0.97, 0.025, "@danpacary", fontsize=12, color=TEXT_DIM, alpha=0.5,
         ha="right", va="bottom", fontfamily="monospace", fontstyle="italic")

# Credit
fig.text(0.03, 0.025, "based on karpathy/autoresearch", fontsize=9, color=TEXT_DIM, alpha=0.35,
         ha="left", va="bottom", fontfamily="monospace")

plt.savefig("autoresearch_comparison.png", dpi=200, bbox_inches="tight",
            facecolor=BG, edgecolor="none", pad_inches=0.3)
print("Saved to autoresearch_comparison.png")
