"""Quick visualization of a training run's loss curve."""
import subprocess
import re
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

print("Running train.py and capturing loss curve (this takes ~5 min)...")
print("You'll see a chart when it's done.\n")

steps, losses, pct_done = [], [], []

proc = subprocess.Popen(
    ["uv", "run", "train.py"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    bufsize=0
)

buffer = b""
for chunk in iter(lambda: proc.stdout.read(1), b""):
    if chunk == b"\r" or chunk == b"\n":
        line = buffer.decode("utf-8", errors="replace")
        buffer = b""
        # Parse: step 00042 (14.2%) | loss: 2.345678 | ...
        m = re.search(r"step\s+(\d+)\s+\(([\d.]+)%\)\s+\|\s+loss:\s+([\d.]+)", line)
        if m:
            steps.append(int(m.group(1)))
            pct_done.append(float(m.group(2)))
            losses.append(float(m.group(3)))
            if len(steps) % 50 == 0:
                print(f"  step {steps[-1]}, loss={losses[-1]:.4f}, {pct_done[-1]:.1f}% done")
        # Also print final summary lines
        if line.startswith("val_bpb:") or line.startswith("---") or line.startswith("training_") or line.startswith("total_") or line.startswith("peak_") or line.startswith("mfu_") or line.startswith("num_") or line.startswith("depth:"):
            print(line)
    else:
        buffer += chunk

proc.wait()

if not steps:
    print("No training steps captured. Something went wrong.")
    exit(1)

# Parse final val_bpb from the summary
val_bpb = None
for i in range(len(steps)):
    pass  # we just need the plot

print(f"\nCaptured {len(steps)} steps. Plotting...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left: Loss over steps
ax1.plot(steps, losses, color="#3498db", linewidth=1.2, alpha=0.8)
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("Smoothed Training Loss", fontsize=12)
ax1.set_title("Training Loss Curve (Baseline Run)", fontsize=13)
ax1.grid(True, alpha=0.2)

# Right: Loss over % of time budget
ax2.plot(pct_done, losses, color="#e74c3c", linewidth=1.2, alpha=0.8)
ax2.set_xlabel("Time Budget Used (%)", fontsize=12)
ax2.set_ylabel("Smoothed Training Loss", fontsize=12)
ax2.set_title("Loss vs Time Budget", fontsize=13)
ax2.grid(True, alpha=0.2)

fig.suptitle(f"Autoresearch Baseline — {len(steps)} steps on Apple Silicon (MPS)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("baseline_loss_curve.png", dpi=150, bbox_inches="tight")
print("Saved to baseline_loss_curve.png")
