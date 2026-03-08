"""Batch size sweep at depth 4. Chains multiple configs automatically."""
import subprocess
import re
import time

TRAIN_PY = "train.py"

tests = [
    # (DEVICE_BATCH_SIZE, TOTAL_BATCH_SIZE, label)
    (64,  2**17, "batch64_total128K"),
    (128, 2**18, "batch128_total256K"),
    (256, 2**19, "batch256_total512K"),
    (32,  2**18, "batch32_total256K"),
    (32,  2**19, "batch32_total512K"),
]

def set_config(device_batch, total_batch):
    with open(TRAIN_PY) as f:
        code = f.read()
    code = re.sub(r"^DEVICE_BATCH_SIZE = \d+.*$",
                  f"DEVICE_BATCH_SIZE = {device_batch}  # batch sweep",
                  code, flags=re.MULTILINE)
    code = re.sub(r"^TOTAL_BATCH_SIZE = .*$",
                  f"TOTAL_BATCH_SIZE = {total_batch} # batch sweep ({total_batch//1024}K)",
                  code, flags=re.MULTILINE)
    with open(TRAIN_PY, "w") as f:
        f.write(code)

def read_result(log_path):
    with open(log_path) as f:
        text = f.read()
    def extract(key):
        m = re.search(rf"^{key}:\s+([\d.]+)", text, re.MULTILINE)
        return float(m.group(1)) if m else None
    return {
        "val_bpb": extract("val_bpb"),
        "steps": int(extract("num_steps")) if extract("num_steps") else None,
        "tokens_M": extract("total_tokens_M"),
    }

# Pre-existing results
results = [
    {"label": "batch16_total65K", "device_batch": 16, "total_batch": 65536,
     "val_bpb": 1.3122, "steps": 368},
    {"label": "batch32_total65K", "device_batch": 32, "total_batch": 65536,
     "val_bpb": 1.3091, "steps": 393},
]

# Check if batch32_total65K just finished
try:
    r = read_result("run_batch32.log")
    if r["val_bpb"]:
        results.append({"label": "batch32_total65K", "device_batch": 32,
                        "total_batch": 65536, **r})
        print(f"batch32_total65K: val_bpb={r['val_bpb']}, steps={r['steps']}")
except:
    pass

for device_batch, total_batch, label in tests:
    print(f"\n{'='*60}")
    print(f"  {label}: DEVICE_BATCH={device_batch}, TOTAL_BATCH={total_batch}")
    print(f"{'='*60}")

    set_config(device_batch, total_batch)
    log_path = f"run_{label}.log"
    t0 = time.time()

    try:
        proc = subprocess.run(["uv", "run", "train.py"], capture_output=True,
                              text=True, timeout=700)
        with open(log_path, "w") as f:
            f.write(proc.stdout)
            f.write(proc.stderr)
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 700s")
        results.append({"label": label, "device_batch": device_batch,
                        "total_batch": total_batch, "val_bpb": None, "steps": None})
        continue

    elapsed = time.time() - t0
    r = read_result(log_path)
    print(f"  val_bpb: {r['val_bpb']}")
    print(f"  steps:   {r['steps']}")
    print(f"  tokens:  {r['tokens_M']}M")
    print(f"  wall:    {elapsed:.0f}s")
    results.append({"label": label, "device_batch": device_batch,
                    "total_batch": total_batch, **r})

# Restore defaults
set_config(16, 2**16)

# Print summary
print(f"\n{'='*60}")
print(f"  BATCH SWEEP RESULTS (depth=4)")
print(f"{'='*60}")
print(f"{'Label':<25} {'Batch':>6} {'Total':>8} {'Steps':>6} {'val_bpb':>10}")
print("-" * 60)
for r in results:
    bpb = f"{r['val_bpb']:.6f}" if r.get('val_bpb') else "FAIL"
    steps = str(r.get('steps', '?'))
    tb = f"{r['total_batch']//1024}K"
    print(f"{r['label']:<25} {r['device_batch']:>6} {tb:>8} {steps:>6} {bpb:>10}")

best = min((r for r in results if r.get('val_bpb')), key=lambda r: r['val_bpb'])
print(f"\nBEST: {best['label']} — val_bpb={best['val_bpb']:.6f}")
