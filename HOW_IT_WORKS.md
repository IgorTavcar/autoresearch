# How autoresearch Works

A plain-English guide to the autonomous AI research loop.

---

## The One-Sentence Version

An AI agent **edits a training script, runs it for 5 minutes, checks if the model got better, keeps or discards the change, and repeats forever** — so you wake up to dozens of experiments and a better neural network.

---

## The Big Picture

```
 ┌──────────────────────────────────────────────────────────────┐
 │                        YOU (human)                           │
 │                                                              │
 │   1. Run prepare.py once (downloads data, trains tokenizer)  │
 │   2. Point the AI agent at program.md                        │
 │   3. Go to sleep                                             │
 │   4. Wake up, check results.tsv & analysis.ipynb             │
 └──────────────────────┬───────────────────────────────────────┘
                        │
                        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │                     AI AGENT (Claude)                        │
 │                                                              │
 │   Reads program.md for instructions, then loops:             │
 │                                                              │
 │   ┌─────────────┐    ┌─────────────┐    ┌────────────────┐   │
 │   │ Think of an │───▶│ Edit        │───▶│ Run train.py   │   │
 │   │ experiment  │    │ train.py    │    │ (5 min)        │   │
 │   └─────────────┘    └─────────────┘    └────────┬───────┘   │
 │                                                  │           │
 │                          ┌───────────────────────┘           │
 │                          ▼                                   │
 │                  ┌───────────────┐                           │
 │                  │ Got better?   │                           │
 │                  └───┬───────┬───┘                           │
 │                yes   │       │  no                           │
 │                      ▼       ▼                               │
 │               ┌────────┐ ┌─────────┐                         │
 │               │ KEEP   │ │ DISCARD │                         │
 │               │ commit │ │ revert  │                         │
 │               └────┬───┘ └────┬────┘                         │
 │                    │          │                              │
 │                    ▼          ▼                              │
 │               ┌────────────────────┐                         │
 │               │ Log to results.tsv │──▶ loop back            │
 │               └────────────────────┘                         │
 └──────────────────────────────────────────────────────────────┘
```

---

## Key Files at a Glance

| File | Who touches it | Purpose |
|------|---------------|---------|
| `program.md` | Human writes, agent reads | Instructions for the agent — what to do, what rules to follow |
| `train.py` | Agent edits | The model + training loop — the **only** file the agent changes |
| `prepare.py` | Nobody (read-only) | Downloads data, trains tokenizer, provides evaluation utilities |
| `results.tsv` | Agent appends | Log of every experiment: what was tried, did it help |
| `analysis.ipynb` | Human uses | Notebook to visualize progress after a run |

---

## Step by Step

### Phase 1: Setup (one-time, by you)

```
 ┌──────────┐      ┌────────────────────────────────────┐
 │prepare.py│─────▶│  ~/.cache/autoresearch/            │
 └──────────┘      │                                    │
                   │  data/                             │
                   │    shard_00000.parquet             │
                   │    shard_00001.parquet             │
                   │    ... (10 training shards)        │
                   │    shard_06542.parquet (validation)│
                   │                                    │
                   │  tokenizer/                        │
                   │    tokenizer.pkl  (BPE, 8192 vocab)│
                   │    token_bytes.pt (for BPB metric) │
                   │    metadata.json  (integrity hash) │
                   └────────────────────────────────────┘
```

1. **Downloads text data** — parquet shards from HuggingFace (`climbmix-400b-shuffle`)
2. **Trains a BPE tokenizer** — 8,192 token vocabulary using `rustbpe`
3. **Saves everything** to `~/.cache/autoresearch/`

This only runs once. After that, the data and tokenizer are reused.

### Phase 2: The Experiment Loop (autonomous)

You tell the agent: *"Read program.md and start experimenting."*

The agent then:

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                                                                 │
 │  1. CREATE branch: autoresearch/<tag>                           │
 │                                                                 │
 │  2. RUN baseline (unmodified train.py) → establishes starting   │
 │     val_bpb score                                               │
 │                                                                 │
 │  3. LOOP FOREVER:                                               │
 │     ┌─────────────────────────────────────────────────────────┐ │
 │     │                                                         │ │
 │     │  a. Think: "What if I increase the learning rate?"      │ │
 │     │                                                         │ │
 │     │  b. Edit train.py with the change                       │ │
 │     │                                                         │ │
 │     │  c. git commit -m "increase embedding LR to 0.8"        │ │
 │     │                                                         │ │
 │     │  d. uv run train.py  (runs for exactly 5 minutes)       │ │
 │     │                                                         │ │
 │     │  e. Read output:                                        │ │
 │     │       val_bpb:     0.9821   ← the score (lower=better)  │ │
 │     │       peak_vram_mb: 44100   ← memory used               │ │
 │     │                                                         │ │
 │     │  f. Compare to previous best:                           │ │
 │     │       0.9821 < 0.9979 → BETTER! Keep the commit.        │ │
 │     │                                                         │ │
 │     │  g. Append to results.tsv                               │ │
 │     │                                                         │ │
 │     │  h. Go back to (a) with a new idea                      │ │
 │     │                                                         │ │
 │     └─────────────────────────────────────────────────────────┘ │
 │                                                                 │
 └─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Review (by you)

When you come back:
- **`results.tsv`** has every experiment logged (kept, discarded, or crashed)
- **`git log`** on the branch shows only the winning experiments
- **`analysis.ipynb`** plots progress over time

---

## What the Agent Can and Cannot Do

```
 ┌──────────────────────────────────────────────────────────────┐
 │                       ALLOWED                                │
 │                                                              │
 │  ✓ Edit train.py (architecture, hyperparameters, optimizer)  │
 │  ✓ Run train.py                                              │
 │  ✓ Read any file                                             │
 │  ✓ Git commit / reset                                        │
 │  ✓ Write to results.tsv                                      │
 │                                                              │
 ├──────────────────────────────────────────────────────────────┤
 │                      NOT ALLOWED                             │
 │                                                              │
 │  ✗ Edit prepare.py, program.md, or any other file            │
 │  ✗ Add new dependencies                                      │
 │  ✗ Change the tokenizer or data pipeline                     │
 │  ✗ Exceed available GPU memory                               │
 │  ✗ Stop (the agent runs until you interrupt it)              │
 └──────────────────────────────────────────────────────────────┘
```

---

## The Model Being Trained

A small GPT-style transformer, trained from scratch on text data:

```
  Input tokens (sequence of 2048)
         │
         ▼
  ┌────────────────┐
  │  Token         │  Converts token IDs → vectors
  │  Embedding     │
  └──────┬─────────┘
         │
         ▼
  ┌────────────────┐
  │  Transformer   │ ×8-12 layers, each containing:
  │  Block         │
  │  ┌──────────┐  │   • RMS Normalization
  │  │Attention │  │   • Multi-head self-attention (with RoPE)
  │  │(sliding  │  │   • Sliding window: short/long pattern (SSSL)
  │  │ window)  │  │   • Flash Attention 3 kernel
  │  └──────────┘  │
  │  ┌──────────┐  │   • RMS Normalization
  │  │ MLP      │  │   • Linear → ReLU² → Linear
  │  │(feedfwd) │  │
  │  └──────────┘  │
  │  + residual    │   • Skip connections with learnable scaling
  └──────┬─────────┘
         │
         ▼
  ┌──────────────┐
  │ LM Head      │  Vectors → vocabulary probabilities
  │ (unembedding)│
  └──────┬───────┘
         │
         ▼
  Next-token prediction loss
  (cross-entropy → val_bpb)
```

**Optimizer**: MuonAdamW — a hybrid that uses:
- **Muon** (orthogonal updates) for weight matrices
- **AdamW** (adaptive updates) for embeddings and scalars

---

## The Metric: val_bpb

**Bits Per Byte** — how many bits the model needs to encode each byte of text.

- Measured on a **fixed validation set** (shard 06542, never used for training)
- **Lower is better** — the model is compressing text more efficiently
- Independent of vocabulary size — fair comparison across architectural changes

```
  Typical progression:

  Experiment   val_bpb    Status
  ─────────────────────────────────
  baseline     0.9979     keep       ← starting point
  higher LR    0.9821     keep       ← improvement!
  more heads   0.9925     discard    ← worse than 0.9821
  deeper model 0.9756     keep       ← new best!
  GeLU activ.  0.9801     discard    ← worse than 0.9756
  ...
```

---

## Results Tracking

### results.tsv

A tab-separated log the agent appends to after every experiment:

```
commit     val_bpb   memory_gb  status    description
a1b2c3d    0.9979    44.0       keep      baseline run
b2c3d4e    0.9821    44.2       keep      increase embedding LR to 0.8
c3d4e5f    0.9925    44.0       discard   16-head attention
0000000    0.0000    0.0        crash     doubled model width (OOM)
d4e5f6g    0.9756    43.8       keep      increase depth to 12
```

### Git Branch

```
master ─── a1b ─── b2c ─── d4e
            │       │       │
         baseline  LR+    depth+

     (only winning experiments survive on the branch)
```

Failed experiments are reverted with `git reset` — they leave no trace in git, only in `results.tsv`.

---

## Typical Overnight Run

```
  6 PM   ──▶  Human starts agent
               Agent creates branch, runs baseline

  6:10   ──▶  Experiment 1: tweak learning rate → keep
  6:20   ──▶  Experiment 2: change activation → discard
  6:30   ──▶  Experiment 3: deeper model → keep
  ...         (repeating every ~8 min)

  2 AM   ──▶  ~60 experiments completed
               ~15 kept, ~40 discarded, ~5 crashed

  8 AM   ──▶  Human wakes up
               Opens results.tsv: sees all 60 experiments
               Runs analysis.ipynb: sees val_bpb curve
               Reviews git log: sees the 15 improvements
               val_bpb dropped from 0.998 → 0.951
```

---

## File & Data Flow Diagram

```
 ┌────────────┐         ┌─────────────────────────────────────┐
 │ HuggingFace│────────▶│  ~/.cache/autoresearch/             │
 │ (remote)   │  data   │  ├── data/*.parquet                 │
 └────────────┘         │  └── tokenizer/                     │
                        │      ├── tokenizer.pkl              │
                        │      ├── token_bytes.pt             │
       prepare.py ──────│      └── metadata.json              │
       (runs once)      └──────────────┬──────────────────────┘
                                       │
                                       │ loaded at runtime
                                       ▼
 ┌────────────┐  edits   ┌─────────────────────┐   outputs
 │ AI Agent   │─────────▶│     train.py        │──────────────┐
 │ (Claude)   │          │  (model + training) │              │
 │            │◀─────────│                     │              │
 │            │ reads    └─────────────────────┘              │
 │            │ output                                        │
 │            │                                               ▼
 │            │  appends  ┌─────────────────────┐    val_bpb: 0.982
 │            │──────────▶│   results.tsv       │    peak_vram_mb: 44100
 │            │           └─────────────────────┘
 │            │
 │            │  commits  ┌───────────────────────┐
 │            │──────────▶│   git branch          │
 │            │  /resets  │  autoresearch/<tag>   │
 └────────────┘           └───────────────────────┘
                                    │
                                    │  reviewed by human
                                    ▼
                          ┌─────────────────────┐
                          │  analysis.ipynb     │
                          │  (plots & insights) │
                          └─────────────────────┘
```

---

## Why This Design Works

1. **One file to edit** — all changes are in `train.py`, making diffs clean and reviewable
2. **Fixed 5-minute budget** — every experiment takes the same time, enabling fair comparison
3. **Immutable evaluation** — `prepare.py` never changes, so metrics are always comparable
4. **Git as lab notebook** — kept experiments are commits; the branch tells the story
5. **Simple success metric** — lower `val_bpb` = better model, no ambiguity
6. **No human in the loop** — the agent runs all night without needing approval
7. **Fail-safe** — crashes and regressions are caught, logged, and skipped automatically
