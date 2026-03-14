# (auto) autoresearch

This is the search, on top of Karpathy's hyperparameter search. Basically, it's a search for a search.

## My Idea

Karpathy's autoresearch is brilliant. It works, but there is a catch: the "Blank Page Problem" of AI. LLMs are fundamentally reactive engines. They are designed to follow, not to lead. If the leader (the user) is asleep, the follower (the agent) defaults to the "average" of all its knowledge, which is mediocre and boring, and gravitates toward safe, incremental moves.

But move 37 was neither safe nor average.

To fix this, we need another piece of software that acts as a "Chaos Monkey" or a "Creative Director." Think of it as the mutation rate in a genetic algorithm; without it, you can get stuck in a local minimum forever.

This repo is my attempt to build that monkey. This is a search for a search.

## Director Architecture (from ArmanJR-Lab)

Each experiment method lives in its own directory with `train.py`, `prepare.py`, and `program.md`. The `baseline/` directory is Karpathy's vanilla autoresearch loop. Other directories (e.g. `mad-scientist/`) add a **director** — a Go binary that generates creative research directives before each experiment iteration. The director works in three steps:

1. **Summarize** the current `train.py` via DeepSeek Chat (so it always knows the actual code state)
2. **Fetch** a random ML paper abstract from arxiv (external novelty injection)
3. **Generate** a directive via DeepSeek Reasoner, combining code summary + experiment history + paper into one specific idea

The output is a tentative suggestion ("I think you could try...") so the upstream agent reasons about it rather than blindly executing.

> I chose Go because I wanted the running agent to have absolutely zero context about the director and to see it as a black box. Just a binary that spits out ideas.

### Director Structure

```
baseline/                 # vanilla autoresearch (control group)
  train.py, prepare.py, program.md
  .claude/                # agent confinement (hooks + settings)

mad-scientist/            # experiment with director-driven exploration
  train.py, prepare.py, program.md, director, .env
  .claude/                # agent confinement (hooks + settings)

mad-scientist-3-11/       # second director run (full paper summaries, lower temp)
  train.py, prepare.py, program.md, director, .env
  .claude/                # agent confinement (hooks + settings)

director/                 # director source code
  main.go
  configs/*.json          # per-experiment config (prompts, arxiv terms, model, temp)
  logs/api_calls.jsonl    # centralized API call log across all experiments
  .env                    # DEEPSEEK_API_KEY

results/                  # tracking (gitignored)
  <method>/<run-id>/results.tsv

analysis.ipynb            # cross-method comparison (envelope, terminal perf, stall)
Makefile
```

### Current Experiments

| Name | Director | Description |
|------|----------|-------------|
| `baseline` | None | Vanilla autoresearch. The agent decides what to try next on its own. Control group. |
| `mad-scientist` | DeepSeek Reasoner (temp 1.2) | Summarizes current code, reads experiment history, fetches a random ML paper abstract from arxiv, then generates a bold directive framed as a suggestion. Combines code awareness + historical context + external novelty. |
| `mad-scientist-3-11` | DeepSeek Reasoner (temp 1.0) | Iteration on mad-scientist: feeds full paper summaries (not just abstracts) via DeepSeek-summarized ar5iv fetches, narrower arxiv categories (cs.LG + cs.CL only), stricter system prompt with scale-awareness rules and history-dedup guidance. |

#### Director config differences: mad-scientist vs mad-scientist-3-11

| | `mad-scientist` | `mad-scientist-3-11` |
|---|---|---|
| Temperature | 1.2 | 1.0 |
| Paper input | Abstract only (`{{paper_abstract}}`) | Full summary (`{{paper_abstract}}` + `{{paper_summary}}`) |
| arxiv categories | 5 (cs.LG, cs.CL, cs.AI, cs.CV, stat.ML) | 2 (cs.LG, cs.CL) |
| System prompt | Generic "small GPT", loose rules | Specifies ~10M params, stricter rules (read history, one thing at a time, scale-aware) |
| Max response | Not specified | 3 paragraphs |

#### Mad-Scientist Runs vs Original Baseline (Karpathy's H100)

![Relative improvement trajectory](h2h_relative_improvement.png)

| | original-baseline | mad-scientist | mad-scientist-3-11 |
|---|---|---|---|
| Experiments | 126 | 96 | 96 |
| Keeps (rate) | 23 (18.3%) | 22 (22.9%) | 16 (16.7%) |
| Total improvement | 2.83% | 3.94% | 2.79% |
| Improvement/experiment | 0.0224% | 0.0411% | 0.0291% |
| Longest stall | 25 | 15 | 31 |
| Best BPB | 0.969686 | 1.286357 | 1.302036 |

**mad-scientist** remains the strongest run — highest keep rate, most total improvement, and shortest stalls. **mad-scientist-3-11** underperformed despite having richer paper context: lower keep rate (16.7% vs 22.9%), less total improvement (2.79% vs 3.94%), and a 31-iteration stall at the end where it couldn't break past 1.302. The tighter prompt and lower temperature may have over-constrained the director, producing more conservative suggestions that failed to escape local minima.

##### Key moments from mad-scientist (run 1)

The biggest single jump came at **iteration 44** — the agent removed the logit softcap (tanh clamping at ±15), dropping val_bpb from 1.309 to 1.299 in one step. Notably, the director had suggested something entirely different (linear attention to break an 8-iteration stall), but the researcher agent ignored it and made a simpler, more effective change on its own. The director's value here was indirect: by flagging the stall and pushing for radical action, it prompted the agent to re-examine the code and spot the softcap as dead weight — something it had overlooked for 43 iterations.

The second jump came at **iteration 67** — this time the agent followed the director's advice almost exactly. After another 8-iteration stall (best stuck at 1.294), the director suggested reducing attention heads from 6 to 4 while increasing HEAD_DIM from 64 to 96 to preserve the embedding dimension. No paper was fetched; this was purely the director's own reasoning. The agent implemented it verbatim (val_bpb 1.294 → 1.290), then kept pushing the same idea through iterations 68–69 (HEAD_DIM=128 → 192), reaching 1.287 before single-head attention went too far. Both jumps were triggered by the same stall-detection mechanism ("8 consecutive failures → go radical"), but with opposite dynamics: iteration 44 succeeded by ignoring the director, iteration 67 by following it.

##### Key moments from mad-scientist-3-11 (run 2)

The biggest breakthrough came at **iteration 43** — replacing ReLU² with SwiGLU activation (iso-parameter), jumping from 1.316 to 1.306. This was followed by a productive stretch (iterations 47–65) of incremental optimizer tuning that pushed BPB down to 1.302. After iteration 65, the run hit a wall: 31 consecutive experiments without improvement, exploring attention tweaks, initialization changes, and architectural modifications — none broke through.

### Director Commands

```bash
# List available director configs
make list

# Build + deploy director for an experiment (default: linux/arm64 for NVIDIA Jetson)
make deploy EXPERIMENT=mad-scientist

# Build for local macOS instead
make deploy EXPERIMENT=mad-scientist GOOS=darwin GOARCH=arm64

# Run the director (from inside an experiment directory)
./director --verbose

# Run training (from inside an experiment directory)
python3 train.py
```

### Adding a new experiment method

1. Copy `baseline/` to a new directory
2. Create `director/configs/<name>.json` with custom system prompt, user prompt, arxiv terms, model, temperature
3. `make deploy EXPERIMENT=<name>`
4. Edit `program.md` in the new directory to integrate the director into the loop

---

## Apple Neural Engine Training (from ncdrone)

**Apple Silicon LLM training — three accelerators, one chip.**

Autonomous AI research on M4 Max using all three compute paths Apple Silicon offers: the Apple Neural Engine (ANE) via native Obj-C, the GPU via MLX, and the GPU via PyTorch/MPS. Forked from [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Same protocol: an AI agent modifies training code, runs 5-minute experiments, evaluates `val_bpb`, keeps or discards, and loops overnight. But instead of one H100, we're running on a laptop chip — and discovering what works (and what doesn't) on Apple Silicon.

### ANE Results

**ANE (native Obj-C, Apple Neural Engine):**
- 67.6M param GPT, 6 layers, SEQ=512, ~99ms/step
- Best loss: 5.81 (LR=2e-4, 10K steps)
- ANE is invisible to Activity Monitor — runs alongside GPU with zero interference
- Key challenge: activation instability on long runs (cosine schedule must match run length)

**MPS (PyTorch, Metal GPU):**
- 11.5M param GPT, val_bpb=1.308 after 79 autonomous experiments
- bf16 confirmed 2.6x slower on Apple Silicon — fp32 is faster
- H100 findings (embedding WD, init scaling) do not transfer to MPS

**MLX (Apple's native ML framework) — [`mlx/`](mlx/):**
- ~50M param GPT, val_bpb=1.665 baseline (agent optimizing now)
- Native bf16, unified memory, no translation layer
- Replaced MPS — ported from [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx)

### ANE Quick Start

```bash
# ANE (native, macOS Apple Silicon only)
cd native && make all
make test-ane              # verify ANE hardware access
make bench-sram            # probe SRAM performance cliffs
./build/train_overnight_nl6_s512 --steps 10000 --scratch --lr 2e-4 \
  --data data/train.bin --val data/val.bin

# MLX (recommended for Apple Silicon GPU)
cd mlx && uv sync
uv run prepare.py --num-shards 8
uv run train.py

# MPS (retired, kept for reference)
cp pyproject_mac.toml pyproject.toml && uv sync
uv run prepare.py --num-shards 8
uv run train_mac.py
```

### ANE Architecture

```
native/             — ANE hardware-level training (Obj-C, private APIs)
  runtime/          — ANE interface (_ANEInMemoryModel, IOSurface)
  mil/              — MIL code generation, dynamic weight pipeline
  training/         — training loop, CPU fallback ops (RMSNorm, Adam)
  bridge/           — C API for Python ctypes
  probes/           — hardware exploration (SRAM limits, weight patching)

mlx/                — MLX GPU training (Apple's native ML framework)
  train.py          — model + optimizer + loop (agent modifies this)
  prepare.py        — data prep, tokenizer, evaluation (read-only)
  program.md        — agent instructions

train.py            — NVIDIA GPU training (upstream, CUDA)
train_mac.py        — Apple Silicon training (MPS backend, retired)
prepare.py          — data prep, tokenizer, evaluation (read-only)
program.md          — agent instructions
viz/                — result visualizations
```

### Key concept: dynamic weight pipeline (ANE)

Weights are packed into the IOSurface input alongside activations. Kernels compile once at startup; weight updates are just `memcpy` — no recompilation needed. This is the core innovation over [maderix/ANE](https://github.com/maderix/ANE) which rebaked weights into compiled kernels.

### Key findings

- **ANE: 6x bigger model, 8x faster** than MPS on the same chip
- **Both accelerators run simultaneously** with zero interference
- **ANE timing breakdown:** 33% ANE compute, 30% IO, 37% CPU (classifier is 22% bottleneck)
- **Depth U-curve at SEQ=512:** NL=4(6.74) → NL=6(6.34) → NL=8(6.94) → NL=12(7.14)
- **SRAM wall at SEQ=1024** — ANE runs out of on-chip memory
- **Cosine schedule length must match actual run length** or activations explode

---

## Jetson AGX Orin Port

This fork has been adapted to run on an **NVIDIA Jetson AGX Orin 32GB** (JetPack 6, CUDA 12.6, PyTorch 2.10). The key changes from upstream:

- **Replaced Flash Attention 3** with PyTorch's built-in `scaled_dot_product_attention` (FA3/kernels package targets Hopper/desktop Ampere and doesn't build on Jetson's SM 8.7).
- **Removed `torch.compile`** — Triton is not available on aarch64, so the inductor backend fails. Model and optimizer run in eager mode.
- **Removed `kernels` dependency** and the pinned torch CUDA index from `pyproject.toml` (Jetson uses its own JetPack-provided PyTorch).
- **Tuned hyperparameters** for the Orin's ~5–24 TFLOPS BF16 throughput (size-dependent) and 30 GB unified memory.

### Setup on Jetson

```bash
# Install deps (torch is already provided by JetPack)
pip3 install rustbpe tiktoken pyarrow requests numpy pandas matplotlib

# Clone and prep
git clone <repo-url> && cd autoresearch
python3 prepare.py    # download data + train tokenizer
python3 train.py      # baseline run (~5 min)
```

### Hyperparameter sweep results

All runs use the fixed 5-minute time budget on a Jetson AGX Orin 32GB with `MAX_SEQ_LEN=512`, `HEAD_DIM=64`, `WINDOW_PATTERN="L"`:

| DEPTH | DEVICE_BATCH_SIZE | TOTAL_BATCH_SIZE | val_bpb | Steps | Params | VRAM |
|-------|-------------------|------------------|---------|-------|--------|------|
| 4 | 16 | 2^15 | 1.488 | 872 | 11.5M | 1.2 GB |
| 8 | 32 | 2^17 | 1.519 | 95 | 50.3M | 6.2 GB |
| 6 | 32 | 2^17 | 1.419 | 147 | 26.3M | 4.4 GB |
| 6 | 32 | 2^16 | 1.357 | 279 | 26.3M | 4.4 GB |
| 6 | 32 | 2^15 | 1.341 | 535 | 26.3M | 4.4 GB |
| **6** | **32** | **2^14** | **1.338** | **1018** | **26.3M** | **4.3 GB** |

The best configuration is **DEPTH=6, DEVICE_BATCH_SIZE=32, TOTAL_BATCH_SIZE=2^14** (the current defaults in this fork).

---

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

Each experiment directory includes a `.claude/` folder with a `PreToolUse` hook (`cage.sh`) that prevents the agent from reading or writing files outside its own directory. This ensures experiments stay isolated and agents can't accidentally clobber each other's files.

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

```bash
claude --dangerously-skip-permissions -p "Read program.md and start autoresearch."
```

## Platform support

This code currently requires that you have a single NVIDIA GPU. In principle it is quite possible to support CPU, MPS and other platforms but this would also bloat the code. People can reference (or have their agents reference) the full/parent nanochat repository that has wider platform support and shows the various solutions (e.g. a Flash Attention 3 kernels fallback implementation, generic device support, autodetection, etc.), feel free to create forks or discussions for other platforms and I'm happy to link to them here in the README in some new notable forks section or etc.

For running on smaller compute platforms (Macbooks etc.), see the ANE section above, or try one of the notable forks below.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch concept and nanochat
- [ArmanJR-Lab](https://github.com/ArmanJR-Lab/autoautoresearch) — Go director framework and mad-scientist experiments
- [ncdrone](https://github.com/ncdrone/autoresearch-ANE) — ANE native training, MLX port, multi-agent gossip
- [trevin-creator](https://github.com/trevin-creator) — [MLX port](https://github.com/trevin-creator/autoresearch-mlx) that `mlx/` is based on
- [miolini](https://github.com/miolini) — [MPS/macOS port](https://github.com/miolini/autoresearch-macos)
- [maderix](https://github.com/maderix) — [ANE private API](https://github.com/maderix/ANE) reverse engineering
- [Apple MLX team](https://github.com/ml-explore/mlx)

## License

MIT
