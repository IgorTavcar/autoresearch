# Autoresearch Ecosystem Research

> Deep analysis of the most promising forks, extensions, and derivative projects built on
> [karpathy/autoresearch](https://github.com/karpathy/autoresearch).
>
> **Date:** March 27, 2026 | **Upstream:** 58.3k stars, 8k+ forks | **Created:** March 6, 2026

---

## Table of Contents

- [The Core Pattern](#the-core-pattern)
- [Part 1: Fork Leaderboard](#part-1-fork-leaderboard)
  - [Best val_bpb Results](#best-val_bpb-results)
  - [Most Promising Forks (Detailed)](#most-promising-forks-detailed)
  - [Hardware Accessibility Ports](#hardware-accessibility-ports)
- [Part 2: Ecosystem Beyond Forks](#part-2-ecosystem-beyond-forks)
  - [Production Deployments](#production-deployments)
  - [Universal / Generalized Tools](#universal--generalized-tools)
  - [Domain-Specific Applications](#domain-specific-applications)
  - [Agent Self-Improvement](#agent-self-improvement)
  - [Swarm & Collaborative Infrastructure](#swarm--collaborative-infrastructure)
  - [Full Research Pipelines (Idea-to-Paper)](#full-research-pipelines-idea-to-paper)
  - [GPU Kernel Optimization](#gpu-kernel-optimization)
- [Part 3: Academic & Formal Work](#part-3-academic--formal-work)
- [Part 4: HuggingFace Ecosystem](#part-4-huggingface-ecosystem)
  - [Models](#models)
  - [Datasets](#datasets)
  - [Spaces](#spaces)
- [Part 5: Discovery Hubs](#part-5-discovery-hubs)
- [Grok Report Review](#grok-report-review)
- [Key Takeaways](#key-takeaways)

---

## The Core Pattern

Karpathy's autoresearch distills to a ~630-line pattern:

```
Fixed eval harness + mutable train.py + 5-min ratchet loop + git keep/revert
```

1. Define a **scalar metric** (val_bpb, latency, score, etc.)
2. **Constrain scope** to a single mutable file
3. Let an AI agent **loop**: modify -> evaluate -> keep if improved, revert if not
4. Repeat overnight. Wake up to stacked, git-verified improvements.

The original targeted single-GPU nanochat training on H100s. The ecosystem has since
removed both constraints: it works on any hardware, for any measurable goal.

---

## Part 1: Fork Leaderboard

### Best val_bpb Results

| Fork | Best val_bpb | Improvement | Hardware | Method |
|------|-------------|-------------|----------|--------|
| [m0at/autoresearch-rustbrain](https://github.com/m0at/autoresearch-rustbrain) | **0.8673** | -12.6% | 4x H100 SXM | Rust/CUDA rewrite, bin-packing, depth=30, 119.5M params |
| [novix-science/autoresearch](https://github.com/novix-science/autoresearch) | **0.977** | -1.5% | 8x H100 | 8-agent swarm, 2,430 experiments, ~240 GPU-hours |
| karpathy/autoresearch (baseline) | 0.992 | -- | 1x H100 | Reference implementation |
| [thenamangoyal/autoresearch](https://github.com/thenamangoyal/autoresearch) (MLX) | 1.295 | N/A (Apple) | M4 Max | Apple MLX native, overnight agent runs |

### Most Promising Forks (Detailed)

#### 1. Rustbrain -- Rust/CUDA Training Engine

**[m0at/autoresearch-rustbrain](https://github.com/m0at/autoresearch-rustbrain)** (3 stars -- massively underrated)

The most technically impressive fork. Rebuilds the entire training engine in Rust + CUDA
with hand-written kernels. No Python in the training loop, no PyTorch, no autograd.

- **Best-fit bin-packing** of training sequences: 100% token utilization vs ~60% sequential
- **Architecture:** depth=30 (119.5M params), d_model=512, SSSSL attention pattern (4 local-window + 1 full-context, repeating), window=256, MLP 4x, RoPE base 200k
- **Optimizer:** Muon (momentum warmup 0.85->0.95, peak_lr=0.04), AdamW for embeddings (lr=0.9)
- **Neuron rinsing (experimental):** dynamic layer importance scoring via EMA(activation_norm x gradient_norm x generalization_ratio), periodic MLP reinitialization of low-signal layers
- **Result:** val_bpb 0.8673 -- a 0.125 improvement over reference
- No published model weights yet

#### 2. 8-Agent Swarm

**[novix-science/autoresearch](https://github.com/novix-science/autoresearch)** (8 stars)

Orchestrated 8 autonomous agents across 8 H100 GPUs using ClawTeam coordination.

- **2,430+ optimization trials** over ~30 hours per GPU (~240 GPU-hours total)
- Team composition: initially 4 Claude Code + 4 Codex, later all-Claude
- Phases: architecture discovery -> deep tuning -> cross-pollination -> fine-tuning
- **Discovered:** depth=12 > 8, norm-before-RoPE, query scaling (q *= 1.25), fixed sliding window=512, RoPE base=60k
- **Proved negative results:** MoE, grouped query attention, and ALiBi all degraded performance at this scale
- **Result:** val_bpb 1.044 -> 0.977 (6.4% reduction)

#### 3. TTT-RL-Discover -- RL Meta-Learner

**[ar0cket1/test-time-rl-discover-autoresearch](https://github.com/ar0cket1/test-time-rl-discover-autoresearch)** (6 stars)

The most architecturally novel fork. Uses reinforcement learning to meta-learn which code
modifications improve training.

- **Outer loop (RL):** 12 optimization steps x 16 rollouts = 192 evaluations
- **Inner loop:** each rollout runs a real 5-min autoresearch training job
- **Reward:** `1 / (1e-8 + val_bpb)` -- lower bpb = higher reward; failures get zero
- **CPU-side preflight:** patch parsing, AST compilation, batch-divisibility, metric-format checks before expensive GPU eval
- Supports **Kimi K2.5** and **GPT-OSS-120B** as outer-loop models
- **Hyperbolic cloud integration** for on-demand H100 nodes with detached launch + resume
- ~17.5 GPU-hours across 8x H100, completing in 2.5-3.5 wall-clock hours

#### 4. Darwin Derby -- Generalized Optimization Framework

**[kousun12/darwin-derby](https://github.com/kousun12/darwin-derby)** (46 stars)

Extends autoresearch from nanochat training to a universal optimization platform.

- Agents modify files in `state/`, framework executes hidden `scoring/score.py`
- **Blind scoring:** agents never see evaluation code, preventing metric overfitting
- **"Squishy targets":** even subjective artifacts (essay quality, landing page copy) can be optimized via LLM-as-judge
- Reference problems: Rastrigin 10D, TSP 20-city, rectangle packing, Fibonacci speed, GPT training
- Supports local single-machine loops and distributed git-based swarming

#### 5. Autoresearch 2.0 -- Optuna + Multilingual

**[soveshmohapatra/autoresearch-2.0](https://github.com/soveshmohapatra/autoresearch-2.0)** (6 stars)

Adds Bayesian hyperparameter optimization and multilingual training.

- **Optuna integration:** TPE sampler + MedianPruner for early stopping of bad trials
- Persistent `optuna_study.db` for resumable searches across sessions
- **10 languages:** English, French, Spanish, German, Hindi, Japanese, Gujarati, Dutch, Odia, Chinese
- Terminal GUI with live loss sparklines, progress bars, real-time monitoring
- SwiGLU, prenorm, weight tying, GQA, MoE as toggleable experiment flags

#### 6. HF-Autoresearch -- HuggingFace Infrastructure

**[mishig25/hf-autoresearch](https://github.com/mishig25/hf-autoresearch)** (0 stars, created March 27, 2026 -- today)

By a HuggingFace engineer. Runs autoresearch on HF's managed compute.

- **HF Jobs** for A100/H200 GPU access, volume mounts for datasets, storage buckets for checkpoints
- **`hf papers` integration:** agents can search and read recent academic papers during iteration
- Only 2 files: `train.py` + `program.md` -- minimal, clean adaptation
- Very early stage but high potential given infrastructure backing

#### 7. Modal Cloud Backend

**[davidtsong/autoresearch-modal](https://github.com/davidtsong/autoresearch-modal)** (4 stars, active March 23)

No-local-GPU autoresearch via Modal's cloud infrastructure.

- Local machine acts as orchestrator only; training runs on Modal H100s
- `modal_control.py` provides start/status/logs/result/stop commands
- Persistent volume caches training data remotely
- ~12 experiments/hour, ~100 experiments overnight
- Fixed 5-min time budget ensures fair comparison across experiments

### Hardware Accessibility Ports

These forks democratized access beyond H100s:

| Fork | Stars | Target Hardware | Key Feature |
|------|-------|-----------------|-------------|
| [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) | 1,663 | macOS / MPS | Canonical macOS port |
| [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) | 407 | Windows + consumer NVIDIA | RTX 2060-4090 support, VRAM profiling |
| [sanbuphy/autoresearch-cn](https://github.com/sanbuphy/autoresearch-cn) | 158 | Chinese localization | Full Chinese docs + skill abstraction |
| [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) | 1,100 | Apple Silicon / MLX | No PyTorch, native unified memory, ~96x slower than H100 but zero cloud cost |
| [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) | 22 | AMD ROCm | AMD GPU support |
| [lucasgelfond/autoresearch-webgpu](https://github.com/lucasgelfond/autoresearch-webgpu) | 40 | Browser / WebGPU | Train in-browser, zero setup |
| [bro4all/autoresearch-tenstorrent](https://github.com/bro4all/autoresearch-tenstorrent) | 6 | Tenstorrent TT-XLA | Alternative AI accelerator |

---

## Part 2: Ecosystem Beyond Forks

### Production Deployments

#### Shopify -- Liquid Templating Engine

The highest-profile production use. Tobi Lutke (Shopify CEO) ran autoresearch via
[davebcn87/pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) on Shopify's
20-year-old Liquid engine (GitHub PR Shopify/liquid#2056).

- 120 experiments, 93 commits overnight
- **+53% faster parse+render, -61% memory allocations**
- Biggest single win: replacing a regex-based tokenizer with byte-index searching
- All 974 unit tests passed with zero regressions
- Also optimized an internal 0.8B query-expansion model: 37 experiments, 19% validation improvement

*Source: Simon Willison's coverage, Fortune, VentureBeat*

#### SkyPilot -- Multi-GPU Parallel Autoresearch

Extended the pattern to 16 parallel GPU clusters managed by Claude Code.

- ~910 experiments in 8 hours (9x wall-clock speedup vs sequential)
- Agent independently discovered two-tier validation: screening on cheaper H100s, confirming on H200s
- Ran factorial grids of 10-13 experiments per wave rather than greedy hill-climbing

*Source: blog.skypilot.co/scaling-autoresearch*

### Universal / Generalized Tools

These turn the loop into a reusable skill for anything with a scalar metric.

| Project | Stars | Approach |
|---------|-------|----------|
| [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) | 2,538 | Claude Code skill. Multi-mode: code, prompts, agents, security (STRIDE+OWASP), docs. Persistent memory, confidence scoring, auto-hypothesis generation. Install + write `program.md` + sleep. |
| [davebcn87/pi-autoresearch](https://github.com/davebcn87/pi-autoresearch) | 3,018 | Pi agent extension. JSONL logging, live dashboard, confidence thresholds. The tool Shopify used. Includes the exact `program.md` Tobi ran. |
| [leo-lilinxiao/codex-autoresearch](https://github.com/leo-lilinxiao/codex-autoresearch) | 776 | Codex-native skill with resume support and parallel experiments |
| [greyhaven-ai/autocontext](https://github.com/greyhaven-ai/autocontext) | 671 | Recursive self-improving agent harness. Not domain-specific; recursively improves agent context and instructions. |
| [jmilinovich/goal-md](https://github.com/jmilinovich/goal-md) | 100 | GOAL.md pattern: agent must first construct its own fitness function, then optimize. Generalizes autoresearch to "domains with constructed metrics." |
| [krzysztofdudek/ResearcherSkill](https://github.com/krzysztofdudek/ResearcherSkill) | 125 | One-file skill. Drop in, agent becomes a scientist. 30+ experiments while you sleep. |
| [jung-wan-kim/autoresearch-builder](https://github.com/jung-wan-kim/autoresearch-builder) | 21 | Scaffold generator for ML/Web/Flutter/Java. Multi-platform software dev. |

### Domain-Specific Applications

#### Finance & Trading

| Project | Stars | What It Does |
|---------|-------|-------------|
| [chrisworsey55/atlas-gic](https://github.com/chrisworsey55/atlas-gic) | 1,218 | 25 trading agents, Darwinian selection. Rolling Sharpe ratio as loss. Top quartile agents get weight x1.05 daily, bottom x0.95. |
| [CarloNicolini/autoresearch-skfolio](https://github.com/CarloNicolini/autoresearch-skfolio) | -- | Portfolio optimization via skfolio. Tests MeanRisk, HierarchicalRiskParity, NestedClustersOptimization. |
| [sakchhams/strategy-dev](https://github.com/sakchhams/strategy-dev) | 4 | Iterative trading strategy development template |

#### Voice AI

[ArchishmanSengupta/autovoiceevals](https://github.com/ArchishmanSengupta/autovoiceevals) (122 stars) --
Generates adversarial callers to attack voice agents, proposes prompt improvements one at
a time, keeps what improves score. Works with Vapi, Smallest AI, ElevenLabs ConvAI.
~$0.90/experiment, 2-4 minutes each.

#### Adversarial ML / AI Safety

[romovpa/claudini](https://github.com/romovpa/claudini) (63 stars) --
Autoresearch for LLM adversarial attacks. Autonomously discovers and improves attacks
against language models using the keep/discard loop.

#### Algorithm Optimization

[Rkcr7/autoresearch-sudoku](https://github.com/Rkcr7/autoresearch-sudoku) (3 stars, remarkable result) --
AI agent autonomously built the **fastest sudoku solver** on 4/6 standard benchmarks.
Beat Tdoku (#1 since 2019) by 49% and rust_sudoku (#2) by 82%. 312 experiments, 65,275x
speedup, 709 lines of Rust, zero human-written solver code.

#### Other Domains

| Domain | Project | Notes |
|--------|---------|-------|
| Medical imaging | mattlungrenmd/autoresearch-medimage | Autonomous radiology/pathology model optimization |
| Robotics | jellyheadandrew/autoresearch-robotics | Simulation feedback as eval loop |
| Information retrieval | carlaiau/ir-autoresearch | Search ranking on WSJ/TREC benchmarks |
| Physics (DFT) | richafltr/dft-auto-research | Density functional theory simulations |
| Genealogy | [mattprusak/autoresearch-genealogy](https://github.com/mattprusak/autoresearch-genealogy) (936 stars) | Structured prompts + vault templates for family history |
| Hyperparameter tuning | zxh0916/auto-hparam-tuning | Hydra-native, low-invasion HPO skill |
| AutoML benchmark | [ferreirafabio/autoresearch-automl](https://github.com/ferreirafabio/autoresearch-automl) (16 stars) | Compares LLM-based optimization vs classical HPO |
| Sports analytics | oldedb/NCAAB-Model-Tuner | College basketball score prediction |
| Travel planning | michaelpersonal/trip-optimizer | Non-technical: iteratively improves itineraries |
| Marketing/SEO | (multiple blog posts) | Balu Kosuri: doc SEO 24/40 -> 40/40 in 14 cycles. Eric Siu: 36.5k experiments/year vs typical 30. |
| MLP architecture | [HuangShengZeBlueSky/MLP_AutoResearch](https://github.com/HuangShengZeBlueSky/MLP_AutoResearch) (30 stars) | Autonomous MLP model optimization |
| Tabular GLM | [ajzhanghk/autoresearch-glm](https://github.com/ajzhanghk/autoresearch-glm) (7 stars) | Autonomous feature discovery for tabular GLM models |

### Agent Self-Improvement

| Project | Stars | Approach |
|---------|-------|----------|
| [hwchase17/autoresearch-agents](https://github.com/hwchase17/autoresearch-agents) | 123 | By Harrison Chase (LangChain founder). Swaps ML training for agent optimization: editable `agent.py`, fixed `run_eval.py` + `dataset.json`, metric = LangSmith score. Supports LangChain, LangGraph, Anthropic SDK, plain Python. |
| [cavit99/autoresearch-autoresearch](https://github.com/cavit99/autoresearch-autoresearch) | 41 | Meta-repo that distills portable patterns for bounded agent-verifier research loops |
| [hgarud/autoresearch](https://github.com/hgarud/autoresearch) | 10 | Adds lineage tree visualizer and evolutionary database for experiment genealogy |

### Swarm & Collaborative Infrastructure

| Project | Stars | Approach |
|---------|-------|----------|
| [mutable-state-inc/autoresearch-at-home](https://github.com/mutable-state-inc/autoresearch-at-home) | 452 | SETI@home-style: distributed experiment claiming, shared best-config syncing, multi-agent coordination across consumer GPUs |
| [hyperspaceai/agi](https://github.com/hyperspaceai/agi) | 1,213 | Distributed AGI system. Thousands of agents collaboratively train models via P2P gossip |
| [glitch-rabin/swarma](https://github.com/glitch-rabin/swarma) | 136 | QMD + Karpathy loop for AI agent swarm coordination |
| [Human-Agent-Society/CORAL](https://github.com/Human-Agent-Society/CORAL) | 81 | Multi-agent evolution organization |
| [ygivenx/agenthub](https://github.com/ygivenx/agenthub) | 121 | "GitHub for agents." No main branch; DAG of commits; agent-native coordination. First use-case: scaling autoresearch swarms. |

### Full Research Pipelines (Idea-to-Paper)

| Project | Stars | Scope |
|---------|-------|-------|
| [aiming-lab/AutoResearchClaw](https://github.com/aiming-lab/AutoResearchClaw) | 9,097 | "Chat an Idea. Get a Paper." End-to-end: topic -> literature review -> experiments -> analysis -> peer review -> paper |
| [Sibyl-Research-Team/AutoResearch-SibylSystem](https://github.com/Sibyl-Research-Team/AutoResearch-SibylSystem) | 199 | Self-evolving research system on Claude Code with multi-module autonomy |
| [OpenRaiser/NanoResearch](https://github.com/OpenRaiser/NanoResearch) | 272 | Plans experiments, generates code, runs SLURM jobs, analyzes results, writes papers |
| [TenureAI/PhD-Zero](https://github.com/TenureAI/PhD-Zero) | 46 | PhD-level workflows: literature review, hypothesis generation, multi-step methodology |

### GPU Kernel Optimization

**[RightNow-AI/autokernel](https://github.com/RightNow-AI/autokernel)** (854 stars) --
"Give it any PyTorch model, go to sleep, wake up to optimized Triton kernels."

- Profiles PyTorch bottleneck kernels via Amdahl's Law
- Extracts them as standalone optimization targets
- Runs autoresearch loop: modify kernel -> benchmark (5-stage correctness) -> keep/revert
- ~40 experiments/hour (~320 overnight)
- 9 kernel types: matmul, softmax, layernorm, flash_attention, etc.
- Supports NVIDIA H100/A100/RTX 4090 and AMD ROCm
- Exports optimized kernels to HuggingFace

---

## Part 3: Academic & Formal Work

### AutoResearch-RL (arXiv 2603.07300)

Formalizes the autoresearch loop as a **Markov Decision Process** with PPO policy over
code edits. Authors from Yale, Google Cloud, Stanford, UC Berkeley, MIT, Meta, IIT
Bombay, and DeepMind.

- RL agent proposes code modifications to `train.py`
- Executes under fixed time budget, receives scalar reward from val_bpb
- Self-evaluation module aborts unpromising runs early (2.4x throughput recovery)
- Matches or exceeds hand-tuned baselines after ~300 overnight iterations
- Also on [HuggingFace Papers](https://huggingface.co/papers/2603.07300)

### Bilevel Autoresearch (arXiv 2603.23420)

Meta-optimization: an **outer autoresearch loop optimizes the inner loop itself**.

- Outer loop generates and injects new search mechanisms as Python code at runtime
- Discovers strategies from combinatorial optimization, multi-armed bandits, and
  experimental design -- all without human specification
- **5x improvement** over standard inner loop alone (-0.045 vs -0.009 val_bpb improvement)
- Both levels use the same LLM -- no more powerful model required at meta level

### AutoResearch-RL for Post-Training

[vivekvkashyap/autoresearch-rl](https://github.com/vivekvkashyap/autoresearch-rl) (77 stars) --
Built on Prime Intellect's prime-rl. Ran qwen2.5-0.5b-instruct on GSM8K across 60+
autonomous experiments. Eval score 0.475 -> 0.550 in fewer training steps (20 vs 30) --
less compute, better results.

### Dolphin: Closed-loop Open-ended Auto-research (arXiv 2501.03916)

Closed-loop open-ended auto-research through thinking, practice, and feedback. Found in
the [metanerd/auto-research HuggingFace collection](https://huggingface.co/collections/metanerd/auto-research).

### Can LLMs Beat Classical HPO?

[ferreirafabio/autoresearch-automl](https://github.com/ferreirafabio/autoresearch-automl) (16 stars) --
Benchmark comparing autoresearch-style LLM optimization against classical hyperparameter
optimization methods (Bayesian, random search, etc.).

---

## Part 4: HuggingFace Ecosystem

### Models

**226 nanochat models** on HuggingFace Hub as of March 27, 2026.

Official Karpathy models:
- **[karpathy/nanochat-d32](https://huggingface.co/karpathy/nanochat-d32)** -- Base LM, MIT license
- **[karpathy/nanochat-d34](https://huggingface.co/karpathy/nanochat-d34)** -- 2.2B params, 34 layers. Trained ~100h on 8xH100 (~$2,500), 88.7B tokens, 40:1 token-to-param ratio (2x Chinchilla). val_bpb: 0.7045, CORE Score: 0.3382. MIT license.

Notable community models:
- eastlondoner/nanochat-wasm-fused-preview-02 -- WebAssembly-optimized
- fffoivos/nanochat-en-vs-el-353m -- Multilingual English vs. Greek (353M params)
- ChrisMcCormick/nanochat-varlen-d24 -- Variable-length sequence variant
- Solshine/nanochat-d32-sae-layer16-topk32 -- Sparse autoencoder experiment
- chicajas/nanochat-sft-d24-483 -- Supervised fine-tuning variant
- ramen-noodels/autoresearch-diffusion-mar24 -- Diffusion model (no model card yet)

Nanochat is integrated into the **HuggingFace Transformers** library with
[official documentation](https://huggingface.co/docs/transformers/main/model_doc/nanochat).

### Datasets

| Dataset | Rows | Description |
|---------|------|-------------|
| [karpathy/climbmix-400b-shuffle](https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle) | -- | Official training data used by `prepare.py` |
| [davegraham/autoresearch-experiments](https://huggingface.co/datasets/davegraham/autoresearch-experiments) | 2,637 | Cross-platform experiments across 5 GPU types (M5 Max, RTX 4000 Ada, A100 40GB, RTX Pro 6000 Blackwell, MI300X). **Key finding: Sonnet 4.6 found 8x more improvements than Sonnet 4.0.** CC-BY-4.0. |
| [lewtun/autoresearch](https://huggingface.co/datasets/lewtun/autoresearch) | -- | By Lewis Tunstall (HF staff). Framework/dataset implementing the autoresearch methodology. |
| [Creekside/lambda-text-autoresearch](https://huggingface.co/datasets/Creekside/lambda-text-autoresearch) | 49.9k | Training text data for autoresearch experiments |
| reasoning-degeneration-dev/adaevolve-autoresearch-run1 | 50 | AdaEvolve variant experiment logs |

**Multilingual datasets** by Sovesh (for autoresearch-2.0):

| Language | Dataset | Rows |
|----------|---------|------|
| Dutch | Sovesh/autoresearch-nl | 2.13M |
| French | Sovesh/autoresearch-fr | 1.9M |
| German | Sovesh/autoresearch-de | 1.42M |
| Spanish | Sovesh/autoresearch-es | 1.42M |
| Japanese | Sovesh/autoresearch-ja | 1.38M |
| Chinese | Sovesh/autoresearch-zh | 1.33M |
| Hindi | Sovesh/autoresearch-hi | 160k |
| Gujarati | Sovesh/autoresearch-gu | 30.4k |
| Odia | Sovesh/autoresearch-or | 17.3k |

### Spaces

| Space | Author | Description |
|-------|--------|-------------|
| [abidlabs/autoresearch](https://huggingface.co/spaces/abidlabs/autoresearch) | Abubakar Abid (Gradio creator, HF) | Interactive tracking dashboard |
| [burtenshaw/autoresearch_env](https://huggingface.co/spaces/burtenshaw/autoresearch_env) | Ben Burtenshaw | Hosted environment server for running experiments |
| Xkarp/AutoResearchClawDemo | -- | Running demo of AutoResearchClaw |
| Vignesh-1918/AutoResearcher-Multi-Agent | -- | Multi-agent research system |
| samihalawa/autoresearch-langgraph | -- | LangGraph + Gemini + Google Search research agent |

---

## Part 5: Discovery Hubs

| Hub | Stars | Focus |
|-----|-------|-------|
| [alvinunreal/awesome-autoresearch](https://github.com/alvinunreal/awesome-autoresearch) | 795 | Highest-signal index. 59+ projects across categories: general-purpose, research agents, platform ports, domain-specific, evaluation benchmarks. |
| [WecoAI/awesome-autoresearch](https://github.com/WecoAI/awesome-autoresearch) | 317 | Use-case focused (marketing, infra, kernels, RL, peptide design) by the AIDE team. |
| [WecoAI/aideml](https://github.com/WecoAI/aideml) | 1,186 | AIDE: AI-Driven Exploration. ML engineering agent that automates R&D -- closely aligned with autoresearch philosophy. |

---

## Grok Report Review

A Grok-generated REPORT.md (March 27, 2026) was reviewed against verified GitHub API
data. Key corrections:

| Claim | Grok Says | Verified |
|-------|-----------|----------|
| Upstream stars | "42k+" | **58,356** (as of March 27, 2026) |
| `karpathy/agenthub` | "Karpathy's companion" | **Does not exist** (404). The actual agenthub is `ygivenx/agenthub` (121 stars), not by Karpathy. |
| `awesome-copilot/skills/autoresearch` | "GitHub's own first-class Copilot skill" | **Cannot verify existence** |
| hwchase17 agent results | "2-3x better tool-use, 40% lower latency" | **Unverified** -- repo has no description and only 123 stars |
| uditgoenka conversion lift | "56% -> 92% on landing-page copy" | **Unverified** -- plausible but no public evidence found |

**What Grok got right:** project rankings are directionally correct, Shopify production
results match multiple sources, pi-autoresearch as Shopify's tool is confirmed, overall
ecosystem taxonomy is sound. The core recommendations (generalized skills, agent
self-optimization, hardware accessibility, swarm infrastructure) align with verified data.

**What Grok missed entirely:** Rustbrain (best results in the ecosystem), novix-science
8-agent swarm, TTT-RL-Discover fork, the entire HuggingFace ecosystem (226 models, rich
datasets), academic papers (arXiv formalizations), and domain-specific projects like
autoresearch-sudoku, claudini (adversarial ML), autovoiceevals (voice AI), and
autoresearch-glm (tabular data).

---

## Key Takeaways

1. **The pattern is domain-agnostic.** The core abstraction -- define a metric, constrain
   scope, agent loops modify/eval/keep/revert -- transfers from GPU kernels to voice
   prompts to 20-year-old Ruby template engines.

2. **Rustbrain holds the best result.** At val_bpb 0.8673 (Rust+CUDA, no Python), it
   outperforms every other fork by a wide margin. The bin-packing and SSSSL attention
   innovations are worth studying regardless of language choice.

3. **The meta-level is where the leverage is.** Bilevel autoresearch (5x improvement by
   optimizing the loop itself), TTT-RL-Discover (RL over code edits), and
   hwchase17/autoresearch-agents (agents improving agents) point toward recursive
   self-improvement as the natural endgame.

4. **Shopify is the production proof point.** 53% faster rendering on a mature,
   heavily-tested codebase demonstrates the pattern works beyond research settings.

5. **HuggingFace is becoming the model registry.** 226 nanochat models, official
   Transformers integration, and HF staff actively contributing (Tunstall, Abid, Mishig)
   signal institutional adoption.

6. **Swarm infrastructure is the next frontier.** autoresearch-at-home, hyperspaceai/agi,
   and agenthub suggest the 10x scaling step is coordinating many agents across many
   machines -- turning overnight runs into overnight swarms.

7. **Small stars, big results.** Rustbrain (3 stars, best val_bpb), autoresearch-sudoku
   (3 stars, world-beating solver), and novix-science (8 stars, 2,430 experiments) show
   that star count is a poor proxy for technical merit in this ecosystem.

---

*Research conducted March 27, 2026. Star counts and project status verified via GitHub
API. Claims sourced from project READMEs, blog posts, arXiv papers, and HuggingFace Hub.
Unverifiable claims are noted as such.*
