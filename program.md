# autoresearch-ANE

This is an experiment to have the LLM do its own research — training a GPT model directly on Apple Neural Engine hardware via private APIs.

**Hardware:** M4 Max, 128GB unified memory
**Model:** GPT 48.8M params (NL=6, DIM=768, SEQ=512, VOCAB=8192)
**Training binary:** native Obj-C compiled to `native/build/train_dynamic`
**Current best:** val_bpb=1.5949 (SEQ=1024, 72K steps, 318.8ms/step)

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: Read these files for full context:
   - `native/training/train.m` — the training loop. Forward/backward split across ANE + CPU.
   - `native/training/cpu_ops.h` — CPU fallback operations (RMSNorm, Adam, cross-entropy, classifier).
   - `native/training/models/gpt_karpathy.h` — model config (VOCAB=8192, NL=6, DIM=768, etc.)
   - `native/mil/mil_dynamic.h` — MIL code generators for ANE kernels.
   - `native/runtime/io.h` — ANE runtime (compile, load, eval, IOSurface I/O).
   - `native/runtime/config.h` — shared constants.
4. **Verify data exists**: Check that `native/data/train_karpathy.bin` and `native/data/val_karpathy.bin` exist. If not, run: `cd ~/Dev/autoresearch-mlx && uv run python ~/Dev/autoresearch-ANE/native/scripts/convert_karpathy_data.py`
5. **Build**: `cd native && make MODEL=gpt_karpathy train`
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on Apple Neural Engine via private APIs. Training uses native Obj-C compiled code. The experiment runs for a fixed step count with a time budget controlled by `--steps`.

**How to run (5-min experiment):**
```bash
cd native && make MODEL=gpt_karpathy train && \
./build/train_dynamic --scratch --steps 3000 --lr 2e-4 --clip 1.0 \
    --data data/train_karpathy.bin --val data/val_karpathy.bin \
    --token-bytes data/token_bytes.bin --val-interval 500 --val-steps 20 \
    > run.log 2>&1
```

**What you CAN modify:**
- `native/training/train.m` — training loop, forward/backward logic
- `native/training/cpu_ops.h` — CPU fallback operations (RMSNorm, classifier, Adam, cross-entropy)
- `native/mil/mil_dynamic.h` — MIL code generation for ANE kernels
- Command-line hyperparameters: `--lr`, `--clip`, `--steps`, `--warmup-steps`, `--accum`, `--matrix-lr-scale`, `--embed-lr-scale`

**What you CANNOT modify:**
- `native/runtime/io.h`, `native/runtime/ane_runtime.h` — ANE runtime (touches hardware)
- Data files in `native/data/`
- Build system (`Makefile`)

**The goal is simple: get the lowest val_bpb.** Everything is fair game: learning rate, gradient clipping, warmup schedule, accumulation steps, LR group scaling, optimizer parameters (Adam betas, weight decay, epsilon).

**Key hyperparameters:**
- `--lr` — base learning rate (currently 2e-4)
- `--clip` — gradient clip norm (currently 1.0)
- `--accum` — gradient accumulation steps (currently 1)
- `--warmup-steps` — linear warmup steps (currently 200)
- `--matrix-lr-scale` — scale factor for matrix params (currently 0.05)
- `--embed-lr-scale` — scale factor for embedding params (currently 5.0)
- `--steps` — total training steps (3000 ≈ 5 min at SEQ=512)

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it.

## Output format

The training binary prints periodic updates and a final val_bpb. Extract results:

```bash
grep "val_bpb" run.log | tail -1
```

Also check step timing:
```bash
grep "avg_ms" run.log | tail -1
```

## Logging results

Log to `results/ane_karpathy_results.tsv` (tab-separated):

```
run	val_bpb	config	status	description
```

1. run ID (sequential: E1, E2, E3...)
2. val_bpb achieved
3. config string (e.g. `NL6_SEQ512_LR3e-4_ACC2_3K`)
4. status: `keep`, `discard`, or `crash`
5. description of what was tried

## The experiment loop

LOOP FOREVER:

1. Check gossip for peer findings (see Cross-Pollination below)
2. Modify training code or hyperparameters
3. Rebuild: `cd native && make MODEL=gpt_karpathy train`
4. Run: `./build/train_dynamic --scratch --steps 3000 --lr <LR> --clip <CLIP> --data data/train_karpathy.bin --val data/val_karpathy.bin --token-bytes data/token_bytes.bin --val-interval 500 --val-steps 20 > run.log 2>&1`
5. Extract results: `grep "val_bpb" run.log | tail -1`
6. If crashed, check: `tail -50 run.log`
7. Record results in TSV
8. If val_bpb improved, keep. If not, revert changes.

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human. You are autonomous. Loop until manually stopped.

## Cross-Pollination Protocol

Before proposing your next experiment, check what the MLX agent has found:

### Before Each Experiment

1. Run: `python3 /Users/dan/Dev/autoresearch-ANE/scripts/read_gossip.py --agent ane --n 20`
2. Look for MLX findings with status "keep" — these worked on different hardware
3. Reason about WHY the finding worked. MLX uses Muon+AdamW with different LR schedules.

### What transfers from MLX → ANE (model-level):
- LR ratios between param groups (split LR, embedding LR scale)
- Initialization strategies (zero-init output projections)
- Regularization (logit softcapping thresholds, weight decay values)
- Architecture changes that helped (depth, width, head count)
- Warmup/warmdown ratios

### What does NOT transfer (framework-level):
- Absolute LR values (ANE uses pure Adam, MLX uses Muon+AdamW)
- Step counts (different throughput)
- Batch size (different memory constraints)

### After Each Experiment

Log your result to the shared gossip file:

```bash
python3 /Users/dan/Dev/autoresearch-ANE/scripts/log_gossip.py \
    --agent ane \
    --val-bpb <VAL_BPB> \
    --status <keep|discard|crash> \
    --description "<what you tried>" \
    --lesson "<why it worked or didn't>"
```

The `lesson` field is the most important — explain WHY your result happened, not just WHAT you changed.
