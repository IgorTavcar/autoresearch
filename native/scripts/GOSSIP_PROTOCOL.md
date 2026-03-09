# Cross-Pollination Protocol — ANE Agent

Before proposing your next experiment, check peer findings.

## Before Each Experiment

1. Run: `python3 /Users/dan/Dev/autoresearch-ANE/scripts/read_gossip.py --agent ane --n 20`
2. Look for MLX findings with status "keep" — these worked on different hardware
3. Reason about WHY the finding worked, not just WHAT changed

### What transfers from MLX → ANE (model-level):
- LR ratios between param groups (split LR, embedding LR scale)
- Initialization strategies (zero-init, Kaiming, Xavier)
- Architecture choices (VE, attention patterns, MLP width)
- Schedule shapes (warmdown ratio, warmup ratio)
- Regularization (weight decay, softcapping thresholds)

### What does NOT transfer (framework-level):
- Absolute LR values (MLX uses Muon+AdamW, ANE uses pure Adam)
- Step counts (5-min vs overnight)
- Memory constraints (21GB MLX vs ANE IOSurface layout)
- Specific framework features (Muon optimizer is MLX-only)

## After Each Experiment

Log your result:

```bash
python3 /Users/dan/Dev/autoresearch-ANE/scripts/log_gossip.py \
    --agent ane \
    --val-bpb <VAL_BPB> \
    --status <keep|discard|crash> \
    --description "<what you tried>" \
    --lesson "<why it worked or didn't>" \
    --steps <STEPS> \
    --wall-sec <WALL_SEC>
```

The `lesson` field is the most important — it's what the other agent learns from.
