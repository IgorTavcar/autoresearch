# MLX Training

Apple Silicon GPU training via [MLX](https://github.com/ml-explore/mlx). Ported from [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx).

Replaces the PyTorch/MPS approach. MLX is Apple's native ML framework — native bf16, unified memory, no translation layer.

## Setup

```bash
cd mlx
uv sync
uv run prepare.py --num-shards 8
uv run train.py
```

## Agent mode

Run from a separate clone (agent needs `train.py` at repo root):

```bash
cd ~/Dev/autoresearch-mlx
claude --dangerously-skip-permissions -p "Read program.md and start autoresearch."
```

## Key differences from MPS

- **bf16 works natively** (MPS bf16 was 2.6x slower)
- **Larger models feasible** — depth=4 baseline at ~800ms/step (vs MPS 764ms with smaller model)
- Architecture: value embeddings, residual lambdas, logit softcapping, QK norm, ReluSquared
- Separate LR groups: embedding, unembedding, matrix, scalar
- Uses rustbpe tokenizer (vocab=8192) instead of tiktoken

## Baseline (M4 Max 128GB)

| Metric | Value |
|--------|-------|
| val_bpb | 1.665 |
| depth | 4 |
| params | ~50M |
| ms/step | ~800 |
| tok/sec | ~80K |
