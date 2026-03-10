# Credits & Attribution

Everything we built on, and exactly what came from where.

---

## maderix (github.com/maderix/ANE)

**Who:** Anonymous researcher. Published 3 Substack posts on reverse-engineering M4 ANE.
**Source:** https://github.com/maderix/ANE | https://maderix.substack.com

**What we ported/use from them:**
- Private API discovery: `_ANEClient`, `_ANERequest`, `_ANEIOSurfaceObject`, `_ANEInMemoryModel`
- IOSurface I/O pattern for ANE data transfer
- MIL (Machine Learning Intermediate Language) code generation approach
- Dynamic weight pipeline (Pipeline 3): weights packed into IOSurface spatial dimension, kernels compile once, weight updates are memcpy — this was maderix's invention
- CPU fallback ops structure: RMSNorm backward, classifier, cross-entropy via Accelerate.framework
- Training loop architecture: forward/backward split across ANE + CPU
- SRAM probing methodology (32MB cliff discovery)
- Build system pattern: xcrun clang with -fobjc-arc, framework linking
- The entire concept of training on ANE via private APIs without CoreML

**What we did NOT port (but they invented):**
- FP16 gradient underflow fix (256× loss scaling) — we haven't tested if we need this yet
- DeepNet residual scaling (α = 1/√(2N)) — we used a different approach (zero-init + softcap)
- 1×1 conv > matmul throughput discovery (3× faster) — identified but not yet implemented
- Mega-kernel layer fusion research (3-4× forward speedup) — not yet implemented
- Qwen3-0.6B scaling (28 layers, GQA, 416ms/step)
- Training dashboard (blessed library)
- Stories110M model config and weights

---

## Vipul Divyanshu (github.com/vipuldivyanshu92/ANEgpt)

**Who:** Contributor to maderix/ANE (PR #19). Built ANEgpt independently.
**Source:** https://github.com/vipuldivyanshu92/ANEgpt

**What we can learn from them (not yet ported):**
- ANE classifier forward as 32000-channel conv (10.2× faster than CPU cblas)
- ANE softmax over large vocab (33.8× faster than CPU vDSP)
- Fused SDPA forward kernel (QKV + SDPA + Wo in single dispatch)
- Vocab compaction (32K → 9.2K active tokens)
- Deferred cblas wait pattern (push async wait to next step)
- C-callable bridge API for Python ctypes with proper ARC memory management
- M5 ANE probing results: QoS has no effect, weight reload doesn't work, weightsBuffer doesn't override

**What they contributed to maderix/ANE (PR #19):**
- Bridge APIs (bridge/ directory)
- ANE-offloaded classifier, softmax, RMSNorm
- `train_large_ane.m` (Pipeline 2)
- Memory leak workaround for 119-compile limit

---

## thebasedcapital (github.com/thebasedcapital/ane-infer)

**Who:** Built hybrid ANE+Metal+CPU inference engine for Qwen3.5-2B in Rust + Obj-C.
**Source:** https://github.com/thebasedcapital/ane-infer | maderix/ANE issue #44

**Major discoveries we should use:**
- `doEvaluateDirectWithModel:` — bypasses ANE daemon XPC, 10% faster (trivial to add)
- Fused mega-kernels: 8.7% → 80-94% ANE utilization (single-op vs fused). 3.6 TFLOPS fused FFN
- Multi-procedure MIL: N functions in one compiled program, dispatch by `procedureIndex`
- Full `_ANEClient` API: 25+ methods documented (standard, direct, chaining, RT, session)
- `_ANEInMemoryModel` full ivar layout: 13 ivars with byte offsets
- `_perfStatsMask` must be set BEFORE eval to get hardware perf counters
- Weight blob format: 64-byte global header + per-chunk DEADBEEF headers
- IOKit H11ANE: 3 user client types (0=fail/entitlement, 1=direct path, 4=unknown)
- `MLModelConfiguration` undocumented properties: `experimentalMLE5EngineUsage`, `neuralEngineCompilerOptions`

**Chaining status:** `prepareChainingWithModel:` succeeds but `buffersReadyWithModel:` returns NO silently. Not yet working for anyone.

---

## Anemll (github.com/Anemll/Anemll + gist)

**Who:** Author of ANEMLL, production ANE inference library (1,502 stars).
**Source:** https://github.com/Anemll/Anemll | INT8 gist: https://gist.github.com/Anemll/49e219448ad350ef67ff4bfdcb9ebd8c

**Key findings:**
- INT8 W8A8 on ANE: **1.88× throughput** vs FP16 (34 TOPS vs 18 TFLOPS)
- L2 SRAM bandwidth is the bottleneck — quantize/dequantize between layers halves bandwidth
- `constexpr_affine_dequantize()` — compile-time INT8→FP16 weight conversion (zero runtime cost)
- Conv2d is required for ANE — linear layers fall back to CPU/GPU (independently confirms maderix)
- RMSNorm via LayerNorm trick: concat `[x, -x]` for zero mean, use F.layer_norm (ANE-optimized)
- `_ANEDeviceInfo` class: `.numANEs`, `.numANECores`, `.aneSubType` for hardware detection
- QoS values documented: 0=RealTime, 21=Default, 33=UserInteractive
- ANE hardware constants: 16 MAC arrays, 2048 MACs/array, 2 ops/MAC
- In-model argmax: ~8000× reduction in ANE-to-host data transfer

**Not applicable to us (inference only):**
- CoreML-based pipeline (we use private APIs)
- LUT4/LUT6 palette quantization
- iOS/macOS app, TestFlight distribution

---

## Random X User (UNVERIFIED)

**Who:** Unknown. Claims to have disassembled dyld shared cache.
**Source:** X/Twitter (no link saved — should get this)

**What they claimed:**
- `kANEFSkipPreparePhaseKey` — skips memory re-validation on load
- `kANEFEnableLateLatchKey` — defers weight binding to execution time
- `ane_skipAdapterWeightAccessCheck` — skips adapter weight validation
- Together: model load drops from 20ms to 1.6ms

**Verification status:**
- String names confirmed in ANECompilerService binary via `strings` command
- NOT found in any public repo, any documentation, or any indexed web page
- Performance claims completely unverified
- Needs probe testing before any use

---

## Andrej Karpathy (github.com/karpathy)

**Who:** AI researcher, former Tesla AI director, OpenAI founding member.
**Source:** https://github.com/karpathy/autoresearch

**What we use:**
- autoresearch framework concept (LLM agent does its own research in a loop)
- climbmix-400B training data
- rustbpe tokenizer (vocab=8192)
- `prepare.py` data pipeline (read-only, shared with MLX)
- H100 baseline val_bpb=0.998 as comparison target
- `evaluate_bpb()` evaluation function

---

## trevin-creator (github.com/trevin-creator/autoresearch-mlx)

**Who:** MLX port author.
**Source:** https://github.com/trevin-creator/autoresearch-mlx

**What we use:**
- MLX training framework (`train.py`, `program.md`)
- Muon + AdamW optimizer implementation for MLX
- MLX-specific features: VE, residual lambdas, QK norm, ReluSquared, separate LR groups
- 209 MLX experiment results (bootstrapped into gossip system)

---

## What's Truly Ours (not from any of the above)

- Zero-init output projections (Wo, W2) for activation stability
- Logit softcapping (cap=15, tanh-based) with correct backward pass
- Split learning rates: matrix_lr_scale=0.05, embed_lr_scale=5.0
- 55-experiment systematic sweep across 5 phases on ANE
- Karpathy data bridge (rustbpe vocab=8192 → ANE native binary format)
- Multi-agent gossip system (262 experiments, cross-pollination protocol)
- 3-framework comparison methodology (ANE vs MLX vs MPS, same data)
- SEQ=1024 testing, SRAM cliff analysis, quality vs throughput profiling
- Activation explosion diagnosis (3 failures → 3 fixes)
- Overnight autonomous training infrastructure (72K steps, 5 hours)
- Cosine schedule length matching insight (--steps must match actual run)
- val_bpb=1.6347 best ANE result (no one else has published ANE training quality metrics)
