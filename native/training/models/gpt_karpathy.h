// gpt_karpathy.h — Karpathy climbmix-400B config for ANE training
// Uses rustbpe tokenizer (vocab=8192) for 1:1 comparison with MLX
// All tunable params use #ifndef to allow -D overrides for benchmarking
#pragma once

#define MODEL_NAME "GPT-karpathy"

#ifndef DIM
#define DIM 768
#endif
#ifndef HIDDEN
#define HIDDEN 2048       // 4 * DIM (ReluSquared FFN, no gate)
#endif
#ifndef HEADS
#define HEADS 6
#endif
#ifndef KV_HEADS
#define KV_HEADS 6
#endif
#define HD (DIM/HEADS)    // = 128
#define GQA_RATIO 1       // MHA: no GQA
#define Q_DIM (HEADS * HD)   // = 768 = DIM
#define KV_DIM (KV_HEADS * HD) // = 768 = DIM
#ifndef SEQ
#define SEQ 512           // ANE SRAM wall at 1024, use 512
#endif
#ifndef NLAYERS
#define NLAYERS 6         // Optimal depth at SEQ=512
#endif
#ifndef VOCAB
#define VOCAB 8192        // rustbpe tokenizer vocab size
#endif

#define CKPT_PATH "ane_karpathy_dyn_ckpt.bin"
#define DEFAULT_DATA_PATH "data/train_karpathy.bin"
