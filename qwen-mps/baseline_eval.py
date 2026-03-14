"""Evaluate the base Qwen3.5-0.8B model (no training) to get the true baseline val_bpb."""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_PREFER_METAL_DEVICE_UNIFIED_MEMORY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare import MAX_SEQ_LEN, TOKENIZER_DIR, evaluate_bpb
from train import HFTokenizer, CausalLMWrapper, build_token_bytes

MODEL_NAME = "Qwen/Qwen3.5-0.8B"

# Device
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")

# Load model
print(f"Loading {MODEL_NAME}...")
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True,
)
hf_model = hf_model.to(device)
model = CausalLMWrapper(hf_model)

tokenizer = HFTokenizer(hf_tokenizer)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# Build token_bytes if needed
token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
needs_rebuild = True
if os.path.exists(token_bytes_path):
    existing = torch.load(token_bytes_path, weights_only=True)
    if len(existing) == vocab_size:
        needs_rebuild = False
if needs_rebuild:
    build_token_bytes(hf_tokenizer, token_bytes_path)

# Evaluate
EVAL_BATCH_SIZE = 2
model.eval()
autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16)
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, EVAL_BATCH_SIZE)

peak_mem_mb = torch.mps.driver_allocated_memory() / 1024 / 1024 if device.type == "mps" else 0

print("---")
print(f"val_bpb:      {val_bpb:.6f}")
print(f"peak_vram_mb: {peak_mem_mb:.1f}")
print(f"model:        {MODEL_NAME}")
