"""
Autoresearch pretraining script. Single-GPU, single-file.
Continue pretraining Qwen3.5-0.8B on climbmix data.
MPS compatible (Apple Silicon).
Usage: uv run train.py
"""

import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["PYTORCH_MPS_PREFER_METAL_DEVICE_UNIFIED_MEMORY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import Adafactor

from prepare import MAX_SEQ_LEN, TIME_BUDGET, TOKENIZER_DIR, make_dataloader, evaluate_bpb

# ---------------------------------------------------------------------------
# Tokenizer wrapper (matches prepare.py's Tokenizer interface)
# ---------------------------------------------------------------------------

class HFTokenizer:
    """Wraps a HuggingFace tokenizer to match prepare.py's Tokenizer interface."""

    def __init__(self, hf_tokenizer):
        self.hf_tok = hf_tokenizer
        self.bos_token_id = hf_tokenizer.bos_token_id if hf_tokenizer.bos_token_id is not None else 0

    def get_vocab_size(self):
        return self.hf_tok.vocab_size

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if isinstance(text, str):
            ids = self.hf_tok.encode(text, add_special_tokens=False)
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.hf_tok.convert_tokens_to_ids(prepend)
                ids.insert(0, prepend_id)
            return ids
        elif isinstance(text, list):
            ids = [self.hf_tok.encode(t, add_special_tokens=False) for t in text]
            if prepend is not None:
                prepend_id = prepend if isinstance(prepend, int) else self.hf_tok.convert_tokens_to_ids(prepend)
                for row in ids:
                    row.insert(0, prepend_id)
            return ids
        raise ValueError(f"Invalid input type: {type(text)}")

    def decode(self, ids):
        return self.hf_tok.decode(ids)

# ---------------------------------------------------------------------------
# Model wrapper (matches custom GPT's forward interface)
# ---------------------------------------------------------------------------

class CausalLMWrapper(nn.Module):
    """Wraps HF CausalLM to match forward(idx, targets, reduction) interface."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, idx, targets=None, reduction='mean'):
        outputs = self.model(input_ids=idx, use_cache=False)
        logits = outputs.logits
        del outputs

        if targets is not None:
            logits_f = logits.float()
            del logits  # free fp16 copy before cross_entropy allocates
            loss = F.cross_entropy(
                logits_f.view(-1, logits_f.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=reduction,
            )
            return loss
        return logits

# ---------------------------------------------------------------------------
# Token bytes builder (for BPB evaluation)
# ---------------------------------------------------------------------------

def build_token_bytes(hf_tokenizer, save_path):
    """Build token -> byte-length lookup for BPB evaluation."""
    print("Building token_bytes lookup for HF tokenizer...")
    vocab_size = hf_tokenizer.vocab_size
    special_ids = set(hf_tokenizer.all_special_ids)
    token_bytes_list = []
    for token_id in range(vocab_size):
        if token_id in special_ids:
            token_bytes_list.append(0)
        else:
            try:
                decoded = hf_tokenizer.decode([token_id])
                token_bytes_list.append(len(decoded.encode("utf-8")))
            except Exception:
                token_bytes_list.append(0)
    tensor = torch.tensor(token_bytes_list, dtype=torch.int32)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tensor, save_path)
    print(f"Saved token_bytes ({vocab_size} entries) to {save_path}")

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

# Optimization
TOTAL_BATCH_SIZE = 2**11  # ~2K tokens per optimizer step (2 grad accum steps)
LEARNING_RATE = 4e-6      # learning rate
WEIGHT_DECAY = 0.1        # weight decay
ADAM_BETAS = (0.9, 0.95)  # Adam beta1, beta2
WARMUP_RATIO = 0.2        # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.7      # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0       # final LR as fraction of initial
MAX_GRAD_NORM = 0.5       # gradient clipping

# Training
DEVICE_BATCH_SIZE = 1     # micro batch size (2 hit 15GB on M2 — too close to ceiling)
EVAL_BATCH_SIZE = 2       # eval batch size (4 hits 18GB with 0.8B model, 8 hits 20GB)
GRADIENT_CHECKPOINTING = True  # trade compute for memory

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.set_float32_matmul_precision("high")

# Device
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: MPS (Apple Silicon)")
else:
    device = torch.device("cpu")
    print("Using device: CPU (MPS not available)")

autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16)
APPLE_FP16_PEAK_FLOPS = 14.0e12  # M3 Max estimate; adjust for your Mac

# Load pretrained model and tokenizer — float16 weights for MPS speed
print(f"Loading {MODEL_NAME}...")
hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, trust_remote_code=True,
)
if GRADIENT_CHECKPOINTING:
    hf_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
hf_model = hf_model.to(device)

# Freeze embedding and lm_head — large vocab matrices that don't need fine-tuning
for name, param in hf_model.named_parameters():
    if 'embed_tokens' in name or 'lm_head' in name:
        param.requires_grad = False
frozen = sum(1 for p in hf_model.parameters() if not p.requires_grad)
trainable = sum(1 for p in hf_model.parameters() if p.requires_grad)
print(f"Frozen {frozen} params, training {trainable} params")

model = CausalLMWrapper(hf_model)

if device.type == "mps":
    print(f"Model loaded — MPS memory: {torch.mps.driver_allocated_memory() / 1024 / 1024:.0f}MB")
tokenizer = HFTokenizer(hf_tokenizer)
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# Build token_bytes for BPB evaluation (one-time, overwrites custom tokenizer's)
token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
needs_rebuild = True
if os.path.exists(token_bytes_path):
    existing = torch.load(token_bytes_path, weights_only=True)
    if len(existing) == vocab_size:
        needs_rebuild = False
        print(f"Token bytes lookup OK ({vocab_size} entries)")
if needs_rebuild:
    build_token_bytes(hf_tokenizer, token_bytes_path)

# Parameter info
num_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {num_params:,}")
num_flops_per_token = 6 * num_params  # rough estimate

tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# optimizer = torch.optim.AdamW(
#     model.parameters(), lr=LEARNING_RATE,
#     betas=ADAM_BETAS, weight_decay=WEIGHT_DECAY, eps=1e-5,  # aggressive eps for float16
# )
# optimizer = torch.optim.SGD(
#     model.parameters(), lr=LEARNING_RATE,
#     momentum=0.9, weight_decay=WEIGHT_DECAY,
#     nesterov=True,
# )
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE,
    scale_parameter=False, relative_step=False,
    weight_decay=WEIGHT_DECAY,
)

train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
x, y, epoch = next(train_loader)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# Schedule
def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

model.train()
while True:
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()

    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            loss = model(x, y)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y, epoch = next(train_loader)

    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE * lrm

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    if device.type == "mps":
        torch.mps.empty_cache()

    train_loss_f = train_loss.item()
    if train_loss_f > 100 or train_loss_f != train_loss_f:  # NaN check
        print(f"\nFAIL — loss is {train_loss_f} at step {step}")
        exit(1)

    if device.type == "mps":
        torch.mps.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / APPLE_FP16_PEAK_FLOPS
    remaining = max(0, TIME_BUDGET - total_training_time)

    mem_mb = torch.mps.driver_allocated_memory() / 1024 / 1024 if device.type == "mps" else 0

    print(f"step {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mem: {mem_mb:.0f}MB | remaining: {remaining:.0f}s")

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 500 == 0:
        gc.collect()

    step += 1
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

print()
total_tokens = step * TOTAL_BATCH_SIZE

# Free training-only memory before eval
del optimizer, train_loader, x, y
gc.enable()
gc.collect()
if device.type == "mps":
    torch.mps.empty_cache()

# Final eval
model.eval()
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, EVAL_BATCH_SIZE)

# Final summary
t_end = time.time()
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / APPLE_FP16_PEAK_FLOPS if total_training_time > 0 else 0
peak_mem_mb = torch.mps.driver_allocated_memory() / 1024 / 1024 if device.type == "mps" else 0

print("---")
print(f"val_bpb:          {val_bpb:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_mem_mb:.1f}")
print(f"mfu_percent:      {steady_state_mfu:.2f}")
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.1f}")
print(f"model:            {MODEL_NAME}")

# Save checkpoint only if this is the best result so far
CHECKPOINT_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", "checkpoints", "best")
BEST_BPB_FILE = os.path.join(CHECKPOINT_DIR, "best_bpb.txt")

prev_best = float("inf")
if os.path.exists(BEST_BPB_FILE):
    prev_best = float(open(BEST_BPB_FILE).read().strip())

if val_bpb < prev_best:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    hf_model.save_pretrained(CHECKPOINT_DIR)
    hf_tokenizer.save_pretrained(CHECKPOINT_DIR)
    with open(BEST_BPB_FILE, "w") as f:
        f.write(f"{val_bpb:.6f}\n")
    print(f"\n*** NEW BEST! val_bpb {val_bpb:.6f} < {prev_best:.6f} ***")
    print(f"Model saved to {CHECKPOINT_DIR}")
else:
    print(f"\nval_bpb {val_bpb:.6f} >= best {prev_best:.6f} — not saving.")
