#!/usr/bin/env python3
"""Convert Karpathy climbmix-400B data to flat uint16 binary for ANE native training.

Uses the same rustbpe tokenizer (vocab=8192) as the MLX pipeline so val_bpb
is directly comparable across accelerators.

Output files:
  native/data/train_karpathy.bin   — flat uint16 tokens (training shards)
  native/data/val_karpathy.bin     — flat uint16 tokens (validation shard 06542)
  native/data/token_bytes.bin      — int32[8192], byte count per token ID
"""
import os, pickle, struct, sys, time
import numpy as np

CACHE_DIR = os.path.expanduser("~/.cache/autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOK_DIR = os.path.join(CACHE_DIR, "tokenizer")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SEQ_PLUS_1 = 513  # 512 input + 1 target, native code reads SEQ+1 tokens at a time
VAL_SHARD = "shard_06542.parquet"


def load_tokenizer():
    tok_path = os.path.join(TOK_DIR, "tokenizer.pkl")
    with open(tok_path, "rb") as f:
        enc = pickle.load(f)
    assert enc.n_vocab == 8192, f"Expected vocab=8192, got {enc.n_vocab}"
    bos_id = enc.encode_single_token("<|reserved_0|>")
    print(f"Tokenizer loaded: vocab={enc.n_vocab}, BOS={bos_id}")
    return enc, bos_id


def load_token_bytes():
    """Load or build the token byte-length lookup."""
    npy_path = os.path.join(TOK_DIR, "token_bytes.npy")
    if os.path.exists(npy_path):
        tb = np.load(npy_path).astype(np.int32)
        print(f"Loaded token_bytes from {npy_path}: shape={tb.shape}")
        return tb
    raise FileNotFoundError(f"token_bytes.npy not found at {npy_path}")


def get_shards(exclude_val=True):
    """List training parquet shards, optionally excluding the val shard."""
    shards = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    if exclude_val:
        shards = [s for s in shards if s != VAL_SHARD]
    return shards


def tokenize_shard(enc, bos_id, shard_path):
    """Tokenize a parquet shard, prepending BOS to each document."""
    import pyarrow.parquet as pq
    table = pq.read_table(shard_path, columns=["text"])
    texts = table.column("text").to_pylist()
    token_lists = enc.encode_ordinary_batch(texts, num_threads=8)
    all_tokens = []
    for toks in token_lists:
        all_tokens.append(bos_id)
        all_tokens.extend(toks)
    return all_tokens


def write_uint16_bin(tokens, path):
    """Write flat uint16 binary file."""
    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(path)
    print(f"  Wrote {path}: {len(tokens):,} tokens ({os.path.getsize(path)/1e6:.1f} MB)")


def write_token_bytes_bin(token_bytes, path):
    """Write int32 binary for C code to load."""
    arr = token_bytes.astype(np.int32)
    arr.tofile(path)
    print(f"  Wrote {path}: {len(arr)} entries ({os.path.getsize(path)} bytes)")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    enc, bos_id = load_tokenizer()
    token_bytes = load_token_bytes()

    # --- Token bytes ---
    tb_path = os.path.join(OUT_DIR, "token_bytes.bin")
    write_token_bytes_bin(token_bytes, tb_path)

    # --- Validation data ---
    val_shard_path = os.path.join(DATA_DIR, VAL_SHARD)
    if not os.path.exists(val_shard_path):
        print(f"WARNING: Validation shard {VAL_SHARD} not found, skipping")
    else:
        print(f"\nTokenizing validation shard: {VAL_SHARD}")
        t0 = time.time()
        val_tokens = tokenize_shard(enc, bos_id, val_shard_path)
        print(f"  {len(val_tokens):,} tokens in {time.time()-t0:.1f}s")
        val_path = os.path.join(OUT_DIR, "val_karpathy.bin")
        write_uint16_bin(val_tokens, val_path)

    # --- Training data ---
    train_shards = get_shards(exclude_val=True)
    print(f"\nTokenizing {len(train_shards)} training shards...")
    all_train = []
    t0 = time.time()
    for i, shard in enumerate(train_shards):
        shard_path = os.path.join(DATA_DIR, shard)
        toks = tokenize_shard(enc, bos_id, shard_path)
        all_train.extend(toks)
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        print(f"  [{i+1}/{len(train_shards)}] {shard}: {len(toks):,} tokens "
              f"(total {len(all_train):,}, {rate:.1f} shards/s)")

    train_path = os.path.join(OUT_DIR, "train_karpathy.bin")
    write_uint16_bin(all_train, train_path)

    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Train: {len(all_train):,} tokens = {len(all_train)//SEQ_PLUS_1:,} sequences of {SEQ_PLUS_1}")
    if os.path.exists(val_shard_path):
        print(f"Val:   {len(val_tokens):,} tokens")


if __name__ == "__main__":
    main()
