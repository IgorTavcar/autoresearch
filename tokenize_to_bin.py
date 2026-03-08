"""
Convert parquet data shards to flat uint16 binary for native ANE training.

Reads parquet shards + tokenizer from ~/.cache/autoresearch/ (created by prepare.py),
tokenizes all text, and writes a packed uint16 binary file.

Usage (from autoresearch-macos repo which has the right Python deps):
    cd ~/Dev/autoresearch-macos
    uv run python ~/Dev/autoresearch-ANE/tokenize_to_bin.py

Or with explicit output path:
    uv run python ~/Dev/autoresearch-ANE/tokenize_to_bin.py --output ~/Dev/autoresearch-ANE/data/train.bin
"""

import os
import sys
import struct
import argparse
import pickle
import time

import pyarrow.parquet as pq
import tiktoken

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
VAL_SHARD = 6542
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"

def load_tokenizer():
    path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    if not os.path.exists(path):
        print(f"Tokenizer not found at {path}")
        print("Run: cd ~/Dev/autoresearch-macos && uv run python prepare.py")
        sys.exit(1)
    with open(path, "rb") as f:
        enc = pickle.load(f)
    print(f"Tokenizer loaded: vocab_size={enc.n_vocab}")
    return enc

def list_train_shards():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    # Exclude validation shard
    files = [f for f in files if f != VAL_FILENAME]
    return [os.path.join(DATA_DIR, f) for f in files]

def tokenize_to_bin(output_path, max_tokens=None):
    enc = load_tokenizer()
    shards = list_train_shards()
    if not shards:
        print(f"No training shards found in {DATA_DIR}")
        print("Run: cd ~/Dev/autoresearch-macos && uv run python prepare.py")
        sys.exit(1)
    print(f"Found {len(shards)} training shards")

    # BOS token
    special_tokens = [f"<|reserved_{i}|>" for i in range(4)]
    bos_id = enc.encode_single_token(special_tokens[0])
    print(f"BOS token ID: {bos_id}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.time()
    total_tokens = 0
    total_docs = 0

    with open(output_path, "wb") as out:
        for shard_idx, shard_path in enumerate(shards):
            shard_name = os.path.basename(shard_path)
            pf = pq.ParquetFile(shard_path)
            shard_tokens = 0

            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column("text").to_pylist()
                # Batch tokenize
                token_lists = enc.encode_ordinary_batch(texts, num_threads=8)

                for tokens in token_lists:
                    # Prepend BOS, write as uint16
                    doc_ids = [bos_id] + tokens
                    buf = struct.pack(f"<{len(doc_ids)}H", *doc_ids)
                    out.write(buf)
                    shard_tokens += len(doc_ids)
                    total_docs += 1

            total_tokens += shard_tokens
            elapsed = time.time() - t0
            print(f"  [{shard_idx+1}/{len(shards)}] {shard_name}: {shard_tokens:,} tokens ({total_tokens:,} total, {elapsed:.1f}s)")

            if max_tokens and total_tokens >= max_tokens:
                print(f"Reached max_tokens limit ({max_tokens:,})")
                break

    elapsed = time.time() - t0
    file_size = os.path.getsize(output_path)
    print(f"\nDone: {total_tokens:,} tokens from {total_docs:,} documents")
    print(f"Output: {output_path} ({file_size / 1e6:.1f} MB)")
    print(f"Time: {elapsed:.1f}s ({total_tokens / elapsed:,.0f} tokens/sec)")

if __name__ == "__main__":
    # Default output next to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_output = os.path.join(script_dir, "data", "train.bin")

    parser = argparse.ArgumentParser(description="Tokenize parquet shards to uint16 binary for ANE training")
    parser.add_argument("--output", type=str, default=default_output, help=f"Output binary path (default: {default_output})")
    parser.add_argument("--max-tokens", type=int, default=None, help="Stop after this many tokens (default: all)")
    args = parser.parse_args()

    tokenize_to_bin(args.output, args.max_tokens)
