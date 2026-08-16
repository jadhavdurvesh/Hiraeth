"""
prepare_dataset.py
-------------------
Converts DMJ Dataset Builder output (final merged JSONL, schema below)
into a chat-formatted dataset ready for SFT fine-tuning of Hiraeth.

DMJ record schema (from DMJ-Dataset-Builder README):
{
  "id": "DMJ-DS-1.0.0-00000001",
  "instruction": "...",
  "input": "",
  "output": "...",
  "metadata": {
    "language": "python",
    "category": "Programming",
    "topic": "Arrays",
    "difficulty": "Intermediate",
    "estimated_tokens": 261,
    "has_code": true,
    "source": "Magicoder-OSS-Instruct-75K"
  }
}

Usage:
  python prepare_dataset.py \
      --input ../data/final_merged.jsonl \
      --output_dir ../data \
      --val_split 0.02 \
      --system_prompt "You are Hiraeth, a helpful, precise AI assistant."

Output:
  data/train.jsonl  - each line: {"messages": [...]}
  data/val.jsonl    - each line: {"messages": [...]}

Notes on future domain-specialization (per your roadmap):
  - The optional --category_filter flag lets you build a *subset* dataset
    for a specific domain later (e.g. only "category" == "Medicine") without
    touching this script's logic — just point --input at a filtered file,
    or pass --category_filter.
  - metadata is preserved in a side file (data/train_meta.jsonl) so you can
    later do category-weighted sampling or curriculum training if you want.
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[warn] skipping malformed line {line_no}: {e}")
    return records


def to_chat_messages(record, system_prompt):
    instruction = (record.get("instruction") or "").strip()
    user_input = (record.get("input") or "").strip()
    output = (record.get("output") or "").strip()

    if not instruction or not output:
        return None

    user_content = instruction if not user_input else f"{instruction}\n\n{user_input}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output})
    return messages


def find_kaggle_input_jsonl():
    """
    Auto-detects a dataset .jsonl file under /kaggle/input/ so you don't have
    to hardcode a dataset folder name or filename that matches whatever you
    happened to name it when uploading. Only looks at top-level files inside
    each attached dataset folder (Kaggle datasets are usually flat).

    Returns the single found path, or raises a clear error listing what it
    found (or didn't find) so you can pass --input explicitly if needed.
    """
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        raise FileNotFoundError(
            "No --input given and /kaggle/input doesn't exist (not running on "
            "Kaggle, or no dataset attached). Pass --input explicitly."
        )

    candidates = sorted(input_root.glob("*/*.jsonl"))

    # Prefer files that look like Atlas/DMJ output over anything else if there
    # happen to be several — but don't be clever about it beyond that; ambiguity
    # should surface to the user rather than silently guessing wrong.
    if len(candidates) == 1:
        print(f"Auto-detected dataset file: {candidates[0]}")
        return str(candidates[0])

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No --input given and no .jsonl files found under /kaggle/input/. "
            "Make sure you attached your dataset via 'Add Input' in the Kaggle "
            "notebook sidebar, or pass --input explicitly."
        )

    # Multiple .jsonl files found — don't guess, list them so the user can pick
    listing = "\n".join(f"  - {c}" for c in candidates)
    raise FileNotFoundError(
        f"No --input given and multiple .jsonl files found under /kaggle/input/:\n"
        f"{listing}\n"
        f"Pass --input explicitly with the one you want, e.g.:\n"
        f"  --input {candidates[0]}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", default=None,
        help="Path to DMJ/Atlas final merged JSONL. If omitted, auto-detects a "
             "single .jsonl file under /kaggle/input/ (fails with a clear message "
             "if zero or multiple are found).",
    )
    ap.add_argument("--output_dir", default="../data")
    ap.add_argument("--val_split", type=float, default=0.02)
    ap.add_argument(
        "--system_prompt",
        default="You are Hiraeth, a helpful, precise AI assistant.",
    )
    ap.add_argument(
        "--category_filter",
        default=None,
        help="Comma-separated list of metadata.category values to keep (optional)",
    )
    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.input is None:
        args.input = find_kaggle_input_jsonl()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading {args.input} ...")
    records = load_jsonl(args.input)
    print(f"Loaded {len(records)} raw records")

    if args.category_filter:
        wanted = {c.strip().lower() for c in args.category_filter.split(",")}
        before = len(records)
        records = [
            r for r in records
            if (r.get("metadata", {}).get("category", "").lower() in wanted)
        ]
        print(f"Category filter {wanted}: {before} -> {len(records)} records")

    random.shuffle(records)
    if args.max_examples:
        records = records[: args.max_examples]

    categories = Counter(r.get("metadata", {}).get("category", "unknown") for r in records)
    print("Category distribution (top 10):")
    for cat, count in categories.most_common(10):
        print(f"  {cat}: {count}")

    n_val = max(1, int(len(records) * args.val_split))
    val_records = records[:n_val]
    train_records = records[n_val:]

    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    meta_path = os.path.join(args.output_dir, "train_meta.jsonl")

    n_written_train = 0
    n_skipped = 0
    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(meta_path, "w", encoding="utf-8") as f_meta:
        for r in train_records:
            msgs = to_chat_messages(r, args.system_prompt)
            if msgs is None:
                n_skipped += 1
                continue
            f_train.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            f_meta.write(json.dumps({"id": r.get("id"), "metadata": r.get("metadata", {})}, ensure_ascii=False) + "\n")
            n_written_train += 1

    n_written_val = 0
    with open(val_path, "w", encoding="utf-8") as f_val:
        for r in val_records:
            msgs = to_chat_messages(r, args.system_prompt)
            if msgs is None:
                n_skipped += 1
                continue
            f_val.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n_written_val += 1

    print(f"\nWrote {n_written_train} train / {n_written_val} val examples "
          f"({n_skipped} skipped for missing instruction/output)")
    print(f"  -> {train_path}")
    print(f"  -> {val_path}")
    print(f"  -> {meta_path} (metadata, for future curriculum/domain work)")


if __name__ == "__main__":
    main()
