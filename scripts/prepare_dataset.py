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


def _looks_like_jsonl_dataset(path, sample_lines=3):
    """
    Sanity-checks a candidate file actually looks like Hiraeth Atlas data —
    not just any valid JSON, but JSONL where each line has the expected
    shape (Atlas records: instruction/output; or pre-formatted chat data:
    messages). This is what stops a generic config.json or similar
    unrelated file from being silently picked just because it happens to
    parse as JSON. Reads only the first few lines, not the whole file —
    this runs against every candidate, so it needs to stay cheap even on a
    100k+ record file.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            checked = 0
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)  # raises if not valid JSON on its own line
                if not isinstance(record, dict):
                    return False
                has_atlas_shape = "instruction" in record and "output" in record
                has_chat_shape = "messages" in record
                if not (has_atlas_shape or has_chat_shape):
                    return False
                checked += 1
                if checked >= sample_lines:
                    return True
        return checked > 0
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return False


def find_kaggle_input_jsonl():
    """
    Auto-detects a dataset file under /kaggle/input/ so you don't have to
    hardcode a dataset folder name or filename that matches whatever you
    happened to name it when uploading.

    Searches recursively (Kaggle datasets are usually flat, but not always),
    looks at both .jsonl AND .json extensions (Hiraeth Atlas has been
    uploaded as either), skips Kaggle's own auto-generated metadata files,
    and validates each candidate actually contains one-JSON-object-per-line
    content before trusting it — so a stray unrelated .json file won't be
    silently picked.

    Returns the single found path, or raises a clear error listing what it
    found (or didn't find) so you can pass --input explicitly if needed.
    """
    input_root = Path("/kaggle/input")
    if not input_root.exists():
        raise FileNotFoundError(
            "No --input given and /kaggle/input doesn't exist (not running on "
            "Kaggle, or no dataset attached). Pass --input explicitly."
        )

    # Kaggle auto-adds files like dataset-metadata.json to every dataset —
    # these aren't your data, filter them out by name.
    ignored_names = {"dataset-metadata.json"}

    raw_candidates = sorted(
        p for p in list(input_root.rglob("*.jsonl")) + list(input_root.rglob("*.json"))
        if p.name not in ignored_names
    )

    valid_candidates = [p for p in raw_candidates if _looks_like_jsonl_dataset(p)]

    if len(valid_candidates) == 1:
        print(f"Auto-detected dataset file: {valid_candidates[0]}")
        return str(valid_candidates[0])

    if len(valid_candidates) == 0:
        if raw_candidates:
            listing = "\n".join(f"  - {c}" for c in raw_candidates)
            raise FileNotFoundError(
                f"No --input given. Found .json/.jsonl files under /kaggle/input/, but none of "
                f"them look like valid one-JSON-object-per-line data (each line should be its own "
                f"JSON record):\n{listing}\n"
                f"If one of these IS your dataset, check it's actually JSONL formatted, or pass "
                f"--input explicitly to use it anyway."
            )
        raise FileNotFoundError(
            "No --input given and no .json/.jsonl files found anywhere under /kaggle/input/. "
            "Make sure you attached your dataset via 'Add Input' in the Kaggle notebook sidebar "
            "(check with `!ls -R /kaggle/input/` in a cell), or pass --input explicitly."
        )

    # Multiple valid candidates found — don't guess, list them so the user can pick
    listing = "\n".join(f"  - {c}" for c in valid_candidates)
    raise FileNotFoundError(
        f"No --input given and multiple valid-looking dataset files found under "
        f"/kaggle/input/:\n{listing}\n"
        f"Pass --input explicitly with the one you want, e.g.:\n"
        f"  --input {valid_candidates[0]}"
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
