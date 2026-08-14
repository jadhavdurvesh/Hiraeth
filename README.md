# Hiraeth

Fine-tuning pipeline for **Hiraeth**, a chat + code instruction-tuned LLM
built on `Qwen/Qwen2.5-7B-Instruct` via QLoRA, trained on Kaggle's free 2×
GPU (T4/P100) notebooks.

Training data comes from
[Hiraeth-Forge](https://github.com/jadhavdurvesh/Hiraeth-Forge), a separate
pipeline that downloads, cleans, and merges instruction-tuning datasets into
**Hiraeth Atlas** (the actual training set).

## What's here

```
Hiraeth/
├── scripts/
│   ├── prepare_dataset.py   # Hiraeth Atlas JSONL -> chat-formatted train/val
│   ├── train.py             # QLoRA SFT training (2-GPU sharded via device_map="auto")
│   ├── merge_and_save.py    # merge LoRA adapter into a standalone model
│   ├── chat.py              # manual chat/test loop, reports token usage per turn
│   └── requirements.txt
├── notebook/
│   └── hiraeth_train.ipynb  # Kaggle notebook: clone repo -> prep -> train -> merge -> zip
└── docs/
    └── PLANNING.md          # archived original design doc
```

## Status — what's actually verified

Being upfront about this since it matters for anyone picking this up:

| Piece | Status |
|---|---|
| `prepare_dataset.py` | Logic verified locally against the Hiraeth Atlas schema. Converts DMJ/Atlas records into chat-formatted JSONL correctly. |
| `train.py` | Written and syntax-checked. **Not yet run end-to-end** — needs a GPU environment (Kaggle) that this dev environment doesn't have. |
| `merge_and_save.py` | Written and syntax-checked, same caveat — needs an actual trained adapter to merge. |
| `chat.py` | Written and syntax-checked, includes per-turn and session token-usage reporting. Not yet run against a real trained model. |
| `notebook/hiraeth_train.ipynb` | Wired up to clone this repo, pull a Kaggle-Dataset-attached Hiraeth Atlas file, and run the full pipeline. Not yet executed on Kaggle. |

In short: the code is complete and internally consistent, but **the first
real training run hasn't happened yet.** The next real milestone is running
`build.py all` in Hiraeth-Forge to produce a real Hiraeth Atlas dataset,
then running this repo's notebook on Kaggle against it.

## How the pieces fit together

```
Hiraeth-Forge (separate repo)
        │
        ▼
  Hiraeth Atlas (final_merged.jsonl)
        │
        ▼
  prepare_dataset.py  →  train.jsonl / val.jsonl (chat-formatted)
        │
        ▼
  train.py  →  QLoRA adapter (4-bit base + LoRA weights)
        │
        ▼
  merge_and_save.py  →  standalone merged Hiraeth model
        │
        ▼
  chat.py  →  manual testing, with token usage reported per turn
```

## Quick start (on Kaggle)

1. Build a Hiraeth Atlas dataset with
   [Hiraeth-Forge](https://github.com/jadhavdurvesh/Hiraeth-Forge)
   (`python build.py all`), upload the resulting JSONL as a Kaggle Dataset.
2. New Kaggle notebook → Settings → **GPU T4 x2**, **Internet: On** → attach
   your dataset via Add Input.
3. Open `notebook/hiraeth_train.ipynb`, set `GITHUB_USERNAME`,
   `DATASET_NAME`, and `RAW_FILENAME` at the top of the relevant cells to
   match your setup.
4. Run top to bottom. Last cell zips the merged model for download (also has
   a commented-out Hugging Face Hub push as an alternative).

## Token usage in chat.py

`chat.py` reports prompt tokens, completion tokens, and running session
totals after every turn — the same way hosted chat APIs report usage. Two
things worth knowing:

- **Prompt tokens grow every turn** because the full running conversation
  (system prompt + all prior turns) is re-tokenized and re-sent to the model
  each time — this isn't a bug, it's how the chat template works and matches
  how hosted APIs bill multi-turn conversations.
- This is a **real tokenizer count**, different from the `estimated_tokens`
  field in Hiraeth Atlas metadata, which is a rough word-count estimate used
  during dataset prep — not the same number.

## Design notes

- **Base model:** Qwen2.5-7B-Instruct — Apache-2.0, strong on general chat
  and code, small enough to QLoRA-tune on 2× 16GB GPUs.
- **Method:** QLoRA (4-bit NF4 + LoRA adapters), not full fine-tuning — not
  realistic on Kaggle's GPUs or a 10k-100k example dataset.
- **GPU strategy:** `device_map="auto"` shards the base model across both
  GPUs within a single notebook process, avoiding `torchrun`/multi-process
  setup that's awkward inside a Kaggle notebook cell.
- This is supervised fine-tuning (SFT) only — no RLHF/DPO here.

## Known gaps / things to add if you hit them

- No `--resume_from_checkpoint` support in `train.py` yet — if a training
  run doesn't finish in one Kaggle session (sessions cap around 9-12 hours,
  ~30 GPU-hrs/week), you'll want to add this.
- Topic/category balance in training data depends entirely on what
  Hiraeth-Forge produces — check `python build.py stats` there before
  training if you care about balance across categories.
