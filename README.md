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
│   ├── run_eval.py          # runs a checkpoint against eval/eval_prompts.json, writes a report
│   └── requirements.txt
├── eval/
│   ├── eval_prompts.json    # 12 hand-written prompts spanning target skills
│   └── reports/             # generated eval reports land here (gitignored)
├── notebook/
│   └── hiraeth_train.ipynb  # Kaggle notebook: clone -> prep -> smoke test -> train -> merge -> eval -> zip
└── docs/
    ├── GETTING_STARTED.md   # simple, plain-language step-by-step walkthrough
    ├── TRAINING_GUIDE.md    # step-by-step Kaggle guide with troubleshooting
    └── PLANNING.md          # archived original design doc
```

## Status — what's actually verified

Being upfront about this since it matters for anyone picking this up:

| Piece | Status |
|---|---|
| `prepare_dataset.py` | Logic verified locally against the Hiraeth Atlas schema. Converts DMJ/Atlas records into chat-formatted JSONL correctly. |
| `train.py` | Rewritten after a real Kaggle run surfaced two deeper issues beyond the earlier fixes: (1) `device_map="auto"` naive layer-sharding across 2 GPUs was the actual cause of ~400 sec/step and uneven OOM — switched to proper `torchrun`-based DDP (each GPU gets a full model copy) as the recommended path, with the old naive-sharding kept only as an automatic single-process fallback; (2) hardcoded `SFTConfig` kwargs (`max_seq_length`, `warmup_ratio`) crashed on a differently-versioned installed `trl` — now inspects the installed API and drops/renames unsupported args with a warning instead of crashing. Also added real Flash Attention 2 support on T4 (auto-detected, safely falls back to disabling packing if unavailable). `requirements.txt` now pins the exact version combo confirmed to work end-to-end on Kaggle. Still needs a full run to confirm the DDP path performs as expected — the logic has been tested standalone (SFTConfig filtering, FA2/packing resolution, distributed detection) but not against a live GPU. |
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

**New to this? Start with
[`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md)** — a simple,
plain-language walkthrough with no assumed background.

**For the deeper technical guide with troubleshooting, read
[`docs/TRAINING_GUIDE.md`](docs/TRAINING_GUIDE.md)** — it covers real issues
people have hit (multi-GPU crashes, bf16-on-T4 slowdowns, apparent
freezes) and exactly what to check at each step.

Short version:

1. Build a Hiraeth Atlas dataset with
   [Hiraeth-Forge](https://github.com/jadhavdurvesh/Hiraeth-Forge)
   (`python build.py all`), upload the resulting JSONL as a Kaggle Dataset.
2. New Kaggle notebook → Settings → **GPU T4 x2**, **Internet: On** → attach
   your dataset via Add Input.
3. Open `notebook/hiraeth_train.ipynb` and run it — the dataset step
   auto-detects your attached file under `/kaggle/input/`, no filenames to
   edit.
4. Run top to bottom — includes a 20-step smoke test before the full
   training run, and an eval spot-check after merging.

## Pre-training improvements (added before the first real run)

- **NEFTune** (`--neftune_alpha`, default 5): adds noise to input embeddings during
  training — known to improve instruction-following quality at no extra compute
  cost. Set to 0 to disable.
- **Sequence packing** (`--packing`, default on): concatenates short examples into
  full-length blocks instead of padding each one individually, improving GPU
  utilization and training speed for the same GPU-hours. Pass `--packing false`
  to disable if you need each example in its own attention window.
- **LoRA presets** (`--lora_preset standard|large`): `standard` (r=16/alpha=32) is
  the default; `large` (r=32/alpha=64) gives more adapter capacity — worth trying
  if `standard` underfits on a larger dataset. Pass `--lora_r`/`--lora_alpha`
  directly to override the preset entirely.
- **Manual eval prompt set** (`eval/eval_prompts.json` + `scripts/run_eval.py`):
  `eval_loss` alone doesn't tell you if answers are actually good. `run_eval.py`
  runs a checkpoint against 12 prompts spanning code, reasoning, general
  knowledge, instruction-following, multi-step tasks, and safety, and writes a
  timestamped Markdown report — diff reports across checkpoints to see how
  answers change over training.

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

## Multi-GPU strategy

**Always launch training with `torchrun`, not plain `python`:**
```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --train_file ... --val_file ...
```
This runs proper data-parallel (DDP) training — each GPU loads its own full
model copy and processes its own batch, syncing only LoRA gradients. A
naive `device_map="auto"` single-process fallback exists for convenience
(or single-GPU use), but it splits the model's *layers* across GPUs instead
of replicating it — this serializes activations across the GPU boundary on
every layer and was the confirmed cause of ~400 sec/step and uneven OOM in
real testing. See `docs/TRAINING_GUIDE.md` for the full explanation.

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

## License

DMJ Community License (DCL) v1.0 — see [LICENSE.md](LICENSE.md). Free to use, study, fork, and redistribute with attribution; the Hiraeth Atlas dataset and trained Hiraeth model weights have additional restrictions on commercial resale (see Section 5).
