# Hiraeth — Fine-tuning Pipeline

Fine-tunes a base LLM into **Hiraeth** using data built with
[DMJ-Dataset-Builder](https://github.com/jadhavdurvesh/DMJ-Dataset-Builder),
trained with QLoRA on Kaggle's 2x GPU (T4 or P100) notebooks.

## Why this setup

- **Base model:** `Qwen/Qwen2.5-7B-Instruct` (Apache-2.0) — strong on both
  general chat and code, and small enough to QLoRA-tune on 2x 16GB GPUs.
  Swap to a 3B model in the args if you want faster iteration first.
- **Method:** QLoRA (4-bit NF4 base + LoRA adapters) — full fine-tuning of a
  7B model isn't realistic on Kaggle's GPUs or with a 10k-100k example
  dataset (you'd overfit and burn the session time-limit anyway).
- **GPU strategy:** `device_map="auto"` shards the base model across both
  GPUs within a single notebook process. This avoids the multi-process
  (`torchrun`/`accelerate launch`) setup that's awkward inside a Kaggle
  notebook cell, and is the standard approach for 2-GPU QLoRA on Kaggle.
- **Future domain-specialization:** metadata (`category`, `topic`,
  `difficulty`, `source`) from your DMJ records is preserved in
  `data/train_meta.jsonl` and `prepare_dataset.py` has a `--category_filter`
  flag, so you can later carve out a domain-specific subset or do
  category-weighted training without changing the core pipeline.

## Folder structure

```
hiraeth/
├── scripts/
│   ├── prepare_dataset.py   # DMJ JSONL -> chat-formatted train/val JSONL
│   ├── train.py             # QLoRA SFT training
│   ├── merge_and_save.py    # merge LoRA adapter into a standalone model
│   ├── chat.py              # quick manual test/chat with the result
│   └── requirements.txt
├── notebook/
│   └── hiraeth_train.ipynb  # ready-to-run Kaggle notebook (2 GPU)
└── data/                    # (empty — filled by prepare_dataset.py)
```

## Steps

1. **Build your dataset** with DMJ-Dataset-Builder (`build.py download`,
   `convert`, `validate`, `merge`) to produce a final merged `.jsonl`
   matching the schema in its README.

2. **Prepare it for chat fine-tuning:**
   ```
   python scripts/prepare_dataset.py \
       --input path/to/final_merged.jsonl \
       --output_dir data \
       --val_split 0.02
   ```

3. **Upload to Kaggle** — either upload `data/train.jsonl` + `data/val.jsonl`
   as a Kaggle Dataset, or upload the DMJ repo output and run
   `prepare_dataset.py` inside the notebook (the provided notebook does the
   latter). Also upload the `scripts/` folder as notebook input, or paste
   the scripts into cells.

4. **Enable 2x GPU + Internet** in the Kaggle notebook settings, then run
   `notebook/hiraeth_train.ipynb` top to bottom.

5. Result: a merged standalone Hiraeth model in
   `/kaggle/working/hiraeth-merged`, ready to zip/download or push to the
   Hugging Face Hub (`merge_and_save.py --push_to_hub_id you/hiraeth-7b`).

## Tuning knobs for your dataset size (10k-100k examples)

- `--num_train_epochs 3` is a reasonable starting point; watch `eval_loss`
  in the notebook logs and stop earlier if it starts climbing (overfitting).
- `--per_device_train_batch_size 2` + `--gradient_accumulation_steps 8`
  gives an effective batch size of 16 across the 2 GPUs — adjust down if
  you hit OOM, up if you have headroom.
- If your dataset skews heavily toward one category (e.g. mostly code),
  either accept that (fine for a general+code assistant) or use
  `--category_filter` to balance a training subset.

## Notes / things you'll want to decide as you go

- Kaggle notebook sessions have a **~9 hour** and (usually) **30 hrs/week**
  GPU quota — a 7B QLoRA run over ~30-80k examples at 3 epochs will likely
  need to be split across sessions using `--resume_from_checkpoint` (pass
  the last checkpoint dir to `SFTConfig`/`train.py` — you'll want to add
  a `--resume_from_checkpoint` arg if a single run doesn't finish; happy to
  add that if you hit this).
- Nothing here does RLHF/DPO — this is supervised fine-tuning (SFT) only,
  which matches an instruction-tuning dataset like DMJ's output.
