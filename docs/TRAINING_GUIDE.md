# Training Hiraeth on Kaggle — Step by Step

This walks through an actual working run, including the failure points people
hit in practice and exactly what to check at each one.

## 1. Before you open a notebook

Have ready:
- A Hiraeth Atlas dataset (`train.jsonl` + `val.jsonl`, or the raw merged
  file) uploaded as a **Kaggle Dataset**. See
  [Hiraeth-Forge](https://github.com/jadhavdurvesh/Hiraeth-Forge) if you
  haven't built one yet.
- Your GitHub username (this repo is cloned inside the notebook).

## 2. Create the notebook and set it up correctly

1. Kaggle → **Code → New Notebook**.
2. **Settings (right sidebar) → Accelerator → GPU T4 x2**. (P100 x2 also
   works if that's what's offered — the fixes in this guide handle both.)
3. **Settings → Internet → On.**
4. **Add Input** (right sidebar) → attach your Hiraeth Atlas Kaggle Dataset.
5. First cell — confirm the GPUs are actually what you asked for:
   ```python
   !nvidia-smi
   ```
   You should see **two** GPU entries. If you only see one, the accelerator
   setting didn't take — go back to Settings and re-select GPU T4 x2, this
   sometimes needs the notebook session restarted to apply.

## 3. Install dependencies — carefully

```python
!git clone https://github.com/jadhavdurvesh/Hiraeth.git
!pip install -q --no-deps -r Hiraeth/scripts/requirements.txt
```

**Why `--no-deps`:** Kaggle notebooks ship with a PyTorch build already
matched to their CUDA driver. A plain `pip install -r requirements.txt` can
resolve a dependency chain that silently upgrades `torch` to a version that
doesn't match Kaggle's CUDA setup — this causes confusing failures (GPU not
detected, or CUDA errors deep in a stack trace that don't look GPU-related
at all). `--no-deps` installs exactly what's pinned without touching torch.

After installing, confirm torch still sees the GPUs:
```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```
If `torch.cuda.is_available()` is `False` here, something upgraded torch —
restart the kernel and reinstall with `--no-deps` again, or check for a
version conflict in the pip install output.

## 4. Prepare the dataset

```python
!mkdir -p /kaggle/working/data
!python Hiraeth/scripts/prepare_dataset.py \
    --input /kaggle/input/<your-dataset-name>/<your-file>.jsonl \
    --output_dir /kaggle/working/data \
    --val_split 0.02
```
Replace `<your-dataset-name>` and `<your-file>.jsonl` with what's actually
under `/kaggle/input/` — run `!ls /kaggle/input/*/` first if you're not sure
of the exact path.

## 5. Run a smoke test BEFORE the full training run

This is the step most people skip, and it's the single best way to avoid
burning hours of GPU quota on a crash or a silent slowdown.

```python
!python Hiraeth/scripts/train.py \
    --train_file /kaggle/working/data/train.jsonl \
    --val_file /kaggle/working/data/val.jsonl \
    --output_dir /kaggle/working/hiraeth-smoketest \
    --max_steps 20
```

This runs only 20 steps — a few minutes, not hours. Watch for:

- **"Detected 2 GPU(s)"** and **"Using fp16 (GPU does not support bf16...)"**
  printed near the top. If it says "Using bf16" on a T4/P100, something's
  wrong with GPU detection — stop and check `nvidia-smi` output again.
- **"Multi-GPU: marked model as already parallelized"** — confirms the fix
  for the 2-GPU crash is active.
- Step timing in the logs. On a T4 with the default settings, individual
  steps should be in the **range of seconds, not minutes**. If step 1 takes
  a while (CUDA kernel warmup/compilation is normal for the very first
  step) but step 2+ are much faster, that's expected and fine. If EVERY
  step takes multiple minutes, something is still wrong — see Troubleshooting.

If the smoke test finishes cleanly in a few minutes with sane step timing,
you're clear to run the full training job.

## 6. Full training run

```python
!python Hiraeth/scripts/train.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --train_file /kaggle/working/data/train.jsonl \
    --val_file /kaggle/working/data/val.jsonl \
    --output_dir /kaggle/working/hiraeth-qlora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_seq_length 2048
```

Estimate total time before committing: take the per-step time from your
smoke test (after warmup), multiply by total steps
(`num_train_epochs × dataset_size / effective_batch_size`, where effective
batch size = `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`).
Compare against Kaggle's session limit (~9-12 hours) and weekly GPU quota
(~30 hours). If it doesn't fit in one session, you'll need to add
checkpoint-resume — not yet in `train.py`, flag it if you hit this.

## 7. Merge and download

```python
!python Hiraeth/scripts/merge_and_save.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_dir /kaggle/working/hiraeth-qlora \
    --output_dir /kaggle/working/hiraeth-merged

!zip -r -q /kaggle/working/hiraeth-merged.zip /kaggle/working/hiraeth-merged
```
Download `hiraeth-merged.zip` from the notebook's Output panel before the
session ends — `/kaggle/working` is wiped on session close.

## Troubleshooting

**Crash mentioning DataParallel, device mismatch, or "distributed mode"
right at the start of training, with 2 GPUs active:**
Should now be fixed by the `is_parallelizable`/`model_parallel` flags added
to `train.py`. If you still hit this, you're likely on an older cached
version of the script — re-`git pull` or re-clone the repo.

**Every step takes multiple minutes, no exceptions:**
1. Confirm the "Using fp16" log line appeared (see Step 5) — bf16 on
   T4/P100 is the most common cause of this exact symptom.
2. Confirm `torch.cuda.is_available()` is `True` and GPUs show up in
   `nvidia-smi` mid-training (run `!nvidia-smi` in a separate cell while
   training runs) — if GPU utilization is near 0%, training is likely
   running on CPU, which usually means the earlier `--no-deps` step was
   skipped and torch got upgraded to a CUDA-incompatible version.
3. Check you're not accidentally using `--lora_preset large` combined with
   a long `--max_seq_length` on a single GPU with limited headroom — try
   the smoke test with `--lora_preset standard` (the default) first.

**Training appears frozen after the first step:**
Should now be fixed by `dataloader_num_workers=0` (Kaggle notebooks are
known to hang with multi-worker DataLoaders). If it still happens, check
whether it's actually frozen (GPU utilization at 0% in `nvidia-smi`) or
just quiet — with `--logging_steps 5` you'll get a log line every 5 steps,
so brief silence between logs is normal, not a freeze.

**Out of memory (OOM) errors:**
Lower `--per_device_train_batch_size` to 1 and raise
`--gradient_accumulation_steps` proportionally to keep the same effective
batch size, or lower `--max_seq_length`.
