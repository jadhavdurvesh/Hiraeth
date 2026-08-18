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

## 5. Why `torchrun`, not plain `python`

This matters more than anything else in this guide for training speed.

`train.py` supports two ways of using 2 GPUs:

- **`device_map="auto"` (single process, no torchrun)** — splits the
  model's *layers* across both GPUs. Every forward/backward pass has to
  serialize activations across the GPU boundary on every layer. This is
  what produced ~400 sec/step in earlier testing — it's not a bug exactly,
  it's just the wrong tool: naive layer-sharding is for models too big to
  fit on one GPU. A 7B model in 4-bit (~4.5GB) easily fits on a single 16GB
  T4, so sharding it buys nothing but overhead.
- **`torchrun` (proper data-parallel / DDP)** — each GPU loads its own
  *full copy* of the model and processes its own batch independently,
  syncing only the small LoRA gradients (0.5% of params) at the end of
  each step. This is the standard, fast way to use multiple GPUs for a
  model that fits on one.

**Always launch with `torchrun` on 2+ GPUs.** `train.py` still works with
plain `python train.py` (falls back to the slow layer-sharded path
automatically, with a loud warning), but that's a fallback, not the
intended path.

## 6. Run a smoke test BEFORE the full training run

This is the step most people skip, and it's the single best way to avoid
burning hours of GPU quota on a crash or a silent slowdown.

```python
!torchrun --standalone --nproc_per_node=2 Hiraeth/scripts/train.py \
    --train_file /kaggle/working/data/train.jsonl \
    --val_file /kaggle/working/data/val.jsonl \
    --output_dir /kaggle/working/hiraeth-smoketest \
    --max_steps 20
```

This runs only 20 steps — a few minutes, not hours. Watch for:

- **"Using DDP (data-parallel)"** printed for each rank — confirms torchrun
  launched correctly and each process owns its own GPU. If you instead see
  the "[warn] ... running as a SINGLE process" message, torchrun isn't
  actually being used — double check the command.
- **"Using fp16 (GPU does not support bf16...)"** on T4/P100. If it says
  "Using bf16" on these GPUs, something's wrong with GPU detection.
- **"Flash Attention 2 available"** on T4 (if you installed `flash-attn`),
  or a clear fallback message to sdpa with packing disabled otherwise —
  either is fine, both are handled safely now.
- Step timing in the logs. With proper DDP on 2x T4, individual steps
  should be in the **range of seconds, not minutes**. If step 1 takes a
  while (CUDA kernel warmup is normal for the very first step) but step 2+
  are much faster, that's expected. If EVERY step takes multiple minutes
  even under torchrun, see Troubleshooting.

If the smoke test finishes cleanly in a few minutes with sane step timing,
you're clear to run the full training job.

## 7. Full training run

```python
!torchrun --standalone --nproc_per_node=2 Hiraeth/scripts/train.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --train_file /kaggle/working/data/train.jsonl \
    --val_file /kaggle/working/data/val.jsonl \
    --output_dir /kaggle/working/hiraeth-qlora \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --max_seq_length 1024
```

Note: under DDP, effective batch size = `per_device_train_batch_size ×
gradient_accumulation_steps × num_gpus` — with the defaults above and 2
GPUs, that's `2 × 8 × 2 = 32`, double what a single-process run would give
you. `train.py` prints the resolved effective batch size at startup — check
it matches what you expect.

Estimate total time before committing: take the per-step time from your
smoke test (after warmup), multiply by total steps
(`num_train_epochs × dataset_size / effective_batch_size`). Compare against
Kaggle's session limit (~9-12 hours) and weekly GPU quota (~30 hours). If it
doesn't fit in one session, you'll need checkpoint-resume — not yet in
`train.py`, flag it if you hit this.

## 8. Merge and download

```python
!python Hiraeth/scripts/merge_and_save.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_dir /kaggle/working/hiraeth-qlora \
    --output_dir /kaggle/working/hiraeth-merged

!zip -r -q /kaggle/working/hiraeth-merged.zip /kaggle/working/hiraeth-merged
```
Download `hiraeth-merged.zip` from the notebook's Output panel before the
session ends — `/kaggle/working` is wiped on session close. (This step
doesn't need torchrun — it's a single-GPU merge operation, not training.)

## Troubleshooting

**`ModuleNotFoundError: No module named 'trl'` (or transformers/peft/etc):**
The install step didn't complete. Re-run Step 3's `pip install --no-deps -r requirements.txt`
and check the pip output for errors — don't skip straight to running `train.py`.

**`TypeError: SFTConfig.__init__() got an unexpected keyword argument '...'`:**
This should no longer happen — `train.py` now inspects the installed trl
version's actual `SFTConfig` signature and drops/renames unsupported
arguments with a warning instead of crashing (see `build_sft_config` in the
script). If you still see this exact crash, you're on an old cached version
of the script — re-`git pull` or re-clone the repo.

**`ImportError: tokenizers>=X,<Y is required but found Z` (or the same for
huggingface-hub):** A version mismatch between transformers and its
dependencies. Reinstall using the exact pins in `scripts/requirements.txt`
(these versions are confirmed to work together) — don't mix-and-match by
installing one package at its latest version while others stay pinned.

**`PackageNotFoundError: No package metadata was found for bitsandbytes`:**
bitsandbytes didn't install correctly. Re-run
`pip install -U --no-deps bitsandbytes` on its own and check for errors
before continuing.

**`ModuleNotFoundError: No module named 'triton.ops'` (during model
loading, traceback goes through `bitsandbytes` → `triton_based_modules`),
or `Could not find the bitsandbytes CUDA binary`:**
This means the installed bitsandbytes version is too old for Kaggle's
current CUDA version — it doesn't have a prebuilt binary for it, falls back
to a code path that imports `triton.ops`, and newer `triton` versions
removed that module entirely. `requirements.txt` deliberately does NOT pin
bitsandbytes for this reason (a pin can go stale as Kaggle's base image
updates). Fix:
```python
!pip install -q -U --no-deps bitsandbytes
```
Then restart and re-run from the install cell. If you're not sure whether
this is already current, check the version: `import bitsandbytes;
print(bitsandbytes.__version__)` — compare against the
[bitsandbytes releases page](https://github.com/bitsandbytes-foundation/bitsandbytes/releases)
to confirm you're not several versions behind.

**Crash mentioning DataParallel, device mismatch, or "distributed mode"
right at the start of training, with 2 GPUs active (running `python train.py`
directly, no torchrun):**
This is the naive-sharding fallback path, not the recommended path — see
Step 5. Switch to `torchrun --standalone --nproc_per_node=2 train.py ...`
instead, which uses proper DDP and avoids this entirely. If you still hit
this crash *while using torchrun*, that's unexpected — please report it.

**`torch.OutOfMemoryError: CUDA out of memory` on one GPU while the other
has room:**
This is a symptom of the naive single-process sharding path (`device_map=
"auto"`), which splits layers unevenly across GPUs — one GPU can end up
holding more of the model's activations than the other. Switching to
`torchrun` (proper DDP, each GPU holds a symmetric full copy) should
resolve this.

**`torch.OutOfMemoryError: CUDA out of memory` under torchrun, both GPUs
similarly loaded, error trace goes through `ForCausalLMLoss` /
`shift_logits = logits[..., :-1, :].contiguous()` (real trace seen: 13.61GiB
used of 14.56GiB on a T4, needed 1.32GiB more):**
This is a genuine memory limit, not a bug — the loss computation
materializes logits across the entire vocabulary (~152k tokens for
Qwen2.5) for every token in the batch, and that tensor alone can be several
GB at `max_seq_length=2048` with `per_device_train_batch_size=2`. Default
batch size is now 1 for exactly this reason (effective batch size stays at
16 under 2-GPU DDP thanks to `gradient_accumulation_steps=8`). If you still
OOM at batch size 1:
1. Lower `--max_seq_length` (e.g. to 1024) — this directly shrinks the same
   oversized logits tensor, often more effective than batch size alone if
   your dataset's examples are mostly shorter than 2048 tokens anyway
   (check `python build.py stats` in Hiraeth-Forge for your actual
   token-length distribution).
2. `train.py` now sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   automatically, which helps with fragmentation-related OOMs specifically
   — but won't help if memory is genuinely, not just fragmentedly, full.

**Every step takes multiple minutes (~400 sec/step or similar), even with
2 GPUs "detected":**
1. Confirm you launched with `torchrun --standalone --nproc_per_node=2`,
   not plain `python train.py` — this is the single biggest cause of this
   exact symptom (see Step 5's explanation of naive sharding vs. DDP).
2. Confirm the "Using fp16" log line appeared — bf16 on T4/P100 is a
   second common cause of the same symptom.
3. Confirm `torch.cuda.is_available()` is `True` and GPUs show up in
   `nvidia-smi` mid-training (run `!nvidia-smi` in a separate cell while
   training runs) — if GPU utilization is near 0%, training is likely
   running on CPU, which usually means the earlier `--no-deps` step was
   skipped and torch got upgraded to a CUDA-incompatible version.

**"Padding-free training is enabled but the attention implementation is
not a supported Flash Attention variant" / packing cross-contamination
warning:**
Should no longer appear — `train.py` now checks GPU compute capability and
whether `flash-attn` is installed, and automatically disables packing
(falling back to `sdpa`) if Flash Attention 2 isn't actually available,
rather than proceeding with the unsafe combination. If you want full
packing speed on T4, install Flash Attention 2 first:
`pip install flash-attn --no-build-isolation` (can take several minutes to
build). Not supported on P100 regardless (compute capability too low).

**Training appears frozen after the first step:**
Should be fixed by `dataloader_num_workers=0` (Kaggle notebooks are known
to hang with multi-worker DataLoaders). If it still happens, check whether
it's actually frozen (GPU utilization at 0% in `nvidia-smi`) or just quiet
— with `--logging_steps 5` you'll get a log line every 5 steps, so brief
silence between logs is normal, not a freeze.

**Both GPUs show ~100% utilization but the step count never advances past
0/N, under `torchrun` (real symptom seen: stuck at `0/20`, GPUs pinned at
100%):**
This is a different kind of "frozen" than the DataLoader hang above — it's
NCCL stuck waiting on a distributed collective operation, not idle. Look
earlier in the log for this warning:
```
Guessing device ID based on global rank. This can cause a hang if rank to
GPU mapping is heterogeneous. You can specify device_id in
init_process_group()
```
If you see it, that's the cause — `train.py` now explicitly calls
`torch.cuda.set_device(local_rank)` before any distributed communication to
prevent exactly this (fixed after a real Kaggle run hit it). Re-clone or
`git pull` to get the fix, then re-run. 100% GPU utilization with zero step
progress for more than a few minutes is the signal to suspect this — a
merely-slow-but-working first step (CUDA warmup) should still complete
within a minute or two on a 7B model.

**Large list of pip dependency-conflict warnings (google-colab, cudf,
numba, torchvision, etc.) during install:**
These are almost always about *other* preinstalled Kaggle packages
unrelated to this training pipeline, not the actual cause of a training
failure. Don't chase every one of these down — focus on whether `import
torch; import transformers; import trl; import peft; import bitsandbytes`
all succeed and `torch.cuda.is_available()` is `True` after install; that's
what actually matters for training.
