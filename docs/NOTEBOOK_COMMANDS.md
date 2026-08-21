# Kaggle Notebook — Cell-by-Cell Command Reference

Every command below matches `notebook/hiraeth_train.ipynb` exactly, in
order. Paste each into its own cell and run top to bottom. For explanations
of *why* each step exists, see [`GETTING_STARTED.md`](GETTING_STARTED.md)
(simple version) or [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) (technical,
with troubleshooting). This doc is just the commands, no fluff.

---

### Before cell 1: notebook settings

- **Settings → Accelerator → GPU T4 x2** (or P100 x2)
- **Settings → Internet → On**
- **Add Input** (right sidebar) → attach your Hiraeth Atlas Kaggle Dataset

---

### Cell 1 — confirm GPUs

```python
!nvidia-smi
```
Should list two GPUs. If only one shows, re-check the accelerator setting.

---

### Cell 2 — clone the repo

```python
REPO_NAME = "Hiraeth"

!git clone https://github.com/jadhavdurvesh/{REPO_NAME}.git
!ls {REPO_NAME}/scripts/
```

---

### Cell 3 — install dependencies

```python
!pip install -q --no-deps -r {REPO_NAME}/scripts/requirements.txt
!pip install -q -U --no-deps bitsandbytes

import torch
print('CUDA available:', torch.cuda.is_available(), '| GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(' ', i, torch.cuda.get_device_name(i))
```
`CUDA available` must print `True`. If it prints `False`, stop and check
`TRAINING_GUIDE.md` before continuing.

---

### Cell 3b (optional) — install Flash Attention 2

Speeds up training and re-enables `packing`. **T4 only** — skip on P100, it
won't help there.

Building from source (the simple version) can take 20-40+ minutes, since it
compiles many CUDA kernels. Try a prebuilt wheel first — when it matches
your exact torch/CUDA/Python versions, it installs in seconds instead:

```python
import torch, sys
print(torch.__version__, torch.version.cuda, sys.version)
```
If that shows torch 2.10 + CUDA 12.8 + Python 3.12 (Kaggle's usual default
as of writing):
```python
!pip install -q "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```
If your versions differ, look up the matching wheel at
[flashattn.dev](https://flashattn.dev/) instead of guessing.

**Fallback if no prebuilt wheel matches** — build from source (slow):
```python
!pip install -q flash-attn --no-build-isolation
```

If either approach fails, that's fine — this whole step is optional.
`train.py` automatically falls back to a safe default (`sdpa`, packing
disabled) without it.

---

### Cell 4 — check your attached dataset

```python
!ls /kaggle/input/
```

---

### Cell 5 — prepare the dataset

```python
!mkdir -p /kaggle/working/data
!python {REPO_NAME}/scripts/prepare_dataset.py \
    --output_dir /kaggle/working/data \
    --val_split 0.02 \
    --system_prompt "You are Hiraeth, a helpful, precise AI assistant."
```
Auto-detects your attached dataset file — nothing to fill in.

---

### Cell 6 — locate the prepared train/val files

```python
import glob

def find_file(name, prepared_path):
    import os
    if os.path.exists(prepared_path):
        return prepared_path
    matches = glob.glob(f'/kaggle/input/*/{name}')
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f'Could not find {name} at {prepared_path} or uniquely under /kaggle/input/. '
        f'Found: {matches}. Set TRAIN_FILE/VAL_FILE manually if this is ambiguous.'
    )

TRAIN_FILE = find_file('train.jsonl', '/kaggle/working/data/train.jsonl')
VAL_FILE = find_file('val.jsonl', '/kaggle/working/data/val.jsonl')
print('TRAIN_FILE:', TRAIN_FILE)
print('VAL_FILE:', VAL_FILE)
```

---

### Cell 7 — smoke test (always run this before the full training run)

```python
!torchrun --standalone --nproc_per_node=2 {REPO_NAME}/scripts/train.py \
    --train_file {TRAIN_FILE} \
    --val_file {VAL_FILE} \
    --output_dir /kaggle/working/hiraeth-smoketest \
    --max_steps 20
```
20 steps, a few minutes. Check the log for `Using DDP`, `Using fp16`, and
sane step timing before moving on. If anything looks wrong, see
`TRAINING_GUIDE.md` Troubleshooting.

---

### Cell 8 — Hugging Face login + checkpoint config

Requires an `HF_TOKEN` Kaggle Secret first (Add-ons → Secrets, Write-scope
Hugging Face token — see below for how to get one).

```python
from kaggle_secrets import UserSecretsClient
hf_token = UserSecretsClient().get_secret("HF_TOKEN")
!huggingface-cli login --token {hf_token}

HUB_CHECKPOINT_REPO = "YOUR_HF_USERNAME/hiraeth-checkpoints"  # <-- change this

# None = fresh start. Set to HUB_CHECKPOINT_REPO to resume in a NEW session.
# Set to "auto" to resume within the SAME session after a crash.
RESUME_FROM = None
```

---

### Cell 9 — full training run

```python
resume_flag = f"--resume_from_checkpoint {RESUME_FROM}" if RESUME_FROM else ""

!torchrun --standalone --nproc_per_node=2 {REPO_NAME}/scripts/train.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --train_file {TRAIN_FILE} \
    --val_file {VAL_FILE} \
    --output_dir /kaggle/working/hiraeth-qlora \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-4 \
    --max_seq_length 1024 \
    --push_checkpoint_to_hub {HUB_CHECKPOINT_REPO} \
    {resume_flag}
```
This is the long-running one. Checkpoints push to your Hub repo
automatically as it trains.

**Starting a new session later to continue?** Re-run cells 1-6 as usual,
then in cell 8 set `RESUME_FROM = HUB_CHECKPOINT_REPO`, then re-run this
cell — it downloads the latest checkpoint and continues instead of
restarting.

---

### Cell 10 — merge the adapter into a standalone model

```python
!python {REPO_NAME}/scripts/merge_and_save.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_dir /kaggle/working/hiraeth-qlora \
    --output_dir /kaggle/working/hiraeth-merged
```

---

### Cell 11 (optional) — eval spot-check

```python
!python {REPO_NAME}/scripts/run_eval.py \
    --model_dir /kaggle/working/hiraeth-merged \
    --prompts_file {REPO_NAME}/eval/eval_prompts.json \
    --output_dir /kaggle/working/eval_reports
```
Runs the model against 12 hand-written prompts, writes a readable report.

---

### Cell 12 — zip the model for download

```python
!zip -r -q /kaggle/working/hiraeth-merged.zip /kaggle/working/hiraeth-merged
!ls -lh /kaggle/working/hiraeth-merged.zip
```
Download from the notebook's Output panel **before closing the session** —
`/kaggle/working` gets wiped otherwise.

---

### Cell 12 alternative — push the final model to Hugging Face Hub instead

A 7B model zip is ~14-15GB — often easier to push straight to the Hub than
download a zip.

```python
from kaggle_secrets import UserSecretsClient
hf_token = UserSecretsClient().get_secret("HF_TOKEN")
!huggingface-cli login --token {hf_token}
!python {REPO_NAME}/scripts/merge_and_save.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_dir /kaggle/working/hiraeth-qlora \
    --output_dir /kaggle/working/hiraeth-merged \
    --push_to_hub_id YOUR_HF_USERNAME/hiraeth-7b
```

---

## Getting an HF_TOKEN (needed for cells 8 and 12-alt)

1. [huggingface.co/join](https://huggingface.co/join) — free account if you
   don't have one.
2. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   → **New token** → role **Write** → **Create token** → copy it.
3. In the Kaggle notebook: **Add-ons → Secrets → Add a new secret** →
   label exactly `HF_TOKEN`, value = the token you copied → save, make sure
   it's attached to this notebook.
