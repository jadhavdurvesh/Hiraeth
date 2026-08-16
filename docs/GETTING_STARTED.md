# Getting Started — Train Hiraeth on Kaggle (Simple Guide)

This is the plain-language version. If something breaks or you want the
technical "why," see [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) in this same
folder — that one has deeper troubleshooting. This one just gets you
running, step by step.

---

## What you need before starting

- A Kaggle account (free).
- A Hiraeth Atlas dataset file, already built with
  [Hiraeth-Forge](https://github.com/jadhavdurvesh/Hiraeth-Forge). This is
  the `.jsonl` file that ends up in `datasets/final/` after you run
  `python build.py all` there.

That's it. Everything else happens inside Kaggle.

---

## Step 1: Upload your dataset to Kaggle

1. Go to [kaggle.com](https://kaggle.com) and log in.
2. Click **Datasets** in the left sidebar, then **New Dataset**.
3. Drag in your Hiraeth Atlas `.jsonl` file.
4. Give it a name you'll remember, e.g. `hiraeth-atlas`.
5. Click **Create**.

Once it's done uploading, note the exact **filename** you uploaded — you'll
need it in Step 5.

---

## Step 2: Create a new notebook

1. Click **Code** in the left sidebar, then **New Notebook**.
2. On the right side of the screen, find **Settings**.
3. Under **Accelerator**, pick **GPU T4 x2** (or **GPU P100 x2** if that's
   what's offered — either works).
4. Under **Internet**, turn it **On**.
5. Click **Add Input** (also on the right side), search for the dataset
   you uploaded in Step 1, and click to attach it.

---

## Step 3: Check your GPUs are actually there

Click into the first cell of the notebook, paste this in, and run it
(the ▶ button, or Shift+Enter):

```python
!nvidia-smi
```

You should see **two GPU entries** listed. If you only see one, go back to
Settings and re-pick "GPU T4 x2" — sometimes it needs the notebook
restarted to take effect.

---

## Step 4: Get the Hiraeth code and install what it needs

New cell:

```python
!git clone https://github.com/jadhavdurvesh/Hiraeth.git
```

New cell:

```python
!pip install -q --no-deps -r Hiraeth/scripts/requirements.txt

import torch
print('CUDA available:', torch.cuda.is_available(), '| GPU count:', torch.cuda.device_count())
```

Run it. You should see `CUDA available: True` and `GPU count: 2`. If you
see `False`, stop here and check the Troubleshooting section in
`TRAINING_GUIDE.md` before continuing — everything after this depends on
the GPUs actually being usable.

---

## Step 5: Turn your dataset into training format

New cell — **replace the two `<...>` parts** with your actual dataset
folder name (from Step 1) and filename:

```python
!mkdir -p /kaggle/working/data
!python Hiraeth/scripts/prepare_dataset.py \
    --input /kaggle/input/<your-dataset-folder-name>/<your-filename>.jsonl \
    --output_dir /kaggle/working/data \
    --val_split 0.02
```

Not sure of the exact folder/file name? Run `!ls /kaggle/input/*/` in a
cell first — it'll show you exactly what's there.

---

## Step 6: Quick test run (do this before the real training)

This runs just 20 training steps — a few minutes, not hours. It's here so
you find out if something's broken *before* you spend a big chunk of your
weekly GPU time.

```python
!torchrun --standalone --nproc_per_node=2 Hiraeth/scripts/train.py \
    --train_file /kaggle/working/data/train.jsonl \
    --val_file /kaggle/working/data/val.jsonl \
    --output_dir /kaggle/working/hiraeth-smoketest \
    --max_steps 20
```

While this runs, look at the printed output for two lines in particular:

- **`Using DDP (data-parallel)`** — good, this means it's using both GPUs
  properly and quickly.
- **`Using fp16`** — good, this is the correct setting for Kaggle's GPUs.

Also watch the step timing as it trains. It should be a few **seconds**
per step, not minutes. If it looks fast and finishes without errors,
you're ready for the real run.

---

## Step 7: The real training run

Same command, without the `--max_steps 20` limit, and with proper training
settings:

```python
!torchrun --standalone --nproc_per_node=2 Hiraeth/scripts/train.py \
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

This will take a while — could be a few hours depending on your dataset
size. Kaggle sessions have a time limit (roughly 9-12 hours) and a weekly
GPU quota (roughly 30 hours), so keep an eye on the clock.

---

## Step 8: Turn the trained result into a usable model

Once training finishes:

```python
!python Hiraeth/scripts/merge_and_save.py \
    --base_model Qwen/Qwen2.5-7B-Instruct \
    --adapter_dir /kaggle/working/hiraeth-qlora \
    --output_dir /kaggle/working/hiraeth-merged
```

---

## Step 9: Download your model

```python
!zip -r -q /kaggle/working/hiraeth-merged.zip /kaggle/working/hiraeth-merged
!ls -lh /kaggle/working/hiraeth-merged.zip
```

Go to the notebook's **Output** panel (usually on the right or bottom of
the screen) and download `hiraeth-merged.zip`.

**Do this before you close the notebook** — Kaggle deletes everything in
`/kaggle/working` once the session ends. If the zip is very large
(a full 7B model is roughly 14-15GB), downloading straight to Hugging Face
Hub instead of a zip file is often easier — see the commented-out cell at
the bottom of `notebook/hiraeth_train.ipynb` for that option.

---

## If something goes wrong

Don't panic and don't start randomly changing package versions — that
tends to make things worse. Instead:

1. Read the actual error message. What does it say broke?
2. Check the **Troubleshooting** section in
   [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) — many issues people hit are
   already documented there with the exact fix.
3. If it's still unclear, copy the error message and ask for help with it
   directly — a specific error message is much easier to fix than "it
   didn't work."
