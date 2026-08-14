"""
train.py — QLoRA fine-tuning for Hiraeth
------------------------------------------
Runs in a single Kaggle notebook process. Uses 4-bit QLoRA and
device_map="auto" so the base model is automatically sharded across
both GPUs (no torchrun/accelerate-launch needed inside a notebook cell,
which avoids the usual multi-process headaches on Kaggle).

Base model default: Qwen/Qwen2.5-7B-Instruct
  - Apache-2.0 licensed, strong general + code performance
  - 4-bit QLoRA fits within 2x 16GB (T4/P100) with room for activations

Swap --base_model to a smaller model (e.g. Qwen/Qwen2.5-3B-Instruct or
meta-llama/Llama-3.2-3B-Instruct) if you want faster iteration first.

Enabled by default (see argparse help below for details):
  - Packing: concatenates short examples into full-length blocks -> better
    GPU utilization, faster training for the same GPU-hours.
  - NEFTune (alpha=5): noise on input embeddings during training, known to
    improve instruction-following quality at no extra compute cost.
  - LoRA preset "standard" (r=16, alpha=32) — pass --lora_preset large for
    r=32/alpha=64 if you want more adapter capacity on a bigger dataset.

Usage (inside a Kaggle notebook cell, GPU T4 x2 or P100 x2 enabled):

  !python train.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --train_file ../data/train.jsonl \
      --val_file ../data/val.jsonl \
      --output_dir /kaggle/working/hiraeth-qlora \
      --num_train_epochs 3 \
      --per_device_train_batch_size 2 \
      --gradient_accumulation_steps 8 \
      --learning_rate 2e-4 \
      --max_seq_length 2048

  # Try the larger LoRA preset instead:
  !python train.py ... --lora_preset large

  # Disable NEFTune or packing if you want the plain baseline behavior:
  !python train.py ... --neftune_alpha 0 --packing false
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--train_file", default="../data/train.jsonl")
    ap.add_argument("--val_file", default="../data/val.jsonl")
    ap.add_argument("--output_dir", default="/kaggle/working/hiraeth-qlora")
    ap.add_argument("--num_train_epochs", type=float, default=3)
    ap.add_argument("--per_device_train_batch_size", type=int, default=2)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=2)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=2048)

    # LoRA rank/alpha: choose a preset, or override r/alpha manually.
    # "standard" (r=16, alpha=32) — lighter, faster, less VRAM. Good default.
    # "large" (r=32, alpha=64) — more adapter capacity, more VRAM/time.
    #   Worth trying if standard underfits on a larger (50k+) dataset.
    ap.add_argument(
        "--lora_preset", choices=["standard", "large"], default="standard",
        help="standard = r16/alpha32 (default). large = r32/alpha64 (more capacity, more VRAM/time).",
    )
    ap.add_argument("--lora_r", type=int, default=None, help="Overrides --lora_preset if set")
    ap.add_argument("--lora_alpha", type=int, default=None, help="Overrides --lora_preset if set")
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # NEFTune: adds noise to input embeddings during training, which has been shown
    # to improve instruction-following quality at ~no extra compute cost.
    # Set to 0 to disable. 5 is the commonly-used default from the NEFTune paper.
    ap.add_argument(
        "--neftune_alpha", type=float, default=5.0,
        help="NEFTune noise alpha. 0 disables NEFTune entirely.",
    )

    # Packing: concatenates multiple short examples into one max_seq_length block
    # instead of padding each example individually. Meaningfully improves GPU
    # utilization/training speed on datasets with lots of short examples — matters
    # on Kaggle's limited weekly GPU-hour quota. Default on; disable if you need
    # each example to stay in its own attention window (e.g. very long single examples).
    ap.add_argument("--packing", type=lambda x: x.lower() != "false", default=True,
                     help="Pass --packing false to disable")

    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


LORA_PRESETS = {"standard": (16, 32), "large": (32, 64)}


def main():
    args = parse_args()

    # Resolve LoRA r/alpha: explicit --lora_r/--lora_alpha wins, else use the preset
    preset_r, preset_alpha = LORA_PRESETS[args.lora_preset]
    lora_r = args.lora_r if args.lora_r is not None else preset_r
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else preset_alpha
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha} "
          f"({'explicit override' if args.lora_r is not None else f'preset: {args.lora_preset}'})")

    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s)")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" shards the model across BOTH visible GPUs
    # (this is model-parallel sharding, not DDP — the right choice for
    # a single-process Kaggle notebook cell with 2 GPUs).
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading datasets...")
    data_files = {"train": args.train_file}
    if os.path.exists(args.val_file):
        data_files["validation"] = args.val_file
    dataset = load_dataset("json", data_files=data_files)

    def format_example(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if "validation" in dataset else None,
        eval_strategy="steps" if "validation" in dataset else "no",
        save_strategy="steps",
        save_total_limit=2,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        packing=args.packing,
        neftune_noise_alpha=args.neftune_alpha if args.neftune_alpha > 0 else None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Done. To merge the adapter into a standalone model, run merge_and_save.py")


if __name__ == "__main__":
    main()
