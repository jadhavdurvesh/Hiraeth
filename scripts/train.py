"""
train.py — QLoRA fine-tuning for Hiraeth
------------------------------------------
Two ways to run this, and the choice matters a lot for speed:

  RECOMMENDED — true multi-GPU data-parallel (DDP) via torchrun:
    Each GPU loads its own full copy of the quantized model and processes
    its own batch shard independently, only syncing LoRA gradients at the
    end of each step. This is the fast, standard way to use multiple GPUs
    for QLoRA fine-tuning of a model that fits on a single GPU (a 7B model
    in 4-bit is ~4.5GB — comfortably fits on one 16GB T4).

      !torchrun --standalone --nproc_per_node=2 train.py \
          --train_file ../data/train.jsonl \
          --val_file ../data/val.jsonl \
          --output_dir /kaggle/working/hiraeth-qlora

  FALLBACK — single process, `device_map="auto"` (naive pipeline
  parallelism): splits the model's LAYERS across GPUs instead of
  replicating it. Every forward/backward pass has to serialize activations
  across the GPU boundary on every layer — this is dramatically slower
  than DDP (this was very likely the cause of ~400 sec/step in testing) and
  can also OOM unevenly across GPUs. Only used automatically if you run
  `python train.py` directly without torchrun. Fine for a single GPU;
  avoid it for 2+ GPUs if you can use torchrun instead.

    !python train.py --train_file ... --val_file ...   # single GPU, or
                                                          # naive multi-GPU fallback

Base model default: Qwen/Qwen2.5-7B-Instruct
  - Apache-2.0 licensed, strong general + code performance
  - 4-bit QLoRA fits comfortably on a single 16GB T4/P100

Robustness against TRL API drift: different trl versions accept different
SFTConfig keyword arguments (this caused real crashes: `max_seq_length` and
`warmup_ratio` being rejected with "unexpected keyword argument" on some
installed versions). This script now inspects the installed SFTConfig's
actual signature, remaps known renamed args (e.g. max_seq_length ->
max_length in newer trl), and drops+warns on anything unsupported instead
of crashing outright.

Flash Attention 2: enabled automatically when packing is on AND the GPU's
compute capability supports it (>=7.5 — T4 qualifies, P100 does not).
Falls back to sdpa + packing disabled if flash-attn isn't installed or the
GPU doesn't support it, avoiding the cross-sample-contamination risk TRL
warns about when packing without FA2.

Usage:

  # Quick smoke test first (works with or without torchrun) — confirms the
  # pipeline works, a few minutes:
  !torchrun --standalone --nproc_per_node=2 train.py \
      --train_file ../data/train.jsonl --val_file ../data/val.jsonl \
      --output_dir /kaggle/working/hiraeth-smoketest --max_steps 20

  # Full run:
  !torchrun --standalone --nproc_per_node=2 train.py \
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
  !torchrun ... train.py ... --lora_preset large

  # Disable NEFTune or packing if you want the plain baseline behavior:
  !torchrun ... train.py ... --neftune_alpha 0 --packing false
"""

import argparse
import inspect
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

    ap.add_argument(
        "--lora_preset", choices=["standard", "large"], default="standard",
        help="standard = r16/alpha32 (default). large = r32/alpha64 (more capacity, more VRAM/time).",
    )
    ap.add_argument("--lora_r", type=int, default=None, help="Overrides --lora_preset if set")
    ap.add_argument("--lora_alpha", type=int, default=None, help="Overrides --lora_preset if set")
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument(
        "--neftune_alpha", type=float, default=5.0,
        help="NEFTune noise alpha. 0 disables NEFTune entirely.",
    )
    ap.add_argument("--packing", type=lambda x: x.lower() != "false", default=True,
                     help="Pass --packing false to disable. Auto-disabled if Flash Attention 2 "
                          "isn't available, regardless of this flag, to avoid cross-sample "
                          "attention contamination.")

    ap.add_argument("--logging_steps", type=int, default=5)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--max_steps", type=int, default=-1,
        help="Cap total training steps. Use a small number (e.g. 20) for a quick "
             "smoke test before committing to a full run. -1 = no cap.",
    )
    return ap.parse_args()


LORA_PRESETS = {"standard": (16, 32), "large": (32, 64)}


def resolve_distributed():
    """Detect whether we're running under torchrun (proper DDP) or as a
    single process (naive fallback for multi-GPU, or just single-GPU)."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_distributed = local_rank != -1 and world_size > 1
    return is_distributed, local_rank, world_size


def resolve_attention_and_packing(packing_requested, n_gpus):
    """
    Decide attn_implementation + whether packing can safely stay on.
    FA2 needs compute capability >= 7.5 (T4 qualifies, P100 does not) and
    the flash-attn package installed. Without FA2, packing risks
    cross-sample attention contamination (the exact warning TRL gives) —
    so we disable packing rather than silently accept that risk.
    """
    if not torch.cuda.is_available():
        return "sdpa", False

    major, minor = torch.cuda.get_device_capability(0)
    supports_fa2 = (major, minor) >= (7, 5)

    if not packing_requested:
        return "sdpa", False

    if not supports_fa2:
        print(f"[warn] GPU compute capability {major}.{minor} doesn't support Flash Attention 2 "
              f"(needs >=7.5 — T4 qualifies, P100 does not). Disabling packing, using sdpa "
              f"to avoid cross-sample attention contamination.")
        return "sdpa", False

    try:
        import flash_attn  # noqa: F401
        print("Flash Attention 2 available — using it with packing enabled.")
        return "flash_attention_2", True
    except ImportError:
        print("[warn] Packing requested and GPU supports Flash Attention 2, but the `flash-attn` "
              "package isn't installed. Disabling packing, using sdpa instead. To get full packing "
              "speed, install it with: pip install flash-attn --no-build-isolation "
              "(can take several minutes to build).")
        return "sdpa", False


def build_sft_config(**kwargs):
    """
    Filters kwargs against the ACTUALLY INSTALLED trl version's SFTConfig
    signature, remapping known renamed args, and dropping (with a warning)
    anything unsupported instead of crashing. This is what fixes the
    'unexpected keyword argument max_seq_length / warmup_ratio' crashes —
    those happen when the installed trl version's API doesn't match what
    the script assumes.
    """
    sig = inspect.signature(SFTConfig.__init__)
    accepted = set(sig.parameters.keys())

    # Known cross-version renames in trl's SFTConfig
    rename_map = {"max_seq_length": "max_length"}

    filtered = {}
    dropped = []
    for key, value in kwargs.items():
        if key in accepted:
            filtered[key] = value
        elif key in rename_map and rename_map[key] in accepted:
            filtered[rename_map[key]] = value
            print(f"[info] renamed '{key}' -> '{rename_map[key]}' for installed trl version")
        else:
            dropped.append(key)

    if dropped:
        print(f"[warn] installed trl's SFTConfig doesn't support these args, using its defaults "
              f"instead (this is non-fatal): {dropped}")

    return SFTConfig(**filtered)


def main():
    args = parse_args()

    preset_r, preset_alpha = LORA_PRESETS[args.lora_preset]
    lora_r = args.lora_r if args.lora_r is not None else preset_r
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else preset_alpha
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha} "
          f"({'explicit override' if args.lora_r is not None else f'preset: {args.lora_preset}'})")

    is_distributed, local_rank, world_size = resolve_distributed()
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s) visible" + (f", running distributed rank {local_rank}/{world_size}" if is_distributed else ""))
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    if is_distributed:
        # Proper DDP: each process loads its own full model copy pinned to
        # its own GPU. Trainer auto-detects torchrun's env vars and handles
        # gradient sync across processes — no extra code needed for that part.
        device_map = {"": local_rank}
        print(f"Using DDP (data-parallel): this process owns GPU {local_rank}, "
              f"loading a full model copy there. This is the fast multi-GPU path.")
    elif n_gpus > 1:
        device_map = "auto"
        print(f"[warn] {n_gpus} GPUs visible but running as a SINGLE process (no torchrun). "
              f"Falling back to naive layer-sharding across GPUs (device_map='auto') — this is "
              f"MUCH slower than proper DDP (activations serialize across the GPU boundary on "
              f"every layer; this is very likely the cause of extremely slow steps like "
              f"~400 sec/step). Strongly recommended: relaunch with torchrun instead:\n"
              f"  torchrun --standalone --nproc_per_node={n_gpus} train.py ...")
    else:
        device_map = "auto"  # single GPU — harmless, no sharding needed

    use_bf16 = n_gpus > 0 and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Using {'bf16' if use_bf16 else 'fp16'} "
          f"({'GPU supports bf16' if use_bf16 else 'GPU does not support bf16 (T4/P100) — using fp16 instead'})")

    attn_implementation, packing = resolve_attention_and_packing(args.packing, n_gpus)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Only needed for the naive single-process multi-GPU fallback — DDP
    # (is_distributed=True) doesn't need these, Trainer handles it natively
    # via torchrun's env vars.
    if not is_distributed and n_gpus > 1:
        model.is_parallelizable = True
        model.model_parallel = True

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

    sft_config = build_sft_config(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
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
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to="none",
        seed=args.seed,
        dataset_text_field="text",
        packing=packing,
        neftune_noise_alpha=args.neftune_alpha if args.neftune_alpha > 0 else None,
        dataloader_num_workers=0,
        logging_first_step=True,
    )

    effective_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps * max(world_size, 1)
    print(f"Effective batch size: {effective_batch} "
          f"({args.per_device_train_batch_size} per-device x {args.gradient_accumulation_steps} "
          f"grad-accum x {max(world_size, 1)} process(es))")

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
