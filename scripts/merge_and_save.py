"""
merge_and_save.py
-------------------
Merges the trained LoRA adapter weights into the base model to produce
a standalone "Hiraeth" model (no PEFT dependency needed to run it),
then saves it locally and optionally pushes to the Hugging Face Hub.

Usage:
  python merge_and_save.py \
      --base_model Qwen/Qwen2.5-7B-Instruct \
      --adapter_dir /kaggle/working/hiraeth-qlora \
      --output_dir /kaggle/working/hiraeth-merged \
      --push_to_hub_id yourname/hiraeth-7b   # optional
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--push_to_hub_id", default=None)
    ap.add_argument("--hub_private", action="store_true")
    args = ap.parse_args()

    print(f"Loading base model {args.base_model} in bf16 (full precision merge)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print(f"Loading LoRA adapter from {args.adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    print("Merging adapter into base weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub_id:
        print(f"Pushing to Hugging Face Hub: {args.push_to_hub_id}")
        model.push_to_hub(args.push_to_hub_id, private=args.hub_private)
        tokenizer.push_to_hub(args.push_to_hub_id, private=args.hub_private)

    print("Done.")


if __name__ == "__main__":
    main()
