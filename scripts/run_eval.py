"""
run_eval.py — runs a trained/merged Hiraeth checkpoint against eval/eval_prompts.json
and writes a Markdown report for manual spot-checking. This is NOT an automated
benchmark — eval_loss tells you if training is overfitting, but it doesn't tell you
if the model's actual answers are good. This script gives you something to read.

Usage:
  python run_eval.py --model_dir /kaggle/working/hiraeth-merged
  # or against a LoRA adapter without merging:
  python run_eval.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_dir /kaggle/working/hiraeth-qlora

Output: eval/reports/<timestamp>_eval.md — one file per run, so you can diff
reports across checkpoints/epochs to see how answers change over training.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(args):
    if args.model_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
        label = args.model_dir
    else:
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb_config, device_map="auto"
        )
        model = PeftModel.from_pretrained(base, args.adapter_dir)
        label = f"{args.base_model} + adapter {args.adapter_dir}"

    model.eval()
    return model, tokenizer, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None, help="Path to a merged standalone model")
    ap.add_argument("--base_model", default=None, help="Base model id (if using --adapter_dir)")
    ap.add_argument("--adapter_dir", default=None, help="LoRA adapter dir (if not merged)")
    ap.add_argument("--prompts_file", default="eval/eval_prompts.json")
    ap.add_argument("--system_prompt", default="You are Hiraeth, a helpful, precise AI assistant.")
    ap.add_argument("--max_new_tokens", type=int, default=400)
    ap.add_argument("--output_dir", default="eval/reports")
    args = ap.parse_args()

    if not args.model_dir and not (args.base_model and args.adapter_dir):
        raise ValueError("Provide either --model_dir, or both --base_model and --adapter_dir")

    with open(args.prompts_file) as f:
        prompts = json.load(f)

    model, tokenizer, label = load_model(args)

    results = []
    for item in prompts:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": item["prompt"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        completion_ids = output_ids[0][prompt_tokens:]
        reply = tokenizer.decode(completion_ids, skip_special_tokens=True)

        print(f"[{item['id']}] done ({completion_ids.shape[0]} completion tokens)")

        results.append({
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "response": reply,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(completion_ids.shape[0]),
        })

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = Path(args.output_dir) / f"{timestamp}_eval.md"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Hiraeth Eval Report\n\n")
        f.write(f"**Model:** {label}\n\n")
        f.write(f"**Generated:** {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("---\n\n")
        for r in results:
            f.write(f"## [{r['category']}] {r['id']}\n\n")
            f.write(f"**Prompt:** {r['prompt']}\n\n")
            f.write(f"**Response:**\n\n{r['response']}\n\n")
            f.write(f"*({r['prompt_tokens']} prompt tokens, {r['completion_tokens']} completion tokens)*\n\n")
            f.write("---\n\n")

    print(f"\nReport written to {report_path}")
    print("Diff this against reports from earlier checkpoints to see how answers change over training.")


if __name__ == "__main__":
    main()
