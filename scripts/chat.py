"""
chat.py — quick manual test of Hiraeth after training.

Usage:
  python chat.py --model_dir /kaggle/working/hiraeth-merged
  # or point at the LoRA adapter dir directly if base_model is also given:
  python chat.py --base_model Qwen/Qwen2.5-7B-Instruct --adapter_dir /kaggle/working/hiraeth-qlora
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=None, help="Path to merged standalone model")
    ap.add_argument("--base_model", default=None, help="Base model id (if using adapter_dir)")
    ap.add_argument("--adapter_dir", default=None, help="LoRA adapter dir (if not merged)")
    ap.add_argument("--system_prompt", default="You are Hiraeth, a helpful, precise AI assistant.")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    if args.model_dir:
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, device_map="auto"
        )
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

    model.eval()
    messages = [{"role": "system", "content": args.system_prompt}]

    # Running totals for the whole session
    session_prompt_tokens = 0
    session_completion_tokens = 0

    print("Hiraeth is ready. Type 'exit' to quit.\n")
    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in ("exit", "quit"):
            break
        messages.append({"role": "user", "content": user_msg})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Prompt token count = length of the tokenized input (this
        # already includes system prompt + full conversation history,
        # since we re-tokenize the whole running "messages" list each turn)
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

        # Completion token count = everything generated beyond the prompt
        completion_ids = output_ids[0][prompt_tokens:]
        completion_tokens = completion_ids.shape[0]

        reply = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Note: prompt_tokens grows every turn because the full running
        # conversation is re-tokenized each time (same as how hosted chat
        # APIs bill it) — this is NOT double counting, it reflects that
        # every turn re-sends the whole context to the model.
        session_prompt_tokens += prompt_tokens
        session_completion_tokens += completion_tokens
        total_tokens = prompt_tokens + completion_tokens

        print(f"Hiraeth: {reply}\n")
        print(
            f"  [tokens] prompt: {prompt_tokens} | completion: {completion_tokens} "
            f"| turn total: {total_tokens} | session total: "
            f"{session_prompt_tokens + session_completion_tokens}\n"
        )
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
