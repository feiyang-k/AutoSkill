#!/usr/bin/env python
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser("Temporary LoRA merge")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[MERGE] Base model: {args.base_model}")
    print(f"[MERGE] LoRA path : {args.lora_path}")
    print(f"[MERGE] Output dir: {args.output_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, args.lora_path)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("[MERGE] Done.")


if __name__ == "__main__":
    main()