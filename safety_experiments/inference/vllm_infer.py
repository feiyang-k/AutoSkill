#!/usr/bin/env python
import argparse
import json
import os
import shutil
import subprocess
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="vLLM inference with optional temporary LoRA merge."
    )

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora_path", type=str, default=None)

    parser.add_argument("--input_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="adversarial_prompt",
    )

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=2048,
        help="Maximum total sequence length (prompt + generation).",
    )
    return parser.parse_args()


def load_data(path: str, prompt_field: str):
    with open(path) as f:
        data = json.load(f)

    for i, item in enumerate(data):
        if prompt_field not in item:
            raise ValueError(f"Missing field {prompt_field} at index {i}")

    return data


def build_prompt(tokenizer, text: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def main():
    args = parse_args()

    model_path = args.model
    merged_dir = None

    # ================================
    # TEMPORARY LoRA MERGE
    # ================================
    if args.lora_path is not None:
        merged_dir = args.lora_path.rstrip("/") + "_merged_tmp"

        if not os.path.exists(merged_dir):
            print(f"[INFO] Merging LoRA → {merged_dir}")
            subprocess.check_call([
                "python",
                "merge_lora_temp.py",
                "--base_model", args.model,
                "--lora_path", args.lora_path,
                "--output_dir", merged_dir,
            ])
        else:
            print(f"[INFO] Using existing merged dir: {merged_dir}")

        model_path = merged_dir

    # ================================
    # LOAD TOKENIZER & DATA
    # ================================
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = load_data(args.input_json, args.prompt_field)

    prompts = [
        build_prompt(tokenizer, item[args.prompt_field])
        for item in data
    ]

    # ================================
    # vLLM INFERENCE
    # ================================
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    outputs: List[str] = []
    for i in range(0, len(prompts), args.batch_size):
        res = llm.generate(prompts[i:i + args.batch_size], sampling_params)
        outputs.extend(
            r.outputs[0].text if r.outputs else "" for r in res
        )

    for item, out in zip(data, outputs):
        item["model_output"] = out

    with open(args.output_json, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # ================================
    # CLEANUP
    # ================================
    if merged_dir is not None:
        print(f"[CLEANUP] Removing temporary merged dir: {merged_dir}")
        shutil.rmtree(merged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
