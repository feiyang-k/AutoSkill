#!/usr/bin/env python3

import argparse
import json
import os
from typing import List

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =============================
# Reproducibility
# =============================

def set_seed(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================
# Data loading
# =============================

def load_prompts(json_path: str, column: str) -> List[str]:
    with open(json_path, "r") as f:
        data = json.load(f)

    prompts = []
    for i, row in enumerate(data):
        if column not in row:
            raise KeyError(f"Column '{column}' missing in row {i}")
        prompts.append(row[column])

    return prompts


# =============================
# Tokenization
# =============================

def encode_chat(tokenizer, query: str) -> torch.Tensor:
    """
    Single-turn chat prompt.
    Ends exactly where assistant generation would begin.
    """
    chat = [{"role": "user", "content": query}]
    return tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_tensors="pt",
    )


# =============================
# Activation extraction
# =============================

@torch.no_grad()
def get_last_token_activation(
    model,
    tokenizer,
    query: str,
    layer_idx: int,
) -> torch.Tensor:
    """
    Extract last-token hidden state at transformer block `layer_idx`.

    Returns:
        Tensor of shape (d_model,) on CPU (float32)
    """
    input_ids = encode_chat(tokenizer, query).to(model.device)
    last_token_idx = input_ids.shape[-1] - 1

    outputs = model(
        input_ids=input_ids,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    # hidden_states[0] = embeddings → drop it
    hidden_states = outputs.hidden_states[1:]

    h = hidden_states[layer_idx][0, last_token_idx, :]
    return h.detach().float().cpu()


# =============================
# Main
# =============================

def main():
    parser = argparse.ArgumentParser(
        description="Extract last-token activations from a HF causal LM"
    )
    parser.add_argument("--input_json", required=True, type=str)
    parser.add_argument("--column", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--layer_idx", type=int, default=15)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[INFO] Loading tokenizer + model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    print("[INFO] Loading prompts")
    prompts = load_prompts(args.input_json, args.column)
    print(f"[INFO] Number of prompts: {len(prompts)}")

    print(f"[INFO] Extracting layer {args.layer_idx} last-token activations")

    for i, prompt in enumerate(tqdm(prompts, desc="Extracting activations")):
        h = get_last_token_activation(
            model=model,
            tokenizer=tokenizer,
            query=prompt,
            layer_idx=args.layer_idx,
        )

        # ---- NO ZERO PADDING ----
        out_path = os.path.join(args.output_dir, f"activation_{i}.pt")
        torch.save(h, out_path)

    print(f"[DONE] Saved {len(prompts)} activations to {args.output_dir}")


if __name__ == "__main__":
    main()