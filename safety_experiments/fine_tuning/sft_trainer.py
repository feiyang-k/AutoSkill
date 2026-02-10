#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed as hf_set_seed,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, TaskType


# ---------- Utils ----------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)   # <-- correct call
    hf_set_seed(seed)


class SavePerEpochCallback(TrainerCallback):
    """Save adapters + tokenizer at the end of each epoch: <output_dir>/<save_name>_epoch{N}"""
    def __init__(self, output_dir: str, save_name: str, tokenizer: AutoTokenizer):
        self.output_dir = output_dir
        self.save_name = save_name
        self.tokenizer = tokenizer
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num = int(state.epoch) if state.epoch is not None else 0
        tag = f"{self.save_name}_epoch{epoch_num}"
        out_dir = os.path.join(self.output_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        trainer = kwargs.get("trainer", None)
        model = kwargs.get("model", None) or (trainer.model if trainer is not None else None)
        if model is None:
            print("[SavePerEpochCallback] WARNING: model not found in callback kwargs.")
            return control

        model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        if trainer is not None:
            trainer.save_state()

        print(f"[SavePerEpochCallback] Saved checkpoint to: {out_dir}")
        return control


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("Simple LoRA SFT (epoch-based saving; flexible eval cadence)")

    # Data / model
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--eval_path", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, required=True)
    p.add_argument("--tokenizer_name_or_path", type=str, default=None)

    # Epoch-based training
    p.add_argument("--num_train_epochs", type=float, default=3)

    # Optimization
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--optim", type=str, default="adamw_torch")

    # Logging / evaluation / saving
    p.add_argument("--output_dir", type=str, default="./sft_output")
    p.add_argument("--save_name", type=str, required=True, help="Base name for epoch checkpoints")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_every_steps", type=int, default=None,
                   help="If set, run evaluation every N steps; otherwise evaluate at each epoch end.")
    p.add_argument("--report_to_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)

    # Precision
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--target_modules", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Misc
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ---------- Main ----------
def main():
    args = parse_args()
    set_seed(args.seed)

    # Optional W&B
    if args.report_to_wandb:
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    # Data (expects {"messages":[...]} per row)
    dataset = load_dataset(
        "json",
        data_files={"train": args.train_path, "validation": args.eval_path}
    )

    # Tokenizer
    tok_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Model
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        dtype=torch_dtype,
    )

    # LoRA
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()],
    )

    # Evaluation cadence: steps (if provided) else epoch
    eval_strategy = "steps" if (args.eval_every_steps and args.eval_every_steps > 0) else "epoch"

    # Trainer config (save ONLY via callback; evaluation per strategy above)
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,

        # <-- TRL expects 'eval_strategy' (not 'evaluation_strategy') in your version
        eval_strategy=eval_strategy,
        eval_steps=args.eval_every_steps,    # None is fine if strategy='epoch'
        save_strategy="no",                  # we save via epoch-end callback only

        report_to=(["wandb"] if args.report_to_wandb else []),
        bf16=args.bf16,
        fp16=(args.fp16 and not args.bf16),
        optim=args.optim,
        seed=args.seed,
    )

    # Trainer (messages-format handled via tokenizer chat template; dynamic padding)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,   # dynamic padding by default; no packing
        peft_config=peft_cfg,
    )

    # Save at end of each epoch as <save_name>_epochN
    trainer.add_callback(SavePerEpochCallback(args.output_dir, args.save_name, tokenizer))

    # Train
    trainer.train()

    # Final snapshot
    final_dir = os.path.join(args.output_dir, f"{args.save_name}_final")
    os.makedirs(final_dir, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    trainer.save_state()
    print(f"[FINAL] Saved to: {final_dir}")


if __name__ == "__main__":
    main()
