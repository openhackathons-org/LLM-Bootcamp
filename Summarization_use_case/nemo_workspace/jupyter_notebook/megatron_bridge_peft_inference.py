# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""Run inference from a Megatron-Bridge PEFT/LoRA fine-tuned checkpoint.

Megatron-Bridge PEFT training checkpoints store adapter weights rather than a
complete dense model. This script exports the adapter checkpoint to Hugging Face
PEFT format, loads the original HF base model, attaches the adapter, and runs
generation.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import torch
from megatron.bridge import AutoBridge
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ITER_RE = re.compile(r"iter_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-path", required=True, help="Original HF model ID or path.")
    parser.add_argument(
        "--megatron-checkpoint-path",
        required=True,
        type=Path,
        help="Megatron-Bridge PEFT checkpoint root or a specific iter_XXXXXXX directory.",
    )
    parser.add_argument("--checkpoint-step", type=int, help="Checkpoint step to load, e.g. 100.")
    parser.add_argument("--adapter-output-dir", type=Path, help="Directory for exported HF adapter.")
    parser.add_argument("--prompt", default=None, help="Prompt text. Use --prompt-file for long prompts.")
    parser.add_argument("--prompt-file", type=Path, help="File containing prompt text.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sample", action="store_true", help="Use sampling instead of greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--print-full-text", action="store_true")
    parser.add_argument("--tp", type=int, default=1, help="Accepted for command compatibility; unused.")
    parser.add_argument("--pp", type=int, default=1, help="Accepted for command compatibility; unused.")
    return parser.parse_args()


def iter_number(path: Path) -> int:
    match = ITER_RE.search(path.name)
    if not match:
        raise ValueError(f"Not an iteration checkpoint directory: {path}")
    return int(match.group(1))


def resolve_checkpoint(path: Path, checkpoint_step: int | None) -> Path:
    if checkpoint_step is not None:
        if ITER_RE.search(path.name):
            actual_step = iter_number(path)
            if actual_step == checkpoint_step:
                return path
            raise FileNotFoundError(
                f"Requested checkpoint step {checkpoint_step}, but path points to step {actual_step}: {path}"
            )
        candidate = path / f"iter_{checkpoint_step:07d}"
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Checkpoint step {checkpoint_step} was not found at {candidate}")

    if ITER_RE.search(path.name):
        if path.is_dir():
            return path
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    candidates = [candidate for candidate in path.glob("iter_*") if candidate.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No iter_* checkpoint directories found under {path}")
    return max(candidates, key=iter_number)


def default_adapter_dir(checkpoint_root: Path, iter_checkpoint: Path) -> Path:
    if checkpoint_root.name == "checkpoints":
        run_root = checkpoint_root.parent
    elif checkpoint_root.name.startswith("iter_") and checkpoint_root.parent.name == "checkpoints":
        run_root = checkpoint_root.parent.parent
    else:
        run_root = checkpoint_root
    return run_root / f"hf_adapter_{iter_checkpoint.name}"


def read_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return args.prompt_file.read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    raise ValueError("Provide --prompt or --prompt-file.")


def export_adapter_if_needed(args: argparse.Namespace, iter_checkpoint: Path, adapter_dir: Path) -> None:
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    if adapter_config.exists() and adapter_weights.exists():
        print(f"Using existing exported adapter: {adapter_dir}")
        return

    adapter_dir.mkdir(parents=True, exist_ok=True)
    bridge = AutoBridge.from_hf_pretrained(
        args.hf_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    print(f"Exporting Megatron-Bridge adapter {iter_checkpoint} -> {adapter_dir}")
    bridge.export_adapter_ckpt(iter_checkpoint, adapter_dir)


def main() -> None:
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        return

    args = parse_args()
    iter_checkpoint = resolve_checkpoint(args.megatron_checkpoint_path, args.checkpoint_step)
    adapter_dir = args.adapter_output_dir or default_adapter_dir(args.megatron_checkpoint_path, iter_checkpoint)
    prompt = os.environ["prompts"] #read_prompt(args)
    #print("prompt:", prompt)
    export_adapter_if_needed(args, iter_checkpoint, adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_path,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.sample:
        generate_kwargs.update(
            {
                "temperature": args.temperature,
                "top_k": args.top_k,
                "top_p": args.top_p,
            }
        )

    with torch.no_grad():
        generated = model.generate(**inputs, **generate_kwargs)

    if args.print_full_text:
        output_ids = generated[0]
    else:
        output_ids = generated[0][inputs["input_ids"].shape[-1] :]
    print("=" * 50)
    print("Prompt Output")
    print("=" * 50)
    print()
    print(tokenizer.decode(output_ids, skip_special_tokens=True).strip())
    print("=" * 50)
    

if __name__ == "__main__":
    main()
