"""Generate TinyStories-style text from a trained Task 1 checkpoint.

The model config and tokenizer are rebuilt from the checkpoint itself, so generation
never depends on the current YAML. Samples go to a readable .txt file plus a .jsonl
file with timing and settings that evaluate.py uses for diversity metrics.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model import GPTConfig, GPTLanguageModel  # noqa: E402
from tokenizer import CharTokenizer  # noqa: E402
from utils import get_device, repo_relative, resolve_path, set_seed, utc_iso  # noqa: E402

DEFAULT_OUTPUT = "task1_llm/parth/outputs/generated_samples.txt"


def load_checkpoint(path: str | Path, device: torch.device):
    """Rebuild (model, tokenizer, checkpoint_dict) from a checkpoint written by train.py."""
    ckpt_path = resolve_path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {repo_relative(ckpt_path)}")
    # Why: weights_only=False is needed because the checkpoint also stores the config and
    # tokenizer dicts; these files are only ever produced by this project's train.py.
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tokenizer = CharTokenizer.from_dict(ckpt["tokenizer"])
    cfg = GPTConfig(**ckpt["model_config"])
    model = GPTLanguageModel(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer, ckpt


def decoding_label(greedy: bool, temperature: float, top_k: int | None) -> str:
    if greedy or temperature <= 0:
        return "greedy"
    return f"temperature={temperature},top_k={top_k if top_k else 'none'}"


def generate_text(model: GPTLanguageModel, tokenizer: CharTokenizer, prompt: str,
                  max_new_tokens: int, temperature: float, top_k: int | None,
                  greedy: bool, device: torch.device) -> dict:
    if not prompt:
        raise ValueError("Prompt must contain at least one character")
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)  # (1, T0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                         top_k=top_k, greedy=greedy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    new_ids = out[0, ids.size(1):].tolist()
    return {
        "prompt": prompt,
        "continuation": tokenizer.decode(new_ids),
        "text": tokenizer.decode(out[0].tolist()),
        "new_tokens": len(new_ids),
        "seconds": seconds,
        "tokens_per_sec": len(new_ids) / seconds if seconds > 0 else None,
        "unknown_prompt_chars": tokenizer.count_unknown(prompt),
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate text with the Task 1 character-level GPT")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", action="append", default=None,
                   help="Prompt text; pass several times for several prompts")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50, help="0 disables top-k filtering")
    p.add_argument("--greedy", action="store_true", help="Greedy decoding (ignores temperature/top_k)")
    p.add_argument("--num_samples", type=int, default=1, help="Samples per prompt")
    p.add_argument("--seed", type=int, default=8503)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--append", action="store_true", help="Append instead of overwriting outputs")
    return p.parse_args(argv)


def main(argv=None) -> list[dict]:
    args = parse_args(argv)
    device = get_device()
    model, tokenizer, ckpt = load_checkpoint(args.checkpoint, device)
    prompts = args.prompt or ["Once upon a time"]
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    label = decoding_label(args.greedy, args.temperature, top_k)

    records = []
    for p_idx, prompt in enumerate(prompts):
        for s_idx in range(args.num_samples):
            # Why: a distinct, derived seed per sample makes every sample individually
            # reproducible from the command line without them all being identical.
            sample_seed = args.seed + 1000 * p_idx + s_idx
            set_seed(sample_seed)
            result = generate_text(model, tokenizer, prompt, args.max_new_tokens,
                                   args.temperature, top_k, args.greedy, device)
            result.update({
                "run_id": ckpt.get("run_id"),
                "checkpoint": repo_relative(resolve_path(args.checkpoint)),
                "checkpoint_epoch": ckpt.get("epoch"),
                "decoding": label,
                "temperature": None if label == "greedy" else args.temperature,
                "top_k": None if label == "greedy" else top_k,
                "seed": sample_seed,
                "sample_index": s_idx,
                "device": device.type,
                "created_at": utc_iso(),
            })
            records.append(result)
            if result["unknown_prompt_chars"]:
                print(f"[warn] prompt has {result['unknown_prompt_chars']} character(s) outside the vocabulary")

    txt_path = resolve_path(args.output)
    jsonl_path = txt_path.with_suffix(".jsonl")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with open(txt_path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(
                f"===== run={r['run_id']} | ckpt={r['checkpoint']} (epoch {r['checkpoint_epoch']}) | "
                f"{r['decoding']} | seed={r['seed']} | new_tokens={r['new_tokens']} | "
                f"{r['tokens_per_sec']:.1f} tok/s =====\n"
            )
            f.write(f"PROMPT: {r['prompt']}\n")
            f.write(r["text"] + "\n\n")
    with open(jsonl_path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for r in records:
        print(f"\n--- {r['decoding']} | seed {r['seed']} | {r['tokens_per_sec']:.1f} tok/s ---")
        print(r["text"])
    print(f"\nWrote {len(records)} sample(s) to {repo_relative(txt_path)} and {repo_relative(jsonl_path)}")
    return records


if __name__ == "__main__":
    main()
