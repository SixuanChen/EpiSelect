#!/usr/bin/env python3
"""Optional local Hugging Face runner for instruction-tuned causal LMs.

Example (after installing torch + transformers):
  python scripts/run_hf_transformers.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --requests requests/all_120_requests.jsonl \
    --out results/qwen25_7b.jsonl

This is an optional convenience runner. Generation/scoring themselves use only
Python's standard library.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    model.eval()
    reqs = read_jsonl(args.requests)
    if args.limit is not None:
        reqs = reqs[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for i, req in enumerate(reqs, 1):
            text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            new = out[0, inputs["input_ids"].shape[1]:]
            raw = tok.decode(new, skip_special_tokens=True).strip()
            f.write(json.dumps({
                "item_id": req["item_id"], "trial_type": req.get("trial_type"),
                "model_name": args.model, "response": raw
            }, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(reqs)}] {req['item_id']} -> {raw[:100]!r}")

if __name__ == "__main__":
    main()
