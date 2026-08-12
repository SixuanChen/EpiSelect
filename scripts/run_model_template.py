#!/usr/bin/env python3
"""Minimal provider-neutral runner template.

Edit ONLY call_your_model(messages) to connect your model. Everything else handles
request loading, one-independent-call-per-item execution, raw response storage,
and resumability.

Usage:
  python scripts/run_model_template.py \
    --requests requests/all_120_requests.jsonl \
    --out results/my_model_predictions.jsonl \
    --model-name my-model
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path


def call_your_model(messages, model_name: str) -> str:
    # ---------------------------------------------------------------
    # REPLACE THIS FUNCTION ONLY.
    # messages is a list like:
    #   [{"role":"system","content":"..."}, {"role":"user","content":"..."}]
    # Return the model's raw text response, ideally a JSON object string.
    # IMPORTANT: each benchmark item should be a fresh/stateless model call.
    # ---------------------------------------------------------------
    raise NotImplementedError("Connect your model/API inside call_your_model().")


def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    reqs = read_jsonl(args.requests)
    if args.limit is not None:
        reqs = reqs[:args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.out.exists():
        for x in read_jsonl(args.out):
            if "item_id" in x:
                done.add(x["item_id"])

    with args.out.open("a", encoding="utf-8") as f:
        for i, req in enumerate(reqs, 1):
            if req["item_id"] in done:
                continue
            try:
                raw = call_your_model(req["messages"], args.model_name)
                row = {
                    "item_id": req["item_id"],
                    "trial_type": req.get("trial_type"),
                    "model_name": args.model_name,
                    "response": raw,
                }
            except Exception as e:
                row = {
                    "item_id": req["item_id"],
                    "trial_type": req.get("trial_type"),
                    "model_name": args.model_name,
                    "error": repr(e),
                }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(reqs)}] {req['item_id']}")
            if args.sleep:
                time.sleep(args.sleep)

if __name__ == "__main__":
    main()
