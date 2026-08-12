#!/usr/bin/env python3
"""Convert public benchmark JSONL into provider-neutral, deterministically shuffled chat requests."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

SYSTEM = "Follow the user's instructions exactly. Return only the requested JSON object and no other text."

def read_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]

def shuffle_key(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--shuffle-seed", type=int, default=20260811,
                    help="Deterministic request order; set negative to keep source order.")
    args = ap.parse_args()
    rows = read_jsonl(args.input)
    if args.shuffle_seed >= 0:
        rows = sorted(rows, key=lambda r: shuffle_key(r["item_id"], args.shuffle_seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in rows:
            req = {
                "item_id": r["item_id"],
                "trial_type": r["trial_type"],
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["prompt"]},
                ],
            }
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} requests -> {args.output}")

if __name__ == "__main__":
    main()
