#!/usr/bin/env python3
"""Run the vision track of the Adaptive Pedagogy Benchmark against a VLM.

Consumes the request files written by scripts/render_vision.py, each row of which
carries an `image_path` alongside the prompt:

  vision/all_120_vlm_requests.jsonl          task + inline object listing
  vision/all_120_perception_requests.jsonl   object listing only, no task

Two backends:

  --backend ollama   local server, same setup as scripts/run_ollama.py
  --backend openai   any OpenAI-compatible /chat/completions endpoint -- vLLM,
                     LM Studio, OpenRouter, or a hosted frontier model

Output rows carry exactly the fields scripts/score_ollama_runs.py expects, so a
vision run is scored by the unchanged text-run scorer:

  results_vision/raw/<condition>__<model_slug>__rep<NN>.jsonl
  python scripts/score_ollama_runs.py --outdir results_vision

Runs are resumable: re-running skips item_ids already present in the output.

Usage:
  python scripts/run_vlm.py --backend ollama \
      --models qwen2.5vl:7b llama3.2-vision:11b gemma3:12b --reps 3

  OPENAI_API_KEY=... python scripts/run_vlm.py --backend openai \
      --models gpt-4o --base-url https://api.openai.com/v1
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reused rather than reimplemented: extract_json_object already copes with
# <think> blocks, fenced code and trailing prose, which VLMs emit just as often.
from run_ollama import OllamaClient, extract_json_object, read_jsonl, slugify  # noqa: E402

DEFAULT_REQUESTS = "vision/all_120_vlm_requests.jsonl"


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class VisionOllamaClient(OllamaClient):
    """OllamaClient plus image attachment.

    Ollama takes images as an `images` list of bare base64 strings on the message
    itself -- no data: prefix, no content parts.
    """

    def chat_vision(self, model: str, messages: list[dict], image_b64: str,
                    options: dict, json_format: bool, keep_alive: str,
                    think: bool | None = None) -> dict:
        msgs = [dict(m) for m in messages]
        for m in reversed(msgs):  # attach to the last user turn
            if m.get("role") == "user":
                m["images"] = [image_b64]
                break
        else:
            raise ValueError("no user message to attach the image to")
        payload = {
            "model": model,
            "messages": msgs,
            "stream": False,
            "options": options,
            "keep_alive": keep_alive,
        }
        if json_format:
            payload["format"] = "json"
        if think is not None:
            payload["think"] = think
        return self._post("/api/chat", payload)


class OpenAIClient:
    """Minimal OpenAI-compatible chat client (stdlib only).

    Works against api.openai.com, OpenRouter, vLLM's server, LM Studio -- anything
    exposing /chat/completions with the image_url content part.
    """

    def __init__(self, base_url: str, api_key: str, timeout: float = 600.0):
        self.base = base_url.rstrip("/")
        self.key = api_key
        self.timeout = timeout

    def chat_vision(self, model: str, messages: list[dict], image_b64: str,
                    options: dict, json_format: bool, keep_alive: str,
                    think: bool | None = None) -> dict:
        msgs = []
        for m in messages:
            if m.get("role") == "user":
                msgs.append({"role": "user", "content": [
                    {"type": "text", "text": m["content"]},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"}},
                ]})
            else:
                msgs.append(dict(m))
        payload = {
            "model": model,
            "messages": msgs,
            "max_completion_tokens": options.get("num_predict"),
            "temperature": options.get("temperature"),
        }
        if json_format:
            payload["response_format"] = {"type": "json_object"}
        if options.get("seed") is not None:
            payload["seed"] = options["seed"]
        data = json.dumps({k: v for k, v in payload.items() if v is not None})
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=data.encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        # Normalise onto the Ollama response shape so one code path handles both.
        choice = (raw.get("choices") or [{}])[0]
        usage = raw.get("usage") or {}
        return {
            "message": {"content": (choice.get("message") or {}).get("content", "")},
            "eval_count": usage.get("completion_tokens"),
            "prompt_eval_count": usage.get("prompt_tokens"),
            "done_reason": choice.get("finish_reason"),
        }


# --------------------------------------------------------------------------

def load_image_b64(image_path: str, repo_root: Path, cache: dict) -> str:
    """base64 of one stimulus, cached -- reps re-send the same 120 images."""
    if image_path in cache:
        return cache[image_path]
    p = Path(image_path)
    if not p.is_absolute():
        p = repo_root / p
    if not p.exists():
        raise FileNotFoundError(
            f"stimulus {p} is missing. vision/ is gitignored; regenerate it with "
            f"`python scripts/render_vision.py`.")
    cache[image_path] = base64.b64encode(p.read_bytes()).decode("ascii")
    return cache[image_path]


def run_one(client, model: str, condition: str, rep: int,
            reqs: list[dict], out_path: Path, args) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            if "item_id" in row and "error" not in row:
                done.add(row["item_id"])
    todo = [r for r in reqs if r["item_id"] not in done]

    print(f"\n=== {model} | {condition} | rep {rep} === "
          f"{len(todo)} to run, {len(done)} already done -> {out_path}", flush=True)
    if not todo:
        return {"n_run": 0, "n_error": 0, "n_unparsed": 0}

    seed = args.seed + rep
    options = {
        "temperature": args.temperature,
        "num_predict": args.max_tokens,
        "num_ctx": args.num_ctx,
        "seed": seed,
    }
    think_flag = {"auto": None, "on": True, "off": False}[args.think]

    lock = threading.Lock()
    img_cache: dict[str, str] = {}
    counters = {"n_run": 0, "n_error": 0, "n_unparsed": 0}
    fh = out_path.open("a", encoding="utf-8")

    def work(idx_req):
        i, req = idx_req
        row = {
            "item_id": req["item_id"],
            "trial_type": req.get("trial_type"),
            "model_name": model,
            "condition": condition,
            "rep": rep,
            "seed": seed,
            "temperature": args.temperature,
            "image_path": req.get("image_path"),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                with lock:  # cache is shared across workers
                    b64 = load_image_b64(req["image_path"], args.repo_root, img_cache)
                t0 = time.time()
                resp = client.chat_vision(
                    model, req["messages"], b64, options,
                    json_format=not args.no_json_format,
                    keep_alive=args.keep_alive, think=think_flag)
                wall = time.time() - t0
                msg = resp.get("message") or {}
                raw = msg.get("content", "")
                thinking = msg.get("thinking", "") or ""
                parsed, how = extract_json_object(raw)
                if parsed is None and thinking:
                    parsed, how = extract_json_object(thinking)
                    if parsed is not None:
                        how = "from_thinking"
                row.update({
                    "response": parsed if (parsed and not args.no_extract) else raw,
                    "raw_response": raw,
                    "thinking": thinking,
                    "json_extraction": how,
                    "attempts": attempt,
                    "wall_seconds": round(wall, 3),
                    "eval_count": resp.get("eval_count"),
                    "prompt_eval_count": resp.get("prompt_eval_count"),
                    "total_duration_ns": resp.get("total_duration"),
                    "done_reason": resp.get("done_reason"),
                })
                if how == "failed" and attempt < args.retries:
                    last_err = f"unparseable response: {raw[:200]!r}"
                    time.sleep(args.retry_sleep)
                    continue
                break
            except FileNotFoundError:
                raise  # a missing stimulus is fatal, not retryable
            except (urllib.error.URLError, urllib.error.HTTPError,
                    TimeoutError, OSError, ValueError) as e:
                last_err = repr(e)
                if attempt < args.retries:
                    time.sleep(args.retry_sleep * attempt)
                    continue
                row.update({"error": last_err, "attempts": attempt})
        if "response" not in row and "error" not in row:
            row["error"] = last_err or "unknown failure"

        with lock:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            counters["n_run"] += 1
            if "error" in row:
                counters["n_error"] += 1
            elif row.get("json_extraction") == "failed":
                counters["n_unparsed"] += 1
            n = counters["n_run"]
            if n % args.log_every == 0 or n == len(todo):
                preview = str(row.get("response", row.get("error", "")))[:80]
                print(f"  [{n}/{len(todo)}] {row['item_id']} -> {preview!r}", flush=True)

    try:
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(work, enumerate(todo, 1)))
        else:
            for item in enumerate(todo, 1):
                work(item)
    finally:
        fh.close()

    print(f"  done: {counters['n_run']} calls, {counters['n_error']} errors, "
          f"{counters['n_unparsed']} unparseable", flush=True)
    return counters


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the vision track against a VLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    ap.add_argument("--requests", type=Path, default=None,
                    help=f"request file (default {DEFAULT_REQUESTS})")
    ap.add_argument("--condition", default="all120",
                    help="names the output file and picks the gold file in "
                         "score_ollama_runs.py; use 'perception' for the "
                         "perception-only probe, which that scorer skips")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "results_vision")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    # ollama
    ap.add_argument("--host", default="127.0.0.1:11434")
    ap.add_argument("--keep-alive", default="15m")
    ap.add_argument("--num-ctx", type=int, default=8192,
                    help="TOTAL window: prompt + image + generation, not the "
                         "generation budget (that is --max-tokens). Must exceed "
                         "--max-tokens plus the input or the budget is not real")
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto")
    # openai
    ap.add_argument("--base-url", default="https://api.openai.com/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="env var holding the key; the key is never logged")
    # shared
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="the object listing makes a full answer ~180 tokens, "
                         "well above the text run's needs")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--no-json-format", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    req_path = args.requests or (args.repo_root / DEFAULT_REQUESTS)
    if not req_path.exists():
        raise SystemExit(
            f"no request file at {req_path}. vision/ is gitignored; generate it "
            f"with `python scripts/render_vision.py`.")
    reqs = read_jsonl(req_path)
    if args.limit is not None:
        reqs = reqs[: args.limit]

    if args.backend == "ollama":
        client = VisionOllamaClient(args.host, timeout=args.timeout)
        print(f"waiting for ollama at {args.host} ...", flush=True)
        client.wait_until_ready()
        available = client.list_models()
        print(f"ollama ready, {len(available)} models visible", flush=True)
        if args.list_models:
            for m in sorted(available):
                print(f"  {m}")
            return 0

        def is_available(m: str) -> bool:
            return m in available or (":" not in m and f"{m}:latest" in available)

        missing = [m for m in args.models if not is_available(m)]
        if missing:
            # Fail before burning GPU time: compute nodes cannot pull.
            print(f"\nERROR: not in the model store: {', '.join(missing)}",
                  file=sys.stderr)
            for m in sorted(available):
                print(f"  {m}", file=sys.stderr)
            return 2
    else:
        import os
        key = os.environ.get(args.api_key_env, "")
        if not key:
            raise SystemExit(f"${args.api_key_env} is empty; export your API key")
        client = OpenAIClient(args.base_url, key, timeout=args.timeout)

    manifest = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": args.backend,
        "models": args.models,
        "condition": args.condition,
        "requests": str(req_path),
        "reps": args.reps,
        "temperature": args.temperature,
        "base_seed": args.seed,
        "max_tokens": args.max_tokens,
        "num_ctx": args.num_ctx,
        "json_format": not args.no_json_format,
        "think": args.think,
        "limit": args.limit,
        "n_items": len(reqs),
        "runs": [],
    }

    raw_dir = args.outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    for model in args.models:  # outermost, so each model loads into VRAM once
        for rep in range(1, args.reps + 1):
            out_path = raw_dir / f"{args.condition}__{slugify(model)}__rep{rep:02d}.jsonl"
            counters = run_one(client, model, args.condition, rep, reqs, out_path, args)
            manifest["runs"].append({
                "model": model, "condition": args.condition, "rep": rep,
                "predictions": str(out_path), **counters})

    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["elapsed_seconds"] = round(time.time() - t_start, 1)
    mpath = args.outdir / "run_manifest.json"
    if mpath.exists():
        prev = json.loads(mpath.read_text(encoding="utf-8"))
        manifest["history"] = prev.get("history", []) + [
            {k: v for k, v in prev.items() if k != "history"}]
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_err = sum(r["n_error"] for r in manifest["runs"])
    print(f"\nall runs finished in {manifest['elapsed_seconds']}s, "
          f"{total_err} errored calls. manifest -> {mpath}", flush=True)
    return 1 if total_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
