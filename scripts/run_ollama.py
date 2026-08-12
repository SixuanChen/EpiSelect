#!/usr/bin/env python3
"""Run the Adaptive Pedagogy Benchmark against Ollama models on Oscar.

Talks to a locally running `ollama serve` over its HTTP API (stdlib only, no
`requests`/`ollama` package needed). Each benchmark item is one independent,
stateless chat call, as the benchmark requires.

Sweeps models x conditions x repetitions and writes one JSONL per combination:

  results/raw/<condition>__<model_slug>__rep01.jsonl

Each row carries the fields score_predictions.py expects (`item_id`,
`response`) plus run metadata (timings, token counts, raw text, seed), so the
same file is both the prediction file and the audit trail.

Runs are resumable: re-running skips item_ids already present in the output.

Usage:
  python scripts/run_ollama.py \
      --models llama3.1:8b qwen3:8b gemma3:12b \
      --conditions main controls \
      --reps 3 \
      --temperature 0.0 \
      --outdir results
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# condition name -> requests file (relative to repo root).
# Gold files / scorers for each condition live in scripts/score_ollama_runs.py.
CONDITIONS: dict[str, str] = {
    "main": "requests/main_100_joint_requests.jsonl",
    "all120": "requests/all_120_requests.jsonl",
    "controls_diagnosis": "requests/controls_no_history_diagnosis_10_requests.jsonl",
    "controls_choice": "requests/controls_no_history_choice_10_requests.jsonl",
    "ablation_choice_only": "requests/optional_ablation_choice_only_100_requests.jsonl",
    "ablation_diagnosis_only": "requests/optional_ablation_diagnosis_only_100_requests.jsonl",
    "ablation_action_given_rule": "requests/optional_ablation_action_given_rule_100_requests.jsonl",
}

# Convenience aliases expanding to several conditions.
CONDITION_GROUPS: dict[str, list[str]] = {
    "controls": ["controls_diagnosis", "controls_choice"],
    "ablations": [
        "ablation_choice_only",
        "ablation_diagnosis_only",
        "ablation_action_given_rule",
    ],
}

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# --------------------------------------------------------------------------
# Ollama HTTP client
# --------------------------------------------------------------------------

class OllamaClient:
    def __init__(self, host: str, timeout: float = 600.0):
        self.base = host if host.startswith("http") else f"http://{host}"
        self.timeout = timeout

    def _post(self, path: str, payload: dict, timeout: float | None = None) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _get(self, path: str, timeout: float | None = None) -> dict:
        req = urllib.request.Request(f"{self.base}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout or 30.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def wait_until_ready(self, attempts: int = 60, delay: float = 2.0) -> None:
        last = None
        for _ in range(attempts):
            try:
                self._get("/api/tags", timeout=5.0)
                return
            except Exception as e:  # server not up yet
                last = e
                time.sleep(delay)
        raise RuntimeError(f"Ollama server at {self.base} never became ready: {last!r}")

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._get("/api/tags").get("models", [])]

    def pull(self, model: str) -> None:
        """Blocking pull (non-streaming); only needed for models not already cached."""
        self._post("/api/pull", {"model": model, "stream": False}, timeout=3600.0)

    def chat(self, model: str, messages: list[dict], options: dict,
             json_format: bool, keep_alive: str, think: bool | None = None) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": keep_alive,
        }
        if json_format:
            payload["format"] = "json"
        if think is not None:
            # Thinking models (qwen3, deepseek-r1, ...) emit reasoning into a
            # separate `message.thinking` field. Left on, it can consume the
            # whole num_predict budget before `message.content` ever starts,
            # yielding done_reason="length" with empty content.
            payload["think"] = think
        return self._post("/api/chat", payload)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def read_jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def slugify(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def extract_json_object(text: str) -> tuple[str | None, str]:
    """Return (json_string, how) where how is exact | extracted | failed.

    Reasoning models wrap the answer in prose or <think> blocks; the scorer
    needs a bare JSON object string, so pull out the first balanced-looking
    object when the whole response is not already valid JSON.
    """
    s = (text or "").strip()
    if not s:
        return None, "failed"
    try:
        if isinstance(json.loads(s), dict):
            return s, "exact"
    except json.JSONDecodeError:
        pass
    cleaned = THINK_RE.sub("", s).strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    for candidate in (cleaned, s):
        m = JSON_OBJ_RE.search(candidate)
        if not m:
            continue
        blob = m.group(0)
        # Trim greedily from the right until it parses (handles trailing prose).
        while blob:
            try:
                if isinstance(json.loads(blob), dict):
                    return blob, "extracted"
            except json.JSONDecodeError:
                pass
            cut = blob.rfind("}")
            blob = blob[:cut] if cut > 0 else ""
    return None, "failed"


def expand_conditions(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for c in CONDITION_GROUPS.get(n, [n]):
            if c not in CONDITIONS:
                raise SystemExit(
                    f"unknown condition {c!r}; choose from "
                    f"{sorted(CONDITIONS)} or groups {sorted(CONDITION_GROUPS)}"
                )
            if c not in out:
                out.append(c)
    return out


# --------------------------------------------------------------------------
# One (model, condition, rep) sweep
# --------------------------------------------------------------------------

def run_one(client: OllamaClient, model: str, condition: str, rep: int,
            requests_path: Path, out_path: Path, args) -> dict:
    reqs = read_jsonl(requests_path)
    if args.limit is not None:
        reqs = reqs[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            if "item_id" in row and "error" not in row:
                done.add(row["item_id"])
    todo = [r for r in reqs if r["item_id"] not in done]

    print(f"\n=== {model} | {condition} | rep {rep} === "
          f"{len(todo)} to run, {len(done)} already done -> {out_path}",
          flush=True)
    if not todo:
        return {"n_run": 0, "n_error": 0, "n_unparsed": 0}

    # Distinct but reproducible seed per repetition.
    seed = args.seed + rep
    options = {
        "temperature": args.temperature,
        "num_predict": args.max_tokens,
        "num_ctx": args.num_ctx,
        "seed": seed,
    }
    if args.top_p is not None:
        options["top_p"] = args.top_p

    # None => omit the field entirely (server default); True/False => send it.
    think_flag = {"auto": None, "on": True, "off": False}[args.think]

    lock = threading.Lock()
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
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        last_err = None
        for attempt in range(1, args.retries + 1):
            try:
                t0 = time.time()
                resp = client.chat(model, req["messages"], options,
                                   json_format=not args.no_json_format,
                                   keep_alive=args.keep_alive,
                                   think=think_flag)
                wall = time.time() - t0
                msg = resp.get("message") or {}
                raw = msg.get("content", "")
                thinking = msg.get("thinking", "") or ""
                parsed, how = extract_json_object(raw)
                if parsed is None and thinking:
                    # Never silently lose an answer that landed in the reasoning
                    # channel instead of the content field.
                    parsed, how = extract_json_object(thinking)
                    if parsed is not None:
                        how = "from_thinking"
                row.update({
                    # `response` is what score_predictions.py reads.
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
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
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


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run the Adaptive Pedagogy Benchmark against Ollama models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--models", nargs="+", required=True,
                    help="Ollama model tags, e.g. llama3.1:8b qwen3:8b gemma3:12b")
    ap.add_argument("--conditions", nargs="+", default=["main"],
                    help=f"any of {sorted(CONDITIONS)} or groups {sorted(CONDITION_GROUPS)}")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per model x condition")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--host", default="127.0.0.1:11434")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1000, help="base seed; rep k uses seed+k")
    ap.add_argument("--max-tokens", type=int, default=256, help="options.num_predict")
    ap.add_argument("--num-ctx", type=int, default=4096)
    ap.add_argument("--keep-alive", default="15m", help="how long Ollama keeps the model resident")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent requests; >1 needs OLLAMA_NUM_PARALLEL set accordingly")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout (s)")
    ap.add_argument("--limit", type=int, default=None, help="only first N items (smoke test)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto",
                    help="thinking models: 'off' disables the reasoning channel so the "
                         "num_predict budget goes to the answer; 'auto' omits the field "
                         "and takes the server default")
    ap.add_argument("--no-json-format", action="store_true",
                    help="do not send Ollama's format=json constrained decoding")
    ap.add_argument("--no-extract", action="store_true",
                    help="store the raw text as `response` instead of the extracted JSON object")
    ap.add_argument("--pull-missing", action="store_true",
                    help="pull models absent from OLLAMA_MODELS (needs network + write access; "
                         "compute nodes have no outbound network, so run this from a login node)")
    ap.add_argument("--list-models", action="store_true",
                    help="print the models the server can see, then exit")
    args = ap.parse_args()

    conditions = expand_conditions(args.conditions)
    client = OllamaClient(args.host, timeout=args.timeout)

    print(f"waiting for ollama at {args.host} ...", flush=True)
    client.wait_until_ready()
    available = client.list_models()
    print(f"ollama ready, {len(available)} models visible", flush=True)

    if args.list_models:
        for m in sorted(available):
            print(f"  {m}")
        return 0

    # Ollama reports "llama3.1:8b"; accept a bare "llama3.1" as ":latest".
    def is_available(m: str) -> bool:
        return m in available or (":" not in m and f"{m}:latest" in available)

    missing = [m for m in args.models if not is_available(m)]
    if missing and args.pull_missing:
        for m in missing:
            print(f"pulling {m} ...", flush=True)
            client.pull(m)
        available = client.list_models()
        missing = [m for m in args.models if not is_available(m)]
    if missing:
        # Fail before burning GPU time: compute nodes cannot pull, so every call
        # for a model outside the store would just error 120 times over.
        print(f"\nERROR: {len(missing)} requested model(s) are not in the model store "
              f"({', '.join(missing)}).", file=sys.stderr)
        print("Compute nodes have no outbound network, so these cannot be pulled here.",
              file=sys.stderr)
        print("Available models:", file=sys.stderr)
        for m in sorted(available):
            print(f"  {m}", file=sys.stderr)
        return 2

    raw_dir = args.outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": args.models,
        "conditions": conditions,
        "reps": args.reps,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "base_seed": args.seed,
        "max_tokens": args.max_tokens,
        "num_ctx": args.num_ctx,
        "json_format": not args.no_json_format,
        "think": args.think,
        "limit": args.limit,
        "host": args.host,
        "runs": [],
    }

    t_start = time.time()
    # Model outermost so each model is loaded into GPU memory only once.
    for model in args.models:
        for condition in conditions:
            requests_path = args.repo_root / CONDITIONS[condition]
            for rep in range(1, args.reps + 1):
                out_path = raw_dir / f"{condition}__{slugify(model)}__rep{rep:02d}.jsonl"
                counters = run_one(client, model, condition, rep,
                                   requests_path, out_path, args)
                manifest["runs"].append({
                    "model": model, "condition": condition, "rep": rep,
                    "requests": str(requests_path),
                    "predictions": str(out_path), **counters,
                })

    manifest["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest["elapsed_seconds"] = round(time.time() - t_start, 1)
    manifest_path = args.outdir / "run_manifest.json"
    if manifest_path.exists():  # keep prior manifests from earlier sweeps
        prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        history = prev.get("history", []) + [{k: v for k, v in prev.items() if k != "history"}]
        manifest["history"] = history
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    total_err = sum(r["n_error"] for r in manifest["runs"])
    print(f"\nall runs finished in {manifest['elapsed_seconds']}s, "
          f"{total_err} errored calls. manifest -> {manifest_path}", flush=True)
    return 1 if total_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
