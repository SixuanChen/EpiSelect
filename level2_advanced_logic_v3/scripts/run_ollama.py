#!/usr/bin/env python3
"""Run EpiSelect Level 2 Advanced Logic v3 against Ollama models on Oscar.

Talks to a locally running `ollama serve` over its HTTP API (stdlib only), the
same way the v4 adaptive-pedagogy runner does, and keeps the same experiment
knobs: models x suites x repetitions, temperature 0, seed 1000+rep,
num_predict 4096, num_ctx 8192, format=json.

Level 2 differs from v4 in one important way: the primary protocol is STAGED.
Per context there is one role-neutral Stage-1 inference call, and its exact
response is replayed as the assistant turn into two independent Stage-2
branches (Teacher and Imposter). Stage 1 is therefore called ONCE per context,
never once per role -- calling it twice would let the two branches disagree
about the inferred rule and destroy the matched counterfactual control.

Writes one JSONL per (suite, model, rep):

  <outdir>/raw/<suite>__<model_slug>__rep01.jsonl

Staged rows carry `raw_inference_response` and `raw_action_response`; flat rows
carry `response`. Those are exactly the fields score_predictions.py,
score_naturalistic.py and score_ablations.py read, so each file is both the
prediction file and the audit trail.

Runs are resumable: re-running skips item_ids already present in the output.

Usage:
  python scripts/run_ollama.py \
      --models llama3.1:8b qwen3:8b gemma3:12b \
      --suites primary null nohistory \
      --reps 3 --temperature 0.0 --max-tokens 4096 --num-ctx 8192 \
      --outdir results_maxtok4096_temp0
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# suite name -> (requests file relative to repo root, protocol)
# "staged" files hold `inference_messages` + `branches`; "flat" files hold
# `messages` and one item_id. Gold files and scorers live in
# scripts/score_ollama_runs.py.
SUITES: dict[str, tuple[str, str]] = {
    "primary":   ("requests/primary_staged_120_contexts.jsonl", "staged"),
    "null":      ("requests/naturalistic_null_120_requests.jsonl", "flat"),
    "nohistory": ("requests/nohistory_staged_12_contexts.jsonl", "staged"),
    # ablations
    "abl_action_given_gold":    ("requests/ablations/action_given_gold.jsonl", "flat"),
    "abl_joint_single_call":    ("requests/ablations/joint_single_call.jsonl", "flat"),
    "abl_choice_only":          ("requests/ablations/choice_only.jsonl", "flat"),
    "abl_all_truthful_null":    ("requests/ablations/all_truthful_null.jsonl", "flat"),
    "abl_closed_hypothesis":    ("requests/ablations/closed_hypothesis_staged.jsonl", "staged"),
    "abl_neutral_goals":        ("requests/ablations/neutral_goals_staged.jsonl", "staged"),
    "abl_naturalistic_staged":  ("requests/ablations/naturalistic_staged.jsonl", "staged"),
    "abl_underdetermined":      ("requests/ablations/underdetermined_staged.jsonl", "staged"),
    "abl_all_truthful_staged":  ("requests/ablations/all_truthful_staged.jsonl", "staged"),
}

# Convenience aliases expanding to several suites.
SUITE_GROUPS: dict[str, list[str]] = {
    "main": ["primary", "null", "nohistory"],
    "ablations": ["abl_action_given_gold", "abl_neutral_goals", "abl_underdetermined",
                  "abl_joint_single_call", "abl_choice_only", "abl_closed_hypothesis",
                  "abl_naturalistic_staged", "abl_all_truthful_staged",
                  "abl_all_truthful_null"],
    "all": ["primary", "null", "nohistory",
            "abl_action_given_gold", "abl_neutral_goals", "abl_underdetermined",
            "abl_joint_single_call", "abl_choice_only", "abl_closed_hypothesis",
            "abl_naturalistic_staged", "abl_all_truthful_staged",
            "abl_all_truthful_null"],
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
        """Blocking pull (non-streaming); only for models not already cached."""
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

    Reasoning models wrap the answer in prose or <think> blocks; the scorers
    need a bare JSON object string, so pull out the first balanced-looking
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


def expand_suites(names: list[str]) -> list[str]:
    out: list[str] = []
    for n in names:
        for s in SUITE_GROUPS.get(n, [n]):
            if s not in SUITES:
                raise SystemExit(
                    f"unknown suite {s!r}; choose from {sorted(SUITES)} "
                    f"or groups {sorted(SUITE_GROUPS)}"
                )
            if s not in out:
                out.append(s)
    return out


# --------------------------------------------------------------------------
# One model call, with retries
# --------------------------------------------------------------------------

def call(client: OllamaClient, model: str, messages: list[dict], options: dict,
         args, think_flag) -> dict:
    """One chat call. Returns a dict of response fields; never raises."""
    last_err = None
    for attempt in range(1, args.retries + 1):
        try:
            t0 = time.time()
            resp = client.chat(model, messages, options,
                               json_format=not args.no_json_format,
                               keep_alive=args.keep_alive, think=think_flag)
            wall = time.time() - t0
            msg = resp.get("message") or {}
            raw = msg.get("content", "") or ""
            thinking = msg.get("thinking", "") or ""
            parsed, how = extract_json_object(raw)
            if parsed is None and thinking:
                # Never silently lose an answer that landed in the reasoning
                # channel instead of the content field.
                parsed, how = extract_json_object(thinking)
                if parsed is not None:
                    how = "from_thinking"
            out = {
                "json": parsed,
                "raw": raw,
                "thinking": thinking,
                "extraction": how,
                "attempts": attempt,
                "wall_seconds": round(wall, 3),
                "eval_count": resp.get("eval_count"),
                "prompt_eval_count": resp.get("prompt_eval_count"),
                "done_reason": resp.get("done_reason"),
            }
            if how == "failed" and attempt < args.retries:
                last_err = f"unparseable response: {raw[:200]!r}"
                time.sleep(args.retry_sleep)
                continue
            return out
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = repr(e)
            if attempt < args.retries:
                time.sleep(args.retry_sleep * attempt)
                continue
    return {"json": None, "raw": "", "thinking": "", "extraction": "failed",
            "attempts": args.retries, "error": last_err or "unknown failure"}


def assistant_turn(inf: dict) -> str:
    """What to replay as the Stage-1 assistant message in both branches.

    The model's own content verbatim, so the branches see exactly what it said.
    Falls back to the JSON recovered from the reasoning channel when content
    came back empty, which is the only case where verbatim would be a lie.
    """
    return inf["raw"] if inf["raw"].strip() else (inf["json"] or "")


# --------------------------------------------------------------------------
# One (model, suite, rep) sweep
# --------------------------------------------------------------------------

def run_one(client: OllamaClient, model: str, suite: str, rep: int,
            requests_path: Path, out_path: Path, args,
            reuse: dict[str, str] | None = None) -> dict:
    protocol = SUITES[suite][1]
    reqs = read_jsonl(requests_path)
    if args.limit is not None:
        reqs = reqs[: args.limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        for row in read_jsonl(out_path):
            if "item_id" in row and "error" not in row:
                done.add(row["item_id"])

    if protocol == "staged":
        todo = [r for r in reqs
                if any(b["item_id"] not in done for b in r["branches"])]
        n_units, unit = len(todo), "contexts"
    else:
        todo = [r for r in reqs if r["item_id"] not in done]
        n_units, unit = len(todo), "items"

    print(f"\n=== {model} | {suite} | rep {rep} === {n_units} {unit} to run, "
          f"{len(done)} items already done -> {out_path}", flush=True)
    if not todo:
        return {"n_calls": 0, "n_rows": 0, "n_error": 0, "n_unparsed": 0}

    seed = args.seed + rep                     # distinct but reproducible per rep
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

    counters = {"n_calls": 0, "n_rows": 0, "n_error": 0, "n_unparsed": 0}
    stamp = lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Written serially: the staged protocol is sequential within a context
    # anyway, and one resident model per GPU is the whole point of WORKERS=1.
    with out_path.open("a", encoding="utf-8") as fh:
        for i, req in enumerate(todo, 1):
            if protocol == "staged":
                ctx = req["context_id"]
                # Stage 1: exactly one call per context, role not yet revealed.
                if req.get("reuse_primary_inference") and reuse and ctx in reuse:
                    inf = {"json": reuse[ctx], "raw": reuse[ctx], "thinking": "",
                           "extraction": "reused", "attempts": 0, "wall_seconds": 0.0}
                else:
                    inf = call(client, model, req["inference_messages"], options,
                               args, think_flag)
                    counters["n_calls"] += 1
                inf_msg = assistant_turn(inf)

                for br in req["branches"]:
                    if br["item_id"] in done:
                        continue
                    msgs = (list(req["inference_messages"])
                            + [{"role": "assistant", "content": inf_msg},
                               {"role": "user", "content": br["message"]}])
                    act = call(client, model, msgs, options, args, think_flag)
                    counters["n_calls"] += 1
                    row = {
                        "item_id": br["item_id"],
                        "context_id": ctx,
                        "role": br.get("role"),
                        "suite": suite,
                        "request_type": req.get("request_type"),
                        "model_name": model,
                        "rep": rep,
                        "seed": seed,
                        "temperature": args.temperature,
                        "timestamp": stamp(),
                        # These two are what the scorers read.
                        "raw_inference_response": inf["json"] if inf["json"] else inf["raw"],
                        "raw_action_response": act["json"] if act["json"] else act["raw"],
                        # Full audit trail.
                        "inference_raw_text": inf["raw"],
                        "inference_thinking": inf["thinking"],
                        "inference_extraction": inf["extraction"],
                        "inference_wall_seconds": inf.get("wall_seconds"),
                        "inference_eval_count": inf.get("eval_count"),
                        "inference_done_reason": inf.get("done_reason"),
                        "action_raw_text": act["raw"],
                        "action_thinking": act["thinking"],
                        "action_extraction": act["extraction"],
                        "action_wall_seconds": act.get("wall_seconds"),
                        "action_eval_count": act.get("eval_count"),
                        "action_done_reason": act.get("done_reason"),
                        "attempts": inf.get("attempts", 0) + act.get("attempts", 0),
                    }
                    err = inf.get("error") or act.get("error")
                    if err:
                        row["error"] = err
                    _write(fh, row, counters,
                           unparsed=(inf["extraction"] == "failed"
                                     or act["extraction"] == "failed"))
            else:
                out = call(client, model, req["messages"], options, args, think_flag)
                counters["n_calls"] += 1
                row = {
                    "item_id": req["item_id"],
                    "context_id": req.get("context_id", req["item_id"]),
                    "suite": suite,
                    "request_type": req.get("request_type"),
                    "model_name": model,
                    "rep": rep,
                    "seed": seed,
                    "temperature": args.temperature,
                    "timestamp": stamp(),
                    # `response` is what the flat scorers read.
                    "response": out["json"] if out["json"] else out["raw"],
                    "raw_response": out["raw"],
                    "thinking": out["thinking"],
                    "json_extraction": out["extraction"],
                    "attempts": out.get("attempts"),
                    "wall_seconds": out.get("wall_seconds"),
                    "eval_count": out.get("eval_count"),
                    "prompt_eval_count": out.get("prompt_eval_count"),
                    "done_reason": out.get("done_reason"),
                }
                if out.get("error"):
                    row["error"] = out["error"]
                _write(fh, row, counters, unparsed=(out["extraction"] == "failed"))

            if i % args.log_every == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)} {unit}] {counters['n_rows']} rows, "
                      f"{counters['n_calls']} calls, {counters['n_error']} errors, "
                      f"{counters['n_unparsed']} unparseable", flush=True)

    print(f"  done: {counters['n_calls']} calls -> {counters['n_rows']} rows, "
          f"{counters['n_error']} errors, {counters['n_unparsed']} unparseable",
          flush=True)
    return counters


def _write(fh, row: dict, counters: dict, unparsed: bool) -> None:
    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    fh.flush()
    counters["n_rows"] += 1
    if "error" in row:
        counters["n_error"] += 1
    elif unparsed:
        counters["n_unparsed"] += 1


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run EpiSelect Level 2 v3 against Ollama models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--models", nargs="+", required=True,
                    help="Ollama model tags, e.g. llama3.1:8b qwen3:8b gemma3:12b")
    ap.add_argument("--suites", nargs="+", default=["main"],
                    help=f"any of {sorted(SUITES)} or groups {sorted(SUITE_GROUPS)}")
    ap.add_argument("--reps", type=int, default=1, help="repetitions per model x suite")
    ap.add_argument("--outdir", type=Path, default=REPO_ROOT / "results")
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--host", default="127.0.0.1:11434")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--seed", type=int, default=1000, help="base seed; rep k uses seed+k")
    ap.add_argument("--max-tokens", type=int, default=4096, help="options.num_predict")
    ap.add_argument("--num-ctx", type=int, default=8192)
    ap.add_argument("--keep-alive", default="15m")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=600.0, help="per-request timeout (s)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only first N contexts/items per suite (smoke test)")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--think", choices=["auto", "on", "off"], default="auto",
                    help="thinking models: 'off' disables the reasoning channel so the "
                         "num_predict budget goes to the answer")
    ap.add_argument("--no-json-format", action="store_true",
                    help="do not send Ollama's format=json constrained decoding")
    ap.add_argument("--pull-missing", action="store_true",
                    help="pull models absent from OLLAMA_MODELS (login node only; "
                         "compute nodes have no outbound network)")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    suites = expand_suites(args.suites)
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
        # for a model outside the store would just error hundreds of times over.
        print(f"\nERROR: {len(missing)} requested model(s) are not in the model store "
              f"({', '.join(missing)}).", file=sys.stderr)
        print("Compute nodes have no outbound network, so these cannot be pulled here.",
              file=sys.stderr)
        for m in sorted(available):
            print(f"  {m}", file=sys.stderr)
        return 2

    raw_dir = args.outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark": "EpiSelect_Level2_AdvancedLogic_v3",
        "backend": "ollama",
        "models": args.models,
        "suites": suites,
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
        for suite in suites:
            requests_path = args.repo_root / SUITES[suite][0]
            for rep in range(1, args.reps + 1):
                out_path = raw_dir / f"{suite}__{slugify(model)}__rep{rep:02d}.jsonl"
                # Staged ablations flagged `reuse_primary_inference` replay the
                # Stage-1 answer from the primary run of the same model+rep, so
                # the framing manipulation is the only thing that differs.
                reuse = None
                primary_path = raw_dir / f"primary__{slugify(model)}__rep{rep:02d}.jsonl"
                if primary_path.exists():
                    reuse = {r["context_id"]: r["raw_inference_response"]
                             for r in read_jsonl(primary_path)
                             if r.get("context_id") and r.get("raw_inference_response")}
                counters = run_one(client, model, suite, rep, requests_path,
                                   out_path, args, reuse)
                manifest["runs"].append({
                    "model": model, "suite": suite, "rep": rep,
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
          f"{total_err} errored rows. manifest -> {manifest_path}", flush=True)
    return 1 if total_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
