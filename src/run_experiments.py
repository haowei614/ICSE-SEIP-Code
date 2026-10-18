"""Driver: task x dataset x model x precision, with full per-run logging.

Integrity:
  - --dry-run (or --backend mock) uses MockBackend, writes ONLY to results/dry_run/,
    tags every record mock=True, and never scores.
  - Real backends write to results/final/ (or --out-dir).
  - NO run is silently discarded: malformed / OOM / timeout / exception are recorded
    as flags on the RunRecord and still written to the log.

Scheduling obeys config `run_order` so the hard reasoning column can run first,
letting Pillar-2 coding start early.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

import yaml

from .backends import build_backend
from .metrics import assert_single_basis, score
from .provenance import (
    cell_identity, config_hash, file_sha256, host_provenance, load_jsonl,
)
from .schema import RunRecord, sha256
from .sampling import stratified_sample
from . import tasks as tasksmod


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--backend", default="mock",
                   help="mock (dry-run) | transformers | llama_cpp | cloud")
    p.add_argument("--dry-run", action="store_true",
                   help="Force mock backend + results/dry_run output. No numbers.")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out-dir", default=None,
                   help="Override output dir (default results/dry_run|final).")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap items per cell (smoke tests / pilot sampling fallback).")
    p.add_argument("--timeout-s", type=float, default=None,
                   help="Per-item soft timeout; records timeout=True if exceeded.")
    p.add_argument("--task", action="append", default=None,
                   help="Run only this task id. Repeatable.")
    p.add_argument("--model", action="append", default=None,
                   help="Run only this model id. Repeatable.")
    p.add_argument("--precision", action="append", default=None,
                   help="Run only this precision. Repeatable.")
    p.add_argument("--resume", action="store_true",
                   help="Append to an existing cell if config hash and git commit match.")
    p.add_argument("--allow-dirty", action="store_true",
                   help="Run even if the git working tree is dirty.")
    return p.parse_args(argv)


def out_dir(dry_run: bool, override: str | None) -> str:
    if override:
        d = override
    else:
        d = os.path.join("results", "dry_run" if dry_run else "final")
    os.makedirs(d, exist_ok=True)
    return d


def _task_name(task_cfg: dict) -> str:
    return task_cfg.get("name") or task_cfg.get("kind") or task_cfg["id"]


def _precisions(cfg: dict) -> list[str]:
    quants = list(cfg.get("quantization", []) or [])
    ref = cfg.get("reference_precision")
    # highest-precision local reference first, then the quant ladder
    if ref:
        return [ref] + [q for q in quants if q != ref]
    return quants


def _is_oom(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return isinstance(exc, MemoryError) or "out of memory" in msg or "oom" in msg


def main(argv=None) -> int:
    args = parse_args(argv)
    with open(args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    dry_run = args.dry_run or args.backend == "mock"
    backend_kind = "mock" if dry_run else args.backend
    basis = cfg["measurement_basis"]
    assert_single_basis(basis, cfg["tasks"])

    experiment_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]
    tasks_by_id = {t["id"]: t for t in cfg["tasks"]}
    run_order = cfg.get("run_order", list(tasks_by_id))
    if args.task:
        unknown = [t for t in args.task if t not in tasks_by_id]
        if unknown:
            print(f"[abort] unknown task(s): {unknown}", file=sys.stderr)
            return 2
        run_order = [t for t in run_order if t in set(args.task)]
    decoding = dict(cfg.get("decoding", {}))
    decoding.setdefault("seed", cfg.get("seed", 0))
    precisions = _precisions(cfg)
    if args.precision:
        precisions = [p for p in precisions if p in set(args.precision)]
        if not precisions:
            print(f"[abort] none of {args.precision} are in this config", file=sys.stderr)
            return 2
    models = list(cfg["models"])
    if args.model:
        wanted = set(args.model)
        models = [m for m in models if m["id"] in wanted]
        if len(models) != len(wanted):
            have = {m["id"] for m in models}
            print(f"[abort] unknown model(s): {sorted(wanted - have)}", file=sys.stderr)
            return 2
    reps = int(cfg.get("reproducibility", {}).get("repeats_per_cell", 1))
    odir = out_dir(dry_run, args.out_dir)
    host = host_provenance()
    if host["git_dirty"] and not dry_run and not args.allow_dirty:
        print("[abort] git working tree is dirty. Commit the run code first, "
              "or pass --allow-dirty.", file=sys.stderr)
        return 2

    print(f"[run] experiment_id={experiment_id} backend={backend_kind} "
          f"dry_run={dry_run} basis={basis}")
    print(f"[run] order={run_order} precisions={precisions} reps={reps} "
          f"models={[m['id'] for m in models]} commit={host['git_commit']} "
          f"llama_cpp={host['llama_cpp_python']} cuda={host['cuda_version']} "
          f"driver={host['driver_version']}")
    if dry_run:
        print("[run] DRY-RUN: MockBackend only; mock=True; results/final untouched.")

    total_written = 0
    # Keep at most ONE loaded backend. Caching all models would OOM a 24GB card
    # (and can stress the shared host — prior lab OOM killed the node).
    active_key = None
    active_backend = None
    for task_id in run_order:
        task_cfg = tasks_by_id[task_id]
        task_name = _task_name(task_cfg)
        try:
            items = tasksmod.load_task(task_cfg, data_root=args.data_root)
        except tasksmod.DatasetMissing as e:
            if dry_run:
                print(f"[dry-run] {task_id}: dataset absent ({e}). "
                      "Using 2 synthetic PROMPT-ONLY probes (no gold, no scoring).")
                items = _dry_probes(task_cfg)
            else:
                print(f"[abort] {e}", file=sys.stderr)
                return 2
        items = _apply_sampling(items, cfg)
        if args.limit:
            items = items[: args.limit]

        for model in models:
            model_precisions = _model_precisions(model, precisions)
            for precision in model_precisions:
                path = os.path.join(
                    odir, f"{task_id}__{model['id']}__{precision}.jsonl")
                gguf_sha = _gguf_sha256(model, precision, dry_run)
                ident = _cell_parts(
                    cfg, model, precision, task_id, decoding, host, dry_run,
                    gguf_sha256=gguf_sha)
                digest = config_hash(ident)
                done_ids = _resume_ids(path, digest, host["git_commit"], args.resume)
                if done_ids == "infeasible":
                    print(f"[skip] infeasible {path}")
                    continue
                if done_ids is None:
                    print(f"[skip] {path} exists with a different config hash or commit; "
                          "not appending", file=sys.stderr)
                    continue
                pending = [
                    it for it in items
                    if it.item_id not in done_ids
                ]
                if not pending:
                    print(f"[skip] complete {path} n={len(done_ids)}")
                    continue
                cache_key = (backend_kind, model["id"], precision)
                if active_key != cache_key:
                    if active_backend is not None:
                        print(f"[unload] {active_key[1]}@{active_key[2]}")
                        try:
                            active_backend.close()
                        except Exception as e:
                            print(f"[warn] unload failed: {e}", file=sys.stderr)
                        active_backend = None
                        active_key = None
                    print(f"[load] backend={backend_kind} model={model['id']} "
                          f"precision={precision}")
                    try:
                        active_backend = _make_backend(
                            backend_kind, model, precision, cfg)
                    except BaseException as e:
                        if _is_oom(e):
                            _write_infeasible(
                                path, experiment_id, task_cfg, task_name, model,
                                precision, basis, decoding, host, digest, ident,
                                f"{type(e).__name__}: {e}", oom=True)
                            print(f"[infeasible] OOM loading {model['id']}@{precision}: {e}")
                            continue
                        raise
                    active_key = cache_key
                    if gguf_sha and getattr(active_backend, "gguf_path", None):
                        active_backend.gguf_sha256 = gguf_sha
                backend = active_backend
                mode = "a" if done_ids else "w"
                with open(path, mode, encoding="utf-8") as out:
                    for item in pending:
                        for rep in range(reps):
                            rec = _run_one(
                                backend, backend_kind, experiment_id, task_cfg,
                                task_name, model, precision, basis, decoding,
                                item, rep, args.timeout_s, host, digest, ident, cfg)
                            out.write(rec.to_json() + "\n")
                            out.flush()
                            total_written += 1
                            if rec.oom:
                                print(f"[oom-item] {task_id} {model['id']} {precision} "
                                      f"{item.item_id}")
                print(f"[ok] wrote {path}")
    if active_backend is not None:
        try:
            active_backend.close()
        except Exception:
            pass
    print(f"[done] experiment_id={experiment_id} records={total_written}")
    return 0


def _model_precisions(model: dict, precisions: list[str]) -> list[str]:
    """Respect a per-model highest-feasible precision when the ladder starts with
    a local reference slot (FP16/BF16). Pure quant ladders are left unchanged."""
    if not precisions:
        return precisions
    hf = model.get("highest_feasible_precision")
    if not hf or precisions[0] not in ("FP16", "BF16", "F16", "CLOUD"):
        return list(dict.fromkeys(precisions))
    out = list(precisions)
    out[0] = hf
    return list(dict.fromkeys(out))


def _make_backend(kind: str, model: dict, precision: str, cfg: dict):
    kw = {}
    if kind == "transformers":
        kw["hf_name"] = model.get("hf")
        kw["revision"] = model.get("revision")
    elif kind == "llama_cpp":
        by = model.get("gguf_by_precision") or {}
        path = by.get(precision) or model.get("gguf_path")
        if not path:
            raise ValueError(
                f"No GGUF path for model={model.get('id')} precision={precision}. "
                "Set models[].gguf_by_precision[precision] or gguf_path.")
        kw["gguf_path"] = path
        # Same HF tokenizer as transformers FP16 — required for chat-template parity.
        kw["hf_name"] = model.get("hf") or model.get("hf_id")
        kw["revision"] = model.get("revision")
        serving = cfg.get("serving", {})
        kw["n_gpu_layers"] = serving.get("n_gpu_layers", -1)
        kw["n_ctx"] = serving.get("n_ctx", 8192)
        kw["n_batch"] = serving.get("n_batch", 512)
    elif kind == "cloud":
        cref = cfg.get("cloud_reference") or {}
        provider = (
            model.get("provider")
            or cref.get("provider")
            or "anthropic"
        )
        api_model = (
            model.get("api_model")
            or cref.get("api_model")
            or model.get("name")
        )
        if not api_model:
            raise ValueError(
                "Cloud backend needs models[].api_model or cloud_reference.api_model")
        default_key = (
            "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY")
        kw["provider"] = provider
        kw["api_model"] = api_model
        kw["key_env"] = (
            model.get("key_env") or cref.get("key_env") or default_key)
        kw["base_url"] = model.get("base_url") or cref.get("base_url")
    return build_backend(kind, model["id"], precision, **kw)


def _run_one(backend, backend_kind, experiment_id, task_cfg, task_name, model,
             precision, basis, decoding, item, rep, timeout_s,
             host=None, digest=None, ident=None, cfg=None) -> RunRecord:
    prompt = tasksmod.build_prompt(task_cfg, item)
    # Per-call copy so task_kind / grammar does not leak across families.
    decoding = dict(decoding)
    decoding.setdefault("task_kind", task_cfg.get("kind"))
    rec = RunRecord(
        experiment_id=experiment_id, task_family=task_cfg["id"],
        task_name=task_name, dataset=task_cfg.get("dataset", ""),
        item_id=item.item_id, model_id=model["id"], precision=precision,
        measurement_basis=basis, seed=int(decoding.get("seed", 0)), rep=rep,
        model_revision=model.get("revision"),
        prompt_version=tasksmod.PROMPT_VERSION, prompt_sha256=sha256(prompt),
        backend=backend_kind, input_length_chars=len(prompt),
        gold_label=item.reference.get("gold_label")
        if isinstance(item.reference, dict) else None,
    )
    t0 = time.time()
    try:
        gen = backend.generate(prompt, decoding)
    except BaseException as e:  # never drop: record and continue
        rec.exception = f"{type(e).__name__}: {e}"
        rec.oom = _is_oom(e)
        rec.total_latency_s = time.time() - t0
        _attach_provenance(rec, backend, decoding, host, digest, ident, cfg)
        return rec

    rec.mock = gen.mock
    rec.finish_reason = gen.finish_reason
    rec.raw_model_output = gen.text
    rec.prompt_tokens = gen.prompt_tokens
    rec.output_tokens = gen.output_tokens
    rec.total_latency_s = gen.total_latency_s
    rec.ttft_s = gen.ttft_s
    rec.tokens_per_s = gen.tokens_per_s
    rec.peak_memory_mb = gen.peak_memory_mb
    rec.peak_memory_method = gen.peak_memory_method

    if timeout_s is not None and (gen.total_latency_s or 0) > timeout_s:
        rec.timeout = True

    # Quality scoring only for real runs with a usable reference.
    if not gen.mock:
        try:
            q, detail = score(basis, task_cfg["kind"], gen.text, item.reference)
            rec.quality, rec.quality_detail = q, detail
            rec.parsed_output = detail.get("pred")
            if detail.get("malformed"):
                rec.malformed_output = True
        except NotImplementedError:
            pass  # scorer not real yet (e.g. basis B); leave quality=None
        except Exception as e:
            rec.exception = f"score:{type(e).__name__}: {e}"
            rec.malformed_output = True
    _attach_provenance(rec, backend, decoding, host, digest, ident, cfg)
    return rec


def _attach_provenance(rec, backend, decoding, host, digest, ident, cfg) -> None:
    if host is None:
        return
    serving = (cfg or {}).get("serving", {})
    wrapper = (
        "hf_chat_template" if decoding.get("use_chat_template", True) else "raw"
    )
    rec.provenance = {
        **host,
        "temperature": decoding.get("temperature"),
        "top_p": decoding.get("top_p"),
        "seed": decoding.get("seed"),
        "max_new_tokens": decoding.get("max_new_tokens"),
        "n_ctx": serving.get("n_ctx"),
        "n_batch": serving.get("n_batch"),
        "n_gpu_layers": serving.get("n_gpu_layers"),
        "wrapper": wrapper,
        "input_mode": getattr(backend, "input_mode", None) or (ident or {}).get("input_mode"),
        "gguf_filename": getattr(backend, "gguf_filename", None),
        "gguf_sha256": getattr(backend, "gguf_sha256", None) or (ident or {}).get("gguf_sha256"),
        "config_hash": digest,
        "config": ident,
    }


_SHA_CACHE: dict[str, str] = {}


def _gguf_sha256(model: dict, precision: str, dry_run: bool):
    if dry_run:
        return None
    by = model.get("gguf_by_precision") or {}
    path = by.get(precision) or model.get("gguf_path")
    if not path:
        return None
    cached = _SHA_CACHE.get(path)
    if cached:
        return cached
    print(f"[sha256] {path}", flush=True)
    digest = file_sha256(path)
    _SHA_CACHE[path] = digest
    return digest


def _cell_parts(cfg, model, precision, task_id, decoding, host, dry_run,
                gguf_sha256=None) -> dict:
    serving = cfg.get("serving", {})
    samp = cfg.get("sampling") or {}
    wrapper = (
        "hf_chat_template" if decoding.get("use_chat_template", True) else "raw"
    )
    input_mode = "token_ids" if cfg.get("decoding", {}).get("input_mode", "token_ids") == "token_ids" else "string"
    if dry_run:
        input_mode = "mock"
    return cell_identity(
        model_id=model["id"],
        precision=precision,
        task_id=task_id,
        prompt_version=tasksmod.PROMPT_VERSION,
        wrapper=wrapper,
        input_mode=input_mode,
        decoding=decoding,
        serving=serving,
        sampling_seed=int(samp.get("seed", cfg.get("seed", 0))),
        sampling_n=int(samp.get("size_per_task") or 0),
        gguf_sha256=gguf_sha256,
    )


def _resume_ids(path: str, digest: str, commit: str, resume: bool):
    """Item ids already stored for this hash and commit.

    Returns None when the file exists but does not match, so the caller must
    not append.
    """
    marker = path + ".infeasible.json"
    if os.path.exists(marker) and not os.path.exists(path):
        try:
            meta = json.loads(open(marker, encoding="utf-8").read())
        except Exception:
            return None
        if meta.get("config_hash") == digest and meta.get("git_commit") == commit:
            return "infeasible"
        return None
    if not os.path.exists(path):
        return set()
    records, torn = load_jsonl(path)
    if torn:
        print(f"[resume] dropped a torn final line in {path}")
    if not records:
        return set()
    if not resume:
        print(f"[abort] {path} exists; pass --resume to continue it", file=sys.stderr)
        return None
    hashes = {((r.get("provenance") or {}).get("config_hash")) for r in records}
    commits = {((r.get("provenance") or {}).get("git_commit")) for r in records}
    if hashes != {digest} or commits != {commit}:
        return None
    return {r.get("item_id") for r in records}


def _write_infeasible(path, experiment_id, task_cfg, task_name, model, precision,
                      basis, decoding, host, digest, ident, message, oom=False):
    marker = path + ".infeasible.json"
    payload = {
        "experiment_id": experiment_id,
        "task_family": task_cfg["id"],
        "task_name": task_name,
        "model_id": model["id"],
        "precision": precision,
        "measurement_basis": basis,
        "oom": oom,
        "infeasible": True,
        "exception": message,
        "config_hash": digest,
        "git_commit": host.get("git_commit"),
        "provenance": {**host, "config_hash": digest, "config": ident},
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(marker, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _apply_sampling(items, cfg):
    samp = cfg.get("sampling")
    if not samp:
        return items
    n = samp.get("size_per_task")
    seed = int(samp.get("seed", cfg.get("seed", 0)))
    stratum = None
    if samp.get("stratified"):
        def stratum(it):
            ref = it.reference if isinstance(it.reference, dict) else {}
            g = ref.get("gold_label")
            return str(g) if g is not None else None
    return stratified_sample(items, n, seed, id_of=lambda it: it.item_id,
                             stratum_of=stratum)


def _dry_probes(task_cfg: dict) -> list["tasksmod.Item"]:
    ref = {"source": "S-DRY"} if task_cfg["kind"] == "trace_link" else {}
    return [
        tasksmod.Item(item_id=f"dry-{i}", prompt=f"[dry probe {i}]",
                      reference=ref, long_context=task_cfg.get("long_context", False))
        for i in range(2)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
