#!/usr/bin/env python3
"""Aggregate the aligned-v2 ladder against FP16, legacy raw-prompt, and superseded cells.

Item-level percentile bootstrap, 10_000 resamples, numpy Generator seed 20260905.
The script that wrote analysis/out/pillar1_bootstrap_ci.csv is not in the repo;
this is the same estimator family (resample items, 2.5/97.5 percentiles) and is
not claimed to reproduce that file to the last digit.

q_ok is the mean of quality on rows with malformed_output false.
A family macro is the unweighted mean of the per-model values.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

SEED = 20260905
N_BOOT = 10_000


def load_mid_refs(data_root: str = "data") -> dict:
    from src import tasks as T
    items = T.load_task(
        {"kind": "span_detection", "dataset": "damir_pure", "long_context": False},
        data_root=data_root,
    )
    return {it.item_id: it.reference for it in items}


def mid_p3a(rows: list[dict], refs: dict) -> dict:
    """P3-A cell scores. See MID_METRIC.md. Missing gold is skipped."""
    from src.metrics import score
    combined, span_amb, decision, qok = [], [], [], []
    by_id = {}
    n_skip = 0
    for r in rows:
        ref = refs.get(r.get("item_id"))
        if ref is None:
            n_skip += 1
            continue
        _q, detail = score(
            "A", "span_detection", r.get("raw_model_output") or "", ref, mid_metric="p3a")
        combined.append(float(detail["combined"]))
        by_id[r.get("item_id")] = float(detail["combined"])
        decision.append(float(detail["decision"]))
        if detail["gold_ambiguous"]:
            span_amb.append(float(detail["span_f1"]))
        if not detail["malformed"]:
            qok.append(float(detail["combined"]))
    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")
    return {
        "combined": mean(combined),
        "span_f1_ambiguous": mean(span_amb),
        "decision_accuracy": mean(decision),
        "q_ok_p3a": mean(qok),
        "n_wellformed": len(qok),
        "n_scored": len(combined),
        "n_missing_gold": n_skip,
        "by_id": by_id,
    }

SOURCES = [
    ("results/pillar1_fp16", "transformers_fp16"),
    ("results/pillar1_fp16_v2", "transformers_fp16_v2"),
    ("results/pillar1_aligned_v2", "aligned_v2"),
    ("results/pillar1_grammar_v2", "grammar_v2"),
    ("results/pillar1_f16_gguf_aligned", "superseded_string_256"),
    ("results/pillar1_quant_aligned", "superseded_string_256"),
    ("results/pillar1_quant", "legacy_raw"),
    ("results/pillar1_f16_gguf", "legacy_raw"),
]


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def cap_of(row: dict, source: str) -> int | None:
    prov = row.get("provenance") or {}
    if prov.get("max_new_tokens"):
        return int(prov["max_new_tokens"])
    task = row.get("task_family") or ""
    if source == "transformers_fp16":
        return 128 if task in ("TF_Surface", "TF_Mid") else 256
    if source == "superseded_string_256":
        return 256
    return None


def hit_cap(row: dict, source: str) -> bool:
    cap = cap_of(row, source)
    finish = row.get("finish_reason") or (row.get("provenance") or {}).get("finish_reason")
    if finish == "length":
        return True
    if cap is None or row.get("output_tokens") is None:
        return False
    return int(row["output_tokens"]) >= cap


def qok(qualities: np.ndarray, malformed: np.ndarray) -> float:
    keep = qualities[~malformed]
    if len(keep) == 0:
        return float("nan")
    return float(keep.mean())


def bootstrap_cell(qualities: np.ndarray, malformed: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    n = len(qualities)
    q_means = np.empty(N_BOOT)
    m_means = np.empty(N_BOOT)
    ok_means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        qq = qualities[idx]
        mm = malformed[idx]
        q_means[i] = qq.mean()
        m_means[i] = mm.mean()
        keep = qq[~mm]
        ok_means[i] = keep.mean() if len(keep) else np.nan
    def pct(a):
        a = a[~np.isnan(a)]
        lo, hi = np.quantile(a, [0.025, 0.975])
        return float(lo), float(hi)
    q_lo, q_hi = pct(q_means)
    m_lo, m_hi = pct(m_means)
    o_lo, o_hi = pct(ok_means)
    return {
        "quality_ci_lo": q_lo, "quality_ci_hi": q_hi,
        "mal_ci_lo": m_lo, "mal_ci_hi": m_hi,
        "q_ok_ci_lo": o_lo, "q_ok_ci_hi": o_hi,
    }


def bootstrap_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Mean of a-b, with a percentile CI. a and b are aligned item qualities."""
    diff = a - b
    rng = np.random.default_rng(SEED)
    n = len(diff)
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        means[i] = diff[rng.integers(0, n, n)].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        open(path, "w").close()
        return
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args()
    os.chdir(args.root)

    cells: dict[tuple, list[dict]] = defaultdict(list)
    manifest = {}
    mid_refs = load_mid_refs("data")
    mid_vectors = {}
    for rel, source in SOURCES:
        if not os.path.isdir(rel):
            print(f"[skip] missing {rel}")
            continue
        for fp in sorted(glob.glob(os.path.join(rel, "*.jsonl"))):
            rows = load_jsonl(fp)
            if not rows or any(r.get("mock") for r in rows):
                continue
            for r in rows:
                r["_source"] = source
                r["_file"] = os.path.basename(fp)
                key = (source, r.get("task_family"), r.get("model_id"), r.get("precision"))
                cells[key].append(r)
                prov = r.get("provenance") or {}
                name = prov.get("gguf_filename")
                digest = prov.get("gguf_sha256")
                if name and digest:
                    manifest[(name, digest)] = {
                        "gguf_filename": name,
                        "sha256": digest,
                        "model_id": r.get("model_id"),
                        "precision": r.get("precision"),
                    }

    aggregate = []
    boot_rows = []
    for key, rows in sorted(cells.items()):
        source, task, model, precision = key
        q = np.array([float(r.get("quality") or 0) for r in rows])
        mal = np.array([bool(r.get("malformed_output")) for r in rows])
        caps = sum(1 for r in rows if hit_cap(r, source))
        stats = bootstrap_cell(q, mal)
        row = {
            "source": source,
            "task": task,
            "model": model,
            "precision": precision,
            "n": len(rows),
            "quality_mean": float(q.mean()),
            "q_ok_mean": qok(q, mal),
            "malformed_rate": float(mal.mean()),
            "n_ok": int((~mal).sum()),
            "cap_hits": caps,
            "wrapper": (rows[0].get("provenance") or {}).get("wrapper")
                       or ("hf_chat_template" if source != "legacy_raw" else "raw"),
            "input_mode": (rows[0].get("provenance") or {}).get("input_mode") or "",
            "span_f1_ambiguous": "",
            "decision_accuracy": "",
            "combined": "",
            "n_wellformed": "",
            "combined_ci_lo": "",
            "combined_ci_hi": "",
        }
        if task == "TF_Mid":
            mid = mid_p3a(rows, mid_refs)
            row["span_f1_ambiguous"] = mid["span_f1_ambiguous"]
            row["decision_accuracy"] = mid["decision_accuracy"]
            row["combined"] = mid["combined"]
            row["n_wellformed"] = mid["n_wellformed"]
            row["q_ok_mean"] = mid["q_ok_p3a"]
            if mid["by_id"]:
                vals = np.array(list(mid["by_id"].values()), dtype=float)
                ci = bootstrap_cell(vals, np.zeros(len(vals), dtype=bool))
                row["combined_ci_lo"] = ci["quality_ci_lo"]
                row["combined_ci_hi"] = ci["quality_ci_hi"]
            mid_vectors[key] = mid["by_id"]
        aggregate.append(row)
        boot_rows.append({**row, **stats})

    write_csv(os.path.join(args.out, "pillar1_aligned_v2_aggregate.csv"), aggregate)
    write_csv(os.path.join(args.out, "pillar1_aligned_v2_bootstrap.csv"), boot_rows)
    man_rows = sorted(manifest.values(), key=lambda r: (r["model_id"], r["precision"]))
    write_csv("gguf_manifest.csv", man_rows)

    # Paired contrasts: aligned_v2 F16 vs each quant, within model and task.
    by_item = defaultdict(dict)
    for (source, task, model, precision), rows in cells.items():
        if source != "aligned_v2":
            continue
        for r in rows:
            by_item[(task, model, r.get("item_id"))][precision] = float(r.get("quality") or 0)
    paired = []
    for task in ("TF_Surface", "TF_Mid", "TF_Reason"):
        for quant in ("Q8_0", "Q6_K", "Q4_K_M"):
            model_diffs = []
            models = sorted({k[2] for k in cells if k[0] == "aligned_v2" and k[1] == task})
            for model in models:
                a, b = [], []
                ids = [i for (t, m, i) in by_item if t == task and m == model]
                for iid in sorted(ids):
                    slot = by_item[(task, model, iid)]
                    if "F16" in slot and quant in slot:
                        a.append(slot["F16"])
                        b.append(slot[quant])
                if len(a) < 2:
                    continue
                mean, lo, hi = bootstrap_paired(np.array(a), np.array(b))
                paired.append({
                    "task": task, "model": model, "contrast": f"F16-{quant}",
                    "metric": "stored_quality",
                    "n": len(a), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
                    "ci_includes_0": lo <= 0 <= hi,
                })
                model_diffs.append((np.array(a), np.array(b)))
            if len(model_diffs) >= 2:
                # Family macro: resample each model independently, then average means.
                rng = np.random.default_rng(SEED)
                macros = np.empty(N_BOOT)
                point = np.mean([ (a - b).mean() for a, b in model_diffs ])
                for i in range(N_BOOT):
                    macros[i] = np.mean([
                        (a - b)[rng.integers(0, len(a), len(a))].mean()
                        for a, b in model_diffs
                    ])
                lo, hi = np.quantile(macros, [0.025, 0.975])
                paired.append({
                    "task": task, "model": "MACRO", "contrast": f"F16-{quant}",
                    "metric": "stored_quality",
                    "n": sum(len(a) for a, _b in model_diffs),
                    "mean_diff": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                    "ci_includes_0": float(lo) <= 0 <= float(hi),
                })
    def _paired_from_vectors(task, model, contrast, a, b, metric):
        ids = sorted(set(a) & set(b))
        if len(ids) < 2:
            return
        aa = np.array([a[i] for i in ids], dtype=float)
        bb = np.array([b[i] for i in ids], dtype=float)
        mean, lo, hi = bootstrap_paired(aa, bb)
        paired.append({
            "task": task, "model": model, "contrast": contrast, "metric": metric,
            "n": len(a), "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "ci_includes_0": lo <= 0 <= hi,
        })

    # Corrected Mid score: prefer the 1024-cap FP16 rerun when that cell is complete.
    fp16_source = {}
    for model in ("qwen25_7b_instruct", "mistral_7b_instruct", "llama31_8b_instruct", "gemma2_9b_it"):
        v2 = mid_vectors.get(("transformers_fp16_v2", "TF_Mid", model, "FP16"))
        old = mid_vectors.get(("transformers_fp16", "TF_Mid", model, "FP16"))
        chosen = v2 if v2 is not None and len(v2) == 40 else old
        if chosen is not None:
            fp16_source[model] = chosen
        f16 = mid_vectors.get(("aligned_v2", "TF_Mid", model, "F16"))
        if chosen is not None and f16 is not None:
            _paired_from_vectors("TF_Mid", model, "FP16-alignedF16", chosen, f16, "mid_combined")
    if len(fp16_source) >= 2:
        models = sorted(fp16_source)
        f16s = []
        for model in models:
            f16 = mid_vectors.get(("aligned_v2", "TF_Mid", model, "F16"))
            ids = sorted(set(fp16_source[model]) & set(f16 or {}))
            if f16 is None or len(ids) < 2:
                f16s = []
                break
            f16s.append((
                np.array([fp16_source[model][i] for i in ids]),
                np.array([f16[i] for i in ids]),
            ))
        if f16s:
            rng = np.random.default_rng(SEED)
            point = float(np.mean([(a - b).mean() for a, b in f16s]))
            macros = np.empty(N_BOOT)
            for i in range(N_BOOT):
                macros[i] = np.mean([
                    (a - b)[rng.integers(0, len(a), len(a))].mean()
                    for a, b in f16s
                ])
            lo, hi = np.quantile(macros, [0.025, 0.975])
            paired.append({
                "task": "TF_Mid", "model": "MACRO", "contrast": "FP16-alignedF16",
                "metric": "mid_combined",
                "n": sum(len(v) for v in fp16_source.values()),
                "mean_diff": point, "ci_lo": float(lo), "ci_hi": float(hi),
                "ci_includes_0": float(lo) <= 0 <= float(hi),
            })
    for quant in ("Q8_0", "Q6_K", "Q4_K_M"):
        vecs = []
        for model in ("qwen25_7b_instruct", "mistral_7b_instruct", "llama31_8b_instruct", "gemma2_9b_it"):
            a = mid_vectors.get(("aligned_v2", "TF_Mid", model, "F16"))
            b = mid_vectors.get(("aligned_v2", "TF_Mid", model, quant))
            if a is None or b is None:
                continue
            ids = sorted(set(a) & set(b))
            if len(ids) < 2:
                continue
            _paired_from_vectors("TF_Mid", model, f"F16-{quant}", a, b, "mid_combined")
            vecs.append((
                np.array([a[i] for i in ids]),
                np.array([b[i] for i in ids]),
            ))
        if len(vecs) >= 2:
            rng = np.random.default_rng(SEED)
            point = float(np.mean([(a - b).mean() for a, b in vecs]))
            macros = np.empty(N_BOOT)
            for i in range(N_BOOT):
                macros[i] = np.mean([
                    (a - b)[rng.integers(0, len(a), len(a))].mean() for a, b in vecs
                ])
            lo, hi = np.quantile(macros, [0.025, 0.975])
            paired.append({
                "task": "TF_Mid", "model": "MACRO", "contrast": f"F16-{quant}",
                "metric": "mid_combined",
                "n": sum(len(a) for a, _b in vecs),
                "mean_diff": point, "ci_lo": float(lo), "ci_hi": float(hi),
                "ci_includes_0": float(lo) <= 0 <= float(hi),
            })

    write_csv(os.path.join(args.out, "pillar1_aligned_v2_paired.csv"), paired)

    def lookup(source, task, model, precision):
        for row in aggregate:
            if (row["source"], row["task"], row["model"], row["precision"]) == (source, task, model, precision):
                return row
        return None

    print("\n== family macros (aligned_v2 and transformers FP16) ==")
    for source, prec in (
        ("transformers_fp16", "FP16"),
        ("aligned_v2", "F16"),
        ("aligned_v2", "Q8_0"),
        ("aligned_v2", "Q6_K"),
        ("aligned_v2", "Q4_K_M"),
        ("superseded_string_256", "F16"),
    ):
        for task in ("TF_Surface", "TF_Mid", "TF_Reason"):
            rows = [r for r in aggregate if r["source"] == source and r["task"] == task and r["precision"] == prec]
            if not rows:
                continue
            print(
                f"{source} {task} {prec} models={len(rows)} "
                f"quality={np.mean([r['quality_mean'] for r in rows]):.4f} "
                f"q_ok={np.nanmean([r['q_ok_mean'] for r in rows]):.4f} "
                f"mal={np.mean([r['malformed_rate'] for r in rows]):.4f} "
                f"cap_hits={sum(r['cap_hits'] for r in rows)}"
            )

    print("\n== per model aligned_v2 vs superseded F16 ==")
    for task in ("TF_Mid", "TF_Reason", "TF_Surface"):
        for model in ("qwen25_7b_instruct", "mistral_7b_instruct", "llama31_8b_instruct", "gemma2_9b_it"):
            new = lookup("aligned_v2", task, model, "F16")
            old = lookup("superseded_string_256", task, model, "F16")
            if not new:
                continue
            old_q = f"{old['q_ok_mean']:.4f}" if old else "NA"
            old_cap = old["cap_hits"] if old else "NA"
            print(
                f"{task} {model} v2_q_ok={new['q_ok_mean']:.4f} v2_mal={new['malformed_rate']:.4f} "
                f"v2_cap={new['cap_hits']} old_q_ok={old_q} old_cap={old_cap}"
            )
    print(f"[ok] cells={len(aggregate)} manifest={len(man_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
