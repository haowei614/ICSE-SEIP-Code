#!/usr/bin/env python3
"""Compute numbers for the paper fix pass from logged runs.

Frozen scoring: parser=lenient_unique, Mid mid_metric=p3a.
Writes CSVs under analysis/out/ and prints a human summary.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np

SEED = 20260905
N_BOOT = 10_000
PARSER = "lenient_unique"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "analysis", "out")
DATA = os.path.join(ROOT, "data")


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    return float(np.mean(xs)) if xs else float("nan")


def score_row(task: str, text: str, ref: dict) -> tuple[float, bool, float]:
    from src.metrics import score

    kind = {
        "TF_Surface": "classification",
        "TF_Mid": "span_detection",
        "TF_Reason": "trace_link",
    }[task]
    q, detail = score(
        "A",
        kind,
        text or "",
        ref,
        mid_metric="p3a" if task == "TF_Mid" else "legacy",
        parser=PARSER,
    )
    mal = bool(detail.get("malformed"))
    # q_ok: mean quality on well-formed; here return item quality if ok else nan
    qok = float(q) if not mal else float("nan")
    return float(q), mal, qok


def load_refs() -> dict[str, dict[str, dict]]:
    from src import tasks as T

    out: dict[str, dict[str, dict]] = {}
    cfgs = [
        ("TF_Surface", {"kind": "classification", "dataset": "promise_nfr", "long_context": False}),
        ("TF_Mid", {"kind": "span_detection", "dataset": "damir_pure", "long_context": False}),
        ("TF_Reason", {"kind": "trace_link", "dataset": "trace_etour", "long_context": False}),
    ]
    for task, cfg in cfgs:
        items = T.load_task(cfg, data_root=DATA)
        out[task] = {it.item_id: it.reference for it in items}
    return out


def cell_scores(rows: list[dict], task: str, refs: dict) -> dict:
    qs, mals, qoks, by_id = [], [], [], {}
    for r in rows:
        iid = r.get("item_id")
        ref = refs.get(iid)
        if ref is None:
            continue
        q, mal, qok = score_row(task, r.get("raw_model_output") or "", ref)
        qs.append(q)
        mals.append(1.0 if mal else 0.0)
        if not mal:
            qoks.append(q)
        by_id[iid] = q
    return {
        "n": len(qs),
        "quality": mean(qs),
        "mal": mean(mals),
        "q_ok": mean(qoks),
        "n_ok": len(qoks),
        "by_id": by_id,
        "qualities": np.array(qs, dtype=float),
        "malformed": np.array(mals, dtype=bool),
    }


def boot_mean_diff(a: np.ndarray, b: np.ndarray, rng: np.random.Generator):
    n = len(a)
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, n)
        diffs[i] = float(np.mean(a[idx] - b[idx]))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(np.mean(a - b)), float(lo), float(hi)


def paired_macro(cells_a: list[np.ndarray], cells_b: list[np.ndarray], rng):
    macros = np.empty(N_BOOT)
    for i in range(N_BOOT):
        vals = []
        for a, b in zip(cells_a, cells_b):
            idx = rng.integers(0, len(a), len(a))
            vals.append(float(np.mean(a[idx] - b[idx])))
        macros[i] = float(np.mean(vals))
    lo, hi = np.quantile(macros, [0.025, 0.975])
    point = float(np.mean([float(np.mean(a - b)) for a, b in zip(cells_a, cells_b)]))
    return point, float(lo), float(hi)


def basename_model_prec(path: str) -> tuple[str, str, str]:
    # TF_X__model__PREC.jsonl
    name = os.path.basename(path).replace(".jsonl", "")
    parts = name.split("__")
    return parts[0], parts[1], parts[2]


def collect_dir(dirpath: str, refs: dict) -> dict:
    cells = {}
    for path in sorted(glob.glob(os.path.join(dirpath, "*.jsonl"))):
        task, model, prec = basename_model_prec(path)
        rows = load_jsonl(path)
        cells[(task, model, prec)] = cell_scores(rows, task, refs[task])
        cells[(task, model, prec)]["path"] = path
        cells[(task, model, prec)]["rows"] = rows
    return cells


def family_macro(cells: dict, task: str, prec: str, key: str) -> float:
    vals = [
        cells[k][key]
        for k in cells
        if k[0] == task and k[2] == prec and not np.isnan(cells[k][key])
    ]
    return mean(vals)


def main() -> None:
    os.chdir(ROOT)
    refs = load_refs()
    rng = np.random.default_rng(SEED)

    # ---- Cloud rescore ----
    cloud_old, cloud_new = [], []
    for path in sorted(glob.glob("results/pillar1_cloud/*.jsonl")):
        task, model, prec = basename_model_prec(path)
        rows = load_jsonl(path)
        old_q = mean([float(r.get("quality") or 0) for r in rows])
        old_mal = mean([1.0 if r.get("malformed_output") else 0.0 for r in rows])
        old_qok = mean(
            [
                float(r["quality"])
                for r in rows
                if not r.get("malformed_output") and r.get("quality") is not None
            ]
        )
        new = cell_scores(rows, task, refs[task])
        cloud_old.append(
            {
                "task": task,
                "quality": round(old_q, 4),
                "mal": round(old_mal, 4),
                "q_ok": round(old_qok, 4),
            }
        )
        cloud_new.append(
            {
                "task": task,
                "quality": round(new["quality"], 4),
                "mal": round(new["mal"], 4),
                "q_ok": round(new["q_ok"], 4),
                "n": new["n"],
            }
        )
    write_csv(os.path.join(OUT, "cloud_rescore_old.csv"), cloud_old)
    write_csv(os.path.join(OUT, "cloud_rescore_new.csv"), cloud_new)

    # ---- Aligned v2 + FP16 ----
    aligned = collect_dir("results/pillar1_aligned_v2", refs)
    fp16 = collect_dir("results/pillar1_fp16", refs)
    legacy = collect_dir("results/pillar1_quant", refs)
    # also legacy F16
    for path in sorted(glob.glob("results/pillar1_f16_gguf/*.jsonl")):
        task, model, prec = basename_model_prec(path)
        rows = load_jsonl(path)
        legacy[(task, model, prec)] = cell_scores(rows, task, refs[task])
        legacy[(task, model, prec)]["path"] = path
        legacy[(task, model, prec)]["rows"] = rows
    superseded = {}
    for path in sorted(glob.glob("results/pillar1_f16_gguf_aligned/*.jsonl")):
        task, model, prec = basename_model_prec(path)
        rows = load_jsonl(path)
        superseded[(task, model, prec)] = cell_scores(rows, task, refs[task])
        superseded[(task, model, prec)]["path"] = path
        superseded[(task, model, prec)]["rows"] = rows
    for path in sorted(glob.glob("results/pillar1_quant_aligned/*.jsonl")):
        task, model, prec = basename_model_prec(path)
        rows = load_jsonl(path)
        superseded[(task, model, prec)] = cell_scores(rows, task, refs[task])
        superseded[(task, model, prec)]["path"] = path
        superseded[(task, model, prec)]["rows"] = rows

    ladder_rows = []
    for task in ("TF_Surface", "TF_Mid", "TF_Reason"):
        for prec in ("FP16", "F16", "Q8_0", "Q6_K", "Q4_K_M"):
            src = fp16 if prec == "FP16" else aligned
            if not any(k[0] == task and k[2] == prec for k in src):
                continue
            ladder_rows.append(
                {
                    "task": task,
                    "precision": prec,
                    "quality": round(family_macro(src, task, prec, "quality"), 4),
                    "mal": round(family_macro(src, task, prec, "mal"), 4),
                    "q_ok": round(family_macro(src, task, prec, "q_ok"), 4),
                }
            )
    write_csv(os.path.join(OUT, "fix_pass_ladder.csv"), ladder_rows)

    # Retention vs aligned F16 (within-stack) and FP16 vs F16 stack parity
    ret_rows = []
    for task in ("TF_Surface", "TF_Mid", "TF_Reason"):
        f16_q = family_macro(aligned, task, "F16", "quality")
        f16_qok = family_macro(aligned, task, "F16", "q_ok")
        fp_q = family_macro(fp16, task, "FP16", "quality")
        fp_qok = family_macro(fp16, task, "FP16", "q_ok")
        ret_rows.append(
            {
                "task": task,
                "contrast": "FP16_vs_F16_quality",
                "ref": round(fp_q, 4),
                "cmp": round(f16_q, 4),
                "retention": round(f16_q / fp_q, 4) if fp_q else float("nan"),
            }
        )
        for prec in ("Q8_0", "Q6_K", "Q4_K_M"):
            q = family_macro(aligned, task, prec, "quality")
            qok = family_macro(aligned, task, prec, "q_ok")
            ret_rows.append(
                {
                    "task": task,
                    "contrast": f"F16_vs_{prec}_quality",
                    "ref": round(f16_q, 4),
                    "cmp": round(q, 4),
                    "retention": round(q / f16_q, 4) if f16_q else float("nan"),
                }
            )
            ret_rows.append(
                {
                    "task": task,
                    "contrast": f"F16_vs_{prec}_q_ok",
                    "ref": round(f16_qok, 4),
                    "cmp": round(qok, 4),
                    "retention": round(qok / f16_qok, 4) if f16_qok else float("nan"),
                }
            )
    write_csv(os.path.join(OUT, "fix_pass_retention.csv"), ret_rows)

    # Paired CIs F16 vs Q*
    paired_rows = []
    models = sorted({k[1] for k in aligned})
    for task in ("TF_Surface", "TF_Mid", "TF_Reason"):
        for prec in ("Q8_0", "Q6_K", "Q4_K_M"):
            a_list, b_list = [], []
            for model in models:
                ka = (task, model, "F16")
                kb = (task, model, prec)
                if ka not in aligned or kb not in aligned:
                    continue
                # align by item order from F16 by_id keys shared
                ids = sorted(set(aligned[ka]["by_id"]) & set(aligned[kb]["by_id"]))
                if not ids:
                    continue
                a = np.array([aligned[ka]["by_id"][i] for i in ids], dtype=float)
                b = np.array([aligned[kb]["by_id"][i] for i in ids], dtype=float)
                a_list.append(a)
                b_list.append(b)
                md, lo, hi = boot_mean_diff(a, b, rng)
                paired_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "contrast": f"F16-{prec}",
                        "mean_diff": round(md, 4),
                        "ci_lo": round(lo, 4),
                        "ci_hi": round(hi, 4),
                        "ci_includes_0": lo <= 0 <= hi,
                    }
                )
            if a_list:
                md, lo, hi = paired_macro(a_list, b_list, rng)
                paired_rows.append(
                    {
                        "task": task,
                        "model": "MACRO",
                        "contrast": f"F16-{prec}",
                        "mean_diff": round(md, 4),
                        "ci_lo": round(lo, 4),
                        "ci_hi": round(hi, 4),
                        "ci_includes_0": lo <= 0 <= hi,
                    }
                )
    write_csv(os.path.join(OUT, "fix_pass_paired.csv"), paired_rows)

    # ---- Baselines on the n=40 eval sample (from Surface F16 item ids or fp16) ----
    # Use item ids from aligned Surface F16 qwen as the locked eval sample proxy:
    # better: read config or use intersection of cloud item ids.
    sample_ids = {
        task: sorted(
            {
                r["item_id"]
                for path in glob.glob(f"results/pillar1_aligned_v2/{task}__*__F16.jsonl")
                for r in load_jsonl(path)
            }
        )
        for task in ("TF_Surface", "TF_Mid", "TF_Reason")
    }
    # majority class on Surface sample
    surf_labels = [
        str(refs["TF_Surface"][iid]["gold_label"]).strip().upper()
        for iid in sample_ids["TF_Surface"]
        if iid in refs["TF_Surface"]
    ]
    maj = Counter(surf_labels).most_common(1)[0][0]
    maj_acc = mean([1.0 if lab == maj else 0.0 for lab in surf_labels])

    # Mid always []
    mid_qs, mid_mals, mid_qoks = [], [], []
    for iid in sample_ids["TF_Mid"]:
        ref = refs["TF_Mid"][iid]
        q, mal, qok = score_row("TF_Mid", "[]", ref)
        mid_qs.append(q)
        mid_mals.append(1.0 if mal else 0.0)
        if not mal:
            mid_qoks.append(q)

    # Reason name-overlap: match UC source tokens against the corpus .java lexicon.
    all_java = set()
    for _iid, ref in refs["TF_Reason"].items():
        all_java.update(ref.get("gold_target_ids") or [])

    def name_overlap_pred(ref: dict) -> str:
        src = ref.get("source") or ""
        tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9]+", src))
        cands = []
        for f in sorted(all_java):
            stem = f.replace(".java", "")
            if stem in tokens or any(
                len(t) > 3 and (stem.lower() in t.lower() or t.lower() in stem.lower())
                for t in tokens
            ):
                cands.append(f)
        return json.dumps(cands[:5])

    rea_qs, rea_mals, rea_qoks = [], [], []
    for iid in sample_ids["TF_Reason"]:
        ref = refs["TF_Reason"][iid]
        text = name_overlap_pred(ref)
        q, mal, qok = score_row("TF_Reason", text, ref)
        rea_qs.append(q)
        rea_mals.append(1.0 if mal else 0.0)
        if not mal:
            rea_qoks.append(q)

    baselines = [
        {
            "task": "TF_Surface",
            "baseline": f"majority({maj})",
            "quality": round(maj_acc, 4),
            "mal": 0.0,
            "q_ok": round(maj_acc, 4),
            "n": len(surf_labels),
        },
        {
            "task": "TF_Mid",
            "baseline": "always_empty_list",
            "quality": round(mean(mid_qs), 4),
            "mal": round(mean(mid_mals), 4),
            "q_ok": round(mean(mid_qoks), 4),
            "n": len(mid_qs),
        },
        {
            "task": "TF_Reason",
            "baseline": "name_overlap_heuristic",
            "quality": round(mean(rea_qs), 4),
            "mal": round(mean(rea_mals), 4),
            "q_ok": round(mean(rea_qoks), 4),
            "n": len(rea_qs),
        },
    ]
    write_csv(os.path.join(OUT, "fix_pass_baselines.csv"), baselines)

    # ---- Misconfig table: raw / string-256 / aligned_v2 ; legacy_greedy vs lenient_unique ----
    mis_rows = []
    for task in ("TF_Mid", "TF_Reason"):
        for model in models:
            for label, store, prec in (
                ("raw_prompt", legacy, "F16"),
                ("string_aligned_256", superseded, "F16"),
                ("aligned_v2", aligned, "F16"),
            ):
                key = (task, model, prec)
                if key not in store:
                    continue
                cell = store[key]
                # also legacy_greedy rescore on same rows
                from src.metrics import score as score_fn

                kind = "span_detection" if task == "TF_Mid" else "trace_link"
                greedy_mal, greedy_q, greedy_qok = [], [], []
                for r in cell["rows"]:
                    ref = refs[task].get(r["item_id"])
                    if ref is None:
                        continue
                    q, d = score_fn(
                        "A",
                        kind,
                        r.get("raw_model_output") or "",
                        ref,
                        mid_metric="p3a" if task == "TF_Mid" else "legacy",
                        parser="legacy_greedy",
                    )
                    mal = bool(d.get("malformed"))
                    greedy_mal.append(1.0 if mal else 0.0)
                    greedy_q.append(float(q))
                    if not mal:
                        greedy_qok.append(float(q))
                mis_rows.append(
                    {
                        "task": task,
                        "model": model,
                        "stack": label,
                        "mal_lenient_unique": round(cell["mal"], 4),
                        "q_lenient_unique": round(cell["quality"], 4),
                        "qok_lenient_unique": round(cell["q_ok"], 4),
                        "mal_legacy_greedy": round(mean(greedy_mal), 4),
                        "q_legacy_greedy": round(mean(greedy_q), 4),
                        "qok_legacy_greedy": round(mean(greedy_qok), 4),
                    }
                )
    # macros per stack
    for task in ("TF_Mid", "TF_Reason"):
        for stack in ("raw_prompt", "string_aligned_256", "aligned_v2"):
            sub = [r for r in mis_rows if r["task"] == task and r["stack"] == stack]
            if not sub:
                continue
            mis_rows.append(
                {
                    "task": task,
                    "model": "MACRO",
                    "stack": stack,
                    "mal_lenient_unique": round(mean([r["mal_lenient_unique"] for r in sub]), 4),
                    "q_lenient_unique": round(mean([r["q_lenient_unique"] for r in sub]), 4),
                    "qok_lenient_unique": round(mean([r["qok_lenient_unique"] for r in sub]), 4),
                    "mal_legacy_greedy": round(mean([r["mal_legacy_greedy"] for r in sub]), 4),
                    "q_legacy_greedy": round(mean([r["q_legacy_greedy"] for r in sub]), 4),
                    "qok_legacy_greedy": round(mean([r["qok_legacy_greedy"] for r in sub]), 4),
                }
            )
    write_csv(os.path.join(OUT, "fix_pass_misconfig.csv"), mis_rows)

    # ---- FP16 caps ----
    cap_rows = []
    for path in sorted(glob.glob("results/pillar1_fp16/*.jsonl")):
        task, model, prec = basename_model_prec(path)
        rows = load_jsonl(path)
        cap = 128 if task in ("TF_Surface", "TF_Mid") else 256
        hits = 0
        for r in rows:
            if r.get("finish_reason") == "length":
                hits += 1
            elif r.get("output_tokens") is not None and int(r["output_tokens"]) >= cap:
                hits += 1
        # frozen mal
        cell = cell_scores(rows, task, refs[task])
        cap_rows.append(
            {
                "task": task,
                "model": model,
                "max_new_tokens": cap,
                "cap_hits": hits,
                "n": len(rows),
                "mal_frozen": round(cell["mal"], 4),
                "quality_frozen": round(cell["quality"], 4),
            }
        )
    write_csv(os.path.join(OUT, "fix_pass_fp16_caps.csv"), cap_rows)

    # llama-cpp version
    sample = load_jsonl(
        "results/pillar1_aligned_v2/TF_Reason__qwen25_7b_instruct__F16.jsonl"
    )[0]
    llama_ver = (sample.get("provenance") or {}).get("llama_cpp_python")

    print("== cloud old ==")
    for r in cloud_old:
        print(r)
    print("== cloud new ==")
    for r in cloud_new:
        print(r)
    print("== ladder ==")
    for r in ladder_rows:
        print(r)
    print("== retention (selected) ==")
    for r in ret_rows:
        if "MACRO" in r.get("model", "") or True:
            print(r)
    print("== paired MACRO ==")
    for r in paired_rows:
        if r["model"] == "MACRO":
            print(r)
    print("== baselines ==")
    for r in baselines:
        print(r)
    print("== misconfig MACRO ==")
    for r in mis_rows:
        if r["model"] == "MACRO":
            print(r)
    print("== fp16 caps (nonzero hits) ==")
    for r in cap_rows:
        if r["cap_hits"]:
            print(r)
    print("llama_cpp_python", llama_ver)


if __name__ == "__main__":
    main()
