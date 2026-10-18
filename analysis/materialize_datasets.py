#!/usr/bin/env python3
"""Materialize locked public RE datasets into harness-readable files.

Sources (downloaded under data/_downloads/):
  - PROMISE NFR: Zenodo 268542 (nfr.arff)
  - DAMIR-PURE: Figshare 29304650 (anaphoric ambiguity on PURE subset)
  - eTour TLR: tobhey/finegrained-traceability (req + solution links)

Does NOT invent labels. Writes:
  data/promise_nfr/nfr.csv
  data/damir_pure/spans.jsonl
  data/trace_etour/items.jsonl
  data/SOURCES.md
"""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DL = ROOT / "data" / "_downloads"
OUT = ROOT / "data"


def materialize_promise() -> int:
    arff = DL / "promise_raw" / "nfr" / "nfr.arff"
    out_dir = OUT / "promise_nfr"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "nfr.csv"
    rows = []
    in_data = False
    with arff.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            if line.upper().startswith("@DATA"):
                in_data = True
                continue
            if not in_data:
                continue
            # ProjectID,'Requirement text...',CLASS
            m = re.match(r"^(\d+)\s*,\s*'(.*)'\s*,\s*([A-Za-z]+)\s*$", line)
            if not m:
                # fallback: last comma splits class; first comma splits project
                try:
                    project, rest = line.split(",", 1)
                    text, label = rest.rsplit(",", 1)
                    text = text.strip().strip("'")
                    label = label.strip()
                except ValueError:
                    continue
            else:
                project, text, label = m.group(1), m.group(2), m.group(3)
            text = text.replace("''", "'").strip()
            rows.append({
                "id": f"p{project}-{len(rows)}",
                "ProjectID": project,
                "RequirementText": text,
                "class": label,
            })
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "ProjectID", "RequirementText", "class"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def _fix_mojibake(s: str) -> str:
    """Repair common UTF-8-as-latin1 mojibake in DAMIR-PURE Context fields."""
    try:
        return s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        repl = {
            "â€™": "'", "â€˜": "'", "â€œ": '"', "â€\x9d": '"', "â€": '"',
            "â€“": "-", "â€”": "-",
        }
        for a, b in repl.items():
            s = s.replace(a, b)
        return s


def _locate_pronoun(text: str, pronoun: str, position: str) -> tuple[int, int, str]:
    """Map (pronoun, Position) to [start, end) char offsets in `text`.

    DAMIR Position is an occurrence index that does not always match naive
    whitespace splits (tokenization differs from the release CSV). We therefore:
      1) collect whole-word matches of the pronoun;
      2) if Position points at a matching whitespace-token, use that;
      3) else pick the match whose whitespace-token index is closest to Position;
      4) else fall back to the first match / raw find.
    """
    p = pronoun
    low = text.lower()
    pl = p.lower()
    # whole-word-ish matches (allow trailing punctuation attached to token)
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        i = low.find(pl, start)
        if i < 0:
            break
        before_ok = i == 0 or not text[i - 1].isalnum()
        after = i + len(pl)
        after_ok = after >= len(text) or not text[after].isalnum()
        if before_ok and after_ok:
            matches.append((i, after))
        start = i + 1

    try:
        target = int(position)
    except ValueError:
        target = -1

    words = text.split()
    # Exact whitespace-token hit at 1-based Position
    if target >= 1 and target <= len(words) and pl in words[target - 1].lower():
        # map token -> char span via walk (preserves original spacing)
        pos = 0
        for wi, w in enumerate(words):
            j = text.find(w, pos)
            if j < 0:
                break
            if wi == target - 1:
                return j, j + len(w), "word_index"
            pos = j + len(w)

    if matches:
        if len(matches) == 1:
            a, b = matches[0]
            return a, b, "unique_match"
        if target >= 1:
            best = min(
                matches,
                key=lambda m: abs(_word_index_at(text, m[0]) - target),
            )
            return best[0], best[1], "nearest_word_index"
        return matches[0][0], matches[0][1], "first_match"

    i = low.find(pl)
    if i >= 0:
        return i, i + len(pl), "raw_find"
    return -1, -1, "missing"


def _word_index_at(text: str, char_i: int) -> int:
    """1-based whitespace-token index containing char_i."""
    if char_i <= 0:
        return 1
    prefix = text[:char_i]
    n = len(prefix.split())
    if prefix and not prefix[-1].isspace():
        return max(1, n)
    return max(1, n + 1)


def materialize_damir() -> int:
    src = DL / "DAMIR-PURE.csv"
    out_dir = OUT / "damir_pure"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "spans.jsonl"

    # Collapse candidate-antecedent rows into one item per (context, pronoun, position).
    # Gold span = the pronoun when AckUnack is Ack or Unack (ambiguous).
    groups: dict[tuple, dict] = {}
    with src.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ctx = _fix_mojibake(row["Context"].strip())
            pronoun = row["Pronoun"].strip()
            pos = row["Position"].strip()
            key = (ctx, pronoun, pos)
            g = groups.setdefault(key, {
                "context": ctx,
                "pronoun": pronoun,
                "position": pos,
                "labels": set(),
            })
            g["labels"].add(row["AckUnack"].strip())

    items = []
    for i, g in enumerate(groups.values()):
        text = g["context"]
        ambiguous = bool(g["labels"] & {"Ack", "Unack"})
        spans = []
        match_method = None
        if ambiguous:
            start, end, match_method = _locate_pronoun(
                text, g["pronoun"], g["position"])
            if start >= 0:
                spans.append({
                    "start": start,
                    "end": end,
                    "label": "anaphoric_ambiguity",
                    "pronoun": g["pronoun"],
                })
        items.append({
            "id": f"damir-{i}",
            "text": text,
            # Enables stratified pilot sampling (ambiguous vs unambiguous).
            "gold_label": "ambiguous" if ambiguous else "unambiguous",
            "spans": spans,
            "meta": {
                "ack_labels": sorted(g["labels"]),
                "position": g["position"],
                "pronoun": g["pronoun"],
                "match_method": match_method,
            },
        })

    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(items)


def materialize_etour() -> int:
    base = DL / "ftlr" / "datasets" / "eTour"
    out_dir = OUT / "trace_etour"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "items.jsonl"

    gold: dict[str, list[str]] = defaultdict(list)
    with (base / "etour_solution_links_english.txt").open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            left, right = line.split(":", 1)
            req = left.strip()
            code = right.strip()
            gold[req].append(code)

    # Candidate catalog: all code filenames + short head (keeps long-context without
    # dumping entire Java sources into every prompt).
    code_dir = base / "code"
    catalog_parts = []
    code_ids = []
    for path in sorted(code_dir.glob("*.java")):
        code_ids.append(path.name)
        head = path.read_text(encoding="utf-8", errors="replace").splitlines()[:25]
        catalog_parts.append(f"### {path.name}\n" + "\n".join(head))
    target_doc = "\n\n".join(catalog_parts)

    req_dir = base / "req"
    items = []
    for req_name, targets in sorted(gold.items()):
        req_path = req_dir / req_name
        if not req_path.exists():
            continue
        source = req_path.read_text(encoding="utf-8", errors="replace").strip()
        items.append({
            "id": req_name.replace(".txt", ""),
            "source_text": source,
            "target_doc": target_doc,
            "gold_target_ids": sorted(set(targets)),
            "meta": {
                "n_candidates": len(code_ids),
                "dataset": "eTour",
                "link_file": "etour_solution_links_english.txt",
            },
        })

    with out.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")

    # Keep a compact pointer to raw FTLR tree for audit (not required by harness).
    (out_dir / "RAW_SOURCE.txt").write_text(
        "Raw eTour files live at data/_downloads/ftlr/datasets/eTour/\n",
        encoding="utf-8",
    )
    return len(items)


def write_sources(n_promise: int, n_damir: int, n_etour: int) -> None:
    text = f"""# Dataset sources (locked 2026-09-05)

| Task | Local dir | Upstream | Notes |
|------|-----------|----------|-------|
| TF_Surface | `promise_nfr/` ({n_promise} rows) | [Zenodo 268542](https://zenodo.org/records/268542) PROMISE NFR | Cleland-Huang et al. |
| TF_Mid | `damir_pure/` ({n_damir} items) | [Figshare DAMIR-PURE](https://doi.org/10.6084/m9.figshare.29304650) | Anaphoric ambiguity on PURE subset (Ezzini et al. ICSE 2022) |
| TF_Reason | `trace_etour/` ({n_etour} UCs) | [tobhey/finegrained-traceability](https://github.com/tobhey/finegrained-traceability) eTour | Gold links: `etour_solution_links_english.txt` (orig. CoEST/TEFSE) |

Raw archives retained under `data/_downloads/` for provenance.
Harness-readable files only under the three dirs above.
"""
    (OUT / "SOURCES.md").write_text(text, encoding="utf-8")


def main() -> None:
    n1 = materialize_promise()
    n2 = materialize_damir()
    n3 = materialize_etour()
    write_sources(n1, n2, n3)
    print(f"promise_nfr={n1} damir_pure={n2} trace_etour={n3}")


if __name__ == "__main__":
    main()
