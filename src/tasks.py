"""Task-family loaders and prompt builders.

Three families span the surface -> reasoning axis (paper §3):
  TF_Surface : FR/NFR and NFR-subtype classification (PROMISE NFR + PURE labels)
  TF_Mid     : ambiguity/defect detection & clause-level extraction (PURE-derived)
  TF_Reason  : long-context trace-link recovery / cross-artifact reasoning

Datasets are NOT redistributed here. Place them under data/ and point paths in
config. Loaders raise a clear error if data is absent — they never invent items.
"""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

# Bump when a frozen prompt template changes (reproducibility control, logged
# per run). NOT a place to tune prompts for scores.
PROMPT_VERSION = "p3"


@dataclass
class Item:
    item_id: str
    prompt: str
    reference: dict          # labels (basis A) or context needed for basis B
    long_context: bool


class DatasetMissing(RuntimeError):
    pass


def _require(path: str, hint: str) -> str:
    if not os.path.exists(path):
        raise DatasetMissing(
            f"Dataset not found at {path}. {hint} "
            "The harness will not synthesize data."
        )
    return path


def _resolve(path: str) -> str:
    """Accept a directory (pick the single data file) or a direct file path."""
    if os.path.isdir(path):
        cands = [f for f in sorted(os.listdir(path))
                 if not f.startswith("._") and not f.startswith(".")
                 and f.lower().endswith((".csv", ".tsv", ".jsonl", ".json", ".arff"))]
        if not cands:
            raise DatasetMissing(f"No csv/tsv/jsonl/json/arff file under {path}.")
        return os.path.join(path, cands[0])
    return path


def load_task(task_cfg: dict, data_root: str = "data") -> list[Item]:
    kind = task_cfg["kind"]
    dataset = task_cfg["dataset"]
    path = os.path.join(data_root, dataset)
    if kind == "classification":
        _require(path, "Provide PROMISE NFR (+PURE) labeled CSV/JSONL "
                       "(columns: text,label [,id]).")
        return _load_classification(_resolve(path), task_cfg)
    if kind == "span_detection":
        _require(path, "Provide PURE-derived ambiguity/defect annotations "
                       "(JSONL: {id,text,spans:[{start,end,label}]}).")
        return _load_span(_resolve(path), task_cfg)
    if kind == "trace_link":
        _require(path, "Provide a trace-link dataset "
                       "(JSONL: {id,source_text,target_doc,gold_target_ids:[...]}).")
        return _load_trace(_resolve(path), task_cfg)
    raise ValueError(f"Unknown task kind: {kind}")


def _sniff_delim(path: str) -> str:
    return "\t" if path.lower().endswith(".tsv") else ","


def _pick(row: dict, names: list[str]) -> str | None:
    lower = {k.lower(): k for k in row}
    for n in names:
        if n in lower:
            return row[lower[n]]
    return None


def _load_classification(path: str, cfg: dict) -> list[Item]:
    """CSV/TSV/JSONL with a text field and a label field.

    Tolerant to common PROMISE NFR column names (RequirementText/class, etc.).
    """
    items: list[Item] = []
    if path.lower().endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                text = _pick(r, ["text", "requirement", "requirementtext", "sentence"])
                label = _pick(r, ["label", "class", "category", "nfr"])
                _need(text, label, path)
                items.append(_clf_item(r, text, label, i, cfg))
        return _nonempty(items, path)
    with open(path, encoding="utf-8", newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh, delimiter=_sniff_delim(path))):
            text = _pick(r, ["text", "requirement", "requirementtext", "sentence"])
            label = _pick(r, ["label", "class", "category", "nfr"])
            _need(text, label, path)
            items.append(_clf_item(r, text, label, i, cfg))
    return _nonempty(items, path)


def _clf_item(r: dict, text: str, label: str, i: int, cfg: dict) -> Item:
    iid = _pick(r, ["id", "item_id", "projectid"]) or f"clf-{i}"
    return Item(item_id=str(iid), prompt=text.strip(),
                reference={"gold_label": label.strip()},
                long_context=cfg.get("long_context", False))


def _load_span(path: str, cfg: dict) -> list[Item]:
    """JSONL: {id, text, spans:[{start,end,label}], optional gold_label}."""
    items: list[Item] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            text = _pick(r, ["text", "requirement", "sentence"])
            if text is None:
                raise DatasetMissing(f"Row {i} in {path} missing a text field.")
            spans = r.get("spans", [])
            iid = _pick(r, ["id", "item_id"]) or f"span-{i}"
            ref = {"gold_spans": spans}
            # Optional stratum for pilot sampling (e.g. ambiguous/unambiguous).
            gl = r.get("gold_label")
            if gl is None:
                gl = "ambiguous" if spans else "unambiguous"
            ref["gold_label"] = str(gl)
            items.append(Item(item_id=str(iid), prompt=text.strip(),
                              reference=ref,
                              long_context=cfg.get("long_context", False)))
    return _nonempty(items, path)


def _load_trace(path: str, cfg: dict) -> list[Item]:
    """JSONL: {id, source_text, target_doc, gold_target_ids:[...]}."""
    items: list[Item] = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            src = _pick(r, ["source_text", "source"])
            tgt = _pick(r, ["target_doc", "target", "document"])
            if src is None or tgt is None:
                raise DatasetMissing(
                    f"Row {i} in {path} needs source_text and target_doc.")
            gold = r.get("gold_target_ids", [])
            iid = _pick(r, ["id", "item_id"]) or f"trace-{i}"
            items.append(Item(item_id=str(iid), prompt=tgt,
                              reference={"source": src, "gold_target_ids": gold},
                              long_context=True))
    return _nonempty(items, path)


def _need(text, label, path):
    if text is None or label is None:
        raise DatasetMissing(
            f"{path}: could not find text/label columns. "
            "Rename to text,label or provide JSONL with those keys.")


class DuplicateItemIds(ValueError):
    pass


def _nonempty(items: list[Item], path: str) -> list[Item]:
    if not items:
        raise DatasetMissing(f"No usable rows parsed from {path}.")
    seen: set[str] = set()
    dups: set[str] = set()
    for it in items:
        if it.item_id in seen:
            dups.add(it.item_id)
        seen.add(it.item_id)
    if dups:
        raise DuplicateItemIds(
            f"{path}: duplicate item_id(s) {sorted(dups)[:5]} "
            f"(+{max(0, len(dups) - 5)} more). Item IDs must be unique.")
    return items


def build_prompt(task_cfg: dict, item: Item) -> str:
    """Frozen prompt templates (reproducibility control, NOT tuned for scores)."""
    # PROMISE NFR class codes (Cleland-Huang). Short glosses reduce format drift;
    # example avoids privileging a single frequent code (e.g. PE).
    nfr_labels = (
        "F (functional), A (availability), L (legal), LF (look-and-feel), "
        "MN (maintainability), O (operational), PE (performance), "
        "SC (scalability), SE (security), US (usability), FT (fault tolerance), "
        "PO (portability)"
    )
    templates = {
        "classification": (
            "Classify the software requirement into exactly one label from this set: "
            f"{nfr_labels}.\n"
            "Reply with ONLY the label code (e.g. F or US). No explanation.\n\n"
            "Requirement:\n{body}\n\nLabel:"
        ),
        "span_detection": (
            "Detect anaphoric ambiguity: mark only pronouns whose antecedent is "
            "unclear or contested in the requirement. Do NOT mark every pronoun.\n"
            "Return ONLY a JSON list of objects with 0-based half-open char offsets "
            '[start, end): {{"start": <int>, "end": <int>, '
            '"label": "anaphoric_ambiguity"}}.\n'
            "If there is no anaphoric ambiguity, return [].\n\n"
            "Requirement:\n{body}\n\nSpans:"
        ),
        "trace_link": (
            "Recover requirement-to-code trace links. "
            "Return ONLY a JSON list of target Java file names that implement the "
            "source use case (e.g. [\"Foo.java\", \"Bar.java\"]). "
            "Do not paste source code.\n\n"
            "Source use case:\n{source}\n\n"
            "Candidate code catalog (truncated):\n{body}\n\n"
            "Links:"
        ),
    }
    return templates[task_cfg["kind"]].format(**item.reference, body=item.prompt)
