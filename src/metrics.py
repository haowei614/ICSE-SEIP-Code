"""Quality scoring under the single locked measurement basis.

Basis A (labeled): score model output against gold labels/spans/links.
Basis B (differential): score agreement/divergence vs the SAME cloud frontier
         reference; report NO absolute correctness.

CRITICAL: the same basis is applied to ALL task families. `score()` enforces this
by refusing to mix per-task bases.
"""
from __future__ import annotations

from typing import Any, Optional


def score(basis: str, task_kind: str, output_text: str,
          reference: dict, cloud_output: Optional[str] = None,
          mid_metric: str = "legacy",
          parser: str = "lenient_unique") -> tuple[float, dict]:
    """Score one item.

    ``mid_metric="legacy"`` is the historical span F1, including empty-gold
    credit when the output does not parse. ``mid_metric="p3a"`` is the
    corrected Mid score documented in analysis/MID_METRIC.md. Other tasks
    ignore that flag.

    ``parser`` selects how Mid and Reason JSON is read. The default is
    ``lenient_unique``: after dropping non-json code fences, exactly one
    distinct schema-valid JSON value. ``strict`` checks whether the whole
    output is that JSON and must not be used as quality or q_ok.
    """
    if basis == "A":
        return _score_labeled(
            task_kind, output_text, reference, mid_metric=mid_metric, parser=parser)
    if basis == "B":
        if cloud_output is None:
            raise ValueError(
                "Basis B requires the cloud frontier reference output for the "
                "same item. Cannot score differentially without it."
            )
        return _score_differential(task_kind, output_text, cloud_output)
    raise ValueError(f"measurement_basis must be 'A' or 'B', got {basis!r}")


def _score_labeled(task_kind: str, output_text: str, reference: dict,
                   mid_metric: str = "legacy", parser: str = "lenient_unique"):
    """Basis A. Per-item score contributing to a macro/micro metric.

    NOTE: these functions score ONE item. Corpus-level macro-F1 / span-F1 /
    link-F1 are computed by aggregating the per-item detail dicts in
    analysis/aggregate.py. We return a per-item quality proxy for convenience,
    but the reported metric is the corpus aggregate over `detail`.
    """
    if task_kind == "classification":
        gold = str(reference["gold_label"]).strip().lower()
        pred = _extract_label(output_text)
        malformed = pred == ""
        correct = int((not malformed) and pred == gold)
        return float(correct), {"metric": "accuracy_item", "pred": pred,
                                "gold": gold, "correct": correct,
                                "malformed": malformed}
    if task_kind == "span_detection":
        return _score_span(output_text, reference, mid_metric=mid_metric, parser=parser)
    if task_kind == "trace_link":
        raw = _parse_task_json(output_text, "trace_link", parser)
        malformed = raw is None
        gold = set(map(str, reference.get("gold_target_ids", [])))
        pred = set(map(str, raw or []))
        tp = len(gold & pred)
        return _f1(tp, len(pred) - tp, len(gold) - tp), {
            "metric": "link_f1_item", "tp": tp, "pred": sorted(pred),
            "pred_n": len(pred), "gold_n": len(gold), "malformed": malformed,
            "parser": parser}
    raise ValueError(f"Unknown task kind for labeled scoring: {task_kind}")


def _score_span(output_text: str, reference: dict, mid_metric: str = "legacy",
                parser: str = "lenient_unique"):
    """Span item score. ``legacy`` keeps empty-gold credit for unparsed output.

    ``p3a`` scores a malformed item as 0. The returned quality under p3a is the
    combined item score: span F1 on ambiguous gold, decision correctness on
    unambiguous gold. See analysis/MID_METRIC.md.
    """
    if mid_metric not in ("legacy", "p3a"):
        raise ValueError(f"mid_metric must be 'legacy' or 'p3a', got {mid_metric!r}")
    raw = _parse_task_json(output_text, "span_detection", parser)
    malformed = raw is None
    gold = _norm_spans(reference.get("gold_spans", []))
    pred = _norm_spans(raw or [])
    tp, matched_pred, matched_gold = _match_spans_soft(gold, pred, iou_thresh=0.5)
    legacy = _f1(tp, len(pred) - matched_pred, len(gold) - matched_gold)
    if reference.get("gold_label") is not None:
        gold_amb = str(reference["gold_label"]).strip().lower() == "ambiguous"
    else:
        gold_amb = len(gold) > 0
    pred_amb = (not malformed) and len(pred) > 0
    decision = 0.0 if malformed else float(pred_amb == gold_amb)
    span_f1 = 0.0 if malformed else legacy
    if malformed:
        combined = 0.0
    elif gold_amb:
        combined = span_f1
    else:
        combined = 1.0 if len(pred) == 0 else 0.0
    detail = {
        "metric": "span_f1_item" if mid_metric == "legacy" else "mid_p3a_item",
        "tp": tp,
        "pred": sorted(map(str, pred)),
        "pred_n": 0 if malformed else len(pred),
        "gold_n": len(gold),
        "malformed": malformed,
        "match": "iou>=0.5",
        "mid_metric": mid_metric,
        "gold_ambiguous": gold_amb,
        "span_f1": span_f1,
        "decision": decision,
        "combined": combined,
        "legacy_quality": legacy,
        "parser": parser,
    }
    if mid_metric == "legacy":
        return legacy, detail
    return combined, detail


def _f1(tp: int, fp: int, fn: int) -> float:
    # Empty gold + empty pred = perfect agreement for span/link tasks.
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _strip_think(text: str) -> str:
    """Drop Qwen3 <think>...</think> (or leftover think body) before scoring."""
    import re
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # If the model emitted think text without tags (greedy continuation), keep last
    # non-empty line as a weak fallback handled by _extract_label.
    return t.strip()


def _extract_label(text: str) -> str:
    """First usable label token. Recognizes PROMISE NFR codes anywhere early."""
    import re
    t = _strip_think(text)
    if not t:
        return ""
    known = {"f", "a", "l", "lf", "mn", "o", "pe", "sc", "se", "us", "ft", "po"}
    # Prefer an exact known code on its own line / after 'Label:'
    for line in t.splitlines()[:8]:
        s = line.strip()
        if ":" in s:
            s = s.split(":", 1)[1].strip()
        s = s.strip(" .`\"'[](){}")
        if s.lower() in known:
            return s.lower()
        # lone token
        tok = re.split(r"\s+", s)[0].strip(" .`\"'[](){}").lower()
        if tok in known:
            return tok
    # fallback: first token of first line
    first = t.splitlines()[0]
    if ":" in first:
        first = first.split(":", 1)[1]
    return first.strip().strip(".").lower()


def _extract_json(text: str):
    """Historical parser: first greedy ``[`` ... ``]`` slice, then json.loads."""
    import json
    import re
    t = _strip_think(text)
    m = re.search(r"\[.*\]", t, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _valid_task_json(value, task_kind: str, label_required: bool = False) -> bool:
    """Schema used by C1. Booleans are not integers.

    Mid requires integer ``start`` and ``end``. ``label`` is optional unless
    ``label_required`` (the sensitivity variant). Extra keys are allowed when
    the label is optional. Reason is a list of strings.
    """
    if task_kind == "span_detection":
        if not isinstance(value, list):
            return False
        for obj in value:
            if not isinstance(obj, dict):
                return False
            if type(obj.get("start")) is not int or type(obj.get("end")) is not int:
                return False
            if label_required:
                if set(obj) != {"start", "end", "label"}:
                    return False
                if not isinstance(obj["label"], str):
                    return False
            elif "label" in obj and not isinstance(obj["label"], str):
                return False
        return True
    if task_kind == "trace_link":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    raise ValueError(f"No JSON schema for task kind {task_kind!r}")


def _unwrap_one_fence(text: str):
    """Strip one surrounding ``` or ```json fence. None if the fence is not alone."""
    import re
    t = text.strip()
    if "```" not in t:
        return t
    lines = t.splitlines()
    if not re.fullmatch(r"```(?:json)?", lines[0].strip(), flags=re.IGNORECASE):
        return None
    if len(lines) < 2 or lines[-1].strip() != "```":
        return None
    inner = "\n".join(lines[1:-1])
    if "```" in inner:
        return None
    return inner


def _parse_strict(text: str, task_kind: str):
    import json
    inner = _unwrap_one_fence(text if text is not None else "")
    if inner is None:
        return None
    try:
        value = json.loads(inner)
    except Exception:
        return None
    if not _valid_task_json(value, task_kind):
        return None
    return value


def _blank_nonjson_fences(text: str) -> str:
    """Blank ``` fences whose language tag is present and not json.

    Untagged fences and ```json fences stay. An unclosed tagged fence blanks
    through the end of the text. Comparison of the tag is case-insensitive.
    """
    import re
    if not text or "```" not in text:
        return text or ""
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.rstrip("\r\n")
        opened = re.match(r"^[ \t]*```([^`]*)$", stripped)
        if not opened:
            out.append(raw)
            i += 1
            continue
        info = opened.group(1).strip()
        lang = info.split()[0] if info else ""
        skip = bool(lang) and lang.lower() != "json"
        if not skip:
            out.append(raw)
            i += 1
            continue
        out.append(" " * len(raw))
        i += 1
        while i < len(lines):
            inner = lines[i]
            inner_stripped = inner.rstrip("\r\n")
            out.append(" " * len(inner))
            i += 1
            if re.match(r"^[ \t]*```[ \t]*$", inner_stripped):
                break
    return "".join(out)


def _canonical_json(value) -> str:
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def distinct_schema_values(text: str, task_kind: str, *,
                           skip_code_fences: bool = True,
                           label_required: bool = False) -> list:
    """Schema-valid JSON values in scan order, with duplicates removed.

    List order is part of the identity. Dict key order is not.
    """
    import json
    src = text or ""
    if skip_code_fences:
        src = _blank_nonjson_fences(src)
    dec = json.JSONDecoder()
    found = []
    seen = set()
    for i, ch in enumerate(src):
        if ch not in "[{":
            continue
        try:
            value, _end = dec.raw_decode(src[i:])
        except json.JSONDecodeError:
            continue
        if not _valid_task_json(value, task_kind, label_required=label_required):
            continue
        key = _canonical_json(value)
        if key in seen:
            continue
        seen.add(key)
        found.append(value)
    return found


def _parse_lenient(text: str, task_kind: str, which: str = "last",
                   label_required: bool = False):
    import json
    dec = json.JSONDecoder()
    src = text or ""
    found = None
    for i, ch in enumerate(src):
        if ch not in "[{":
            continue
        try:
            value, _end = dec.raw_decode(src[i:])
        except json.JSONDecodeError:
            continue
        if not _valid_task_json(value, task_kind, label_required=label_required):
            continue
        if which == "first":
            return value
        found = value
    return found


def only_json_compliant(text: str, task_kind: str) -> bool:
    """True when the whole output is schema-valid JSON after one fence strip.

    This is the strict check. It is not a quality score.
    """
    return _parse_strict(text, task_kind) is not None


def _parse_unique(text: str, task_kind: str, skip_code_fences: bool):
    values = distinct_schema_values(
        text, task_kind, skip_code_fences=skip_code_fences, label_required=False)
    if len(values) == 1:
        return values[0]
    return None


def _parse_task_json(text: str, task_kind: str, parser: str = "lenient_unique"):
    """Return a schema value, or None when the chosen parser rejects the output."""
    if parser == "legacy_greedy":
        return _extract_json(text)
    if parser == "strict":
        return _parse_strict(text, task_kind)
    if parser == "lenient":
        return _parse_lenient(text, task_kind, which="last", label_required=False)
    if parser == "lenient_first":
        return _parse_lenient(text, task_kind, which="first", label_required=False)
    if parser == "lenient_label_required":
        return _parse_lenient(text, task_kind, which="last", label_required=True)
    if parser == "lenient_unique":
        return _parse_unique(text, task_kind, skip_code_fences=True)
    if parser == "lenient_unique_no_fence_skip":
        return _parse_unique(text, task_kind, skip_code_fences=False)
    raise ValueError(
        "parser must be 'lenient_unique', 'lenient_unique_no_fence_skip', "
        "'lenient', 'lenient_first', 'lenient_label_required', 'strict', "
        f"or 'legacy_greedy', got {parser!r}")


def _norm_spans(spans) -> set:
    out = set()
    for s in spans:
        if isinstance(s, dict):
            out.add((s.get("start"), s.get("end"), str(s.get("label", "")).lower()))
        elif isinstance(s, (list, tuple)) and len(s) >= 2:
            lab = str(s[2]).lower() if len(s) > 2 else ""
            out.add((s[0], s[1], lab))
    return out


def _span_iou(a, b) -> float:
    """Char-span IoU for [start, end) tuples (label ignored)."""
    a0, a1 = int(a[0]), int(a[1])
    b0, b1 = int(b[0]), int(b[1])
    if a1 <= a0 or b1 <= b0:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter == 0:
        return 0.0
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / union if union else 0.0


def _match_spans_soft(gold: set, pred: set, iou_thresh: float = 0.5
                      ) -> tuple[int, int, int]:
    """Greedy 1-1 matching: same label + IoU >= thresh (falls back to exact).

    Returns (tp, n_matched_pred, n_matched_gold).
    """
    if not gold and not pred:
        return 0, 0, 0
    # Exact hits first
    exact = gold & pred
    g_left = [s for s in gold if s not in exact]
    p_left = [s for s in pred if s not in exact]
    pairs = []
    for g in g_left:
        for p in p_left:
            if str(g[2]).lower() != str(p[2]).lower():
                continue
            iou = _span_iou(g, p)
            if iou >= iou_thresh:
                pairs.append((iou, g, p))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_g, used_p = set(), set()
    soft = 0
    for _, g, p in pairs:
        if g in used_g or p in used_p:
            continue
        used_g.add(g)
        used_p.add(p)
        soft += 1
    tp = len(exact) + soft
    return tp, len(exact) + soft, len(exact) + soft


def _score_differential(task_kind: str, output_text: str, cloud_output: str):
    """Basis B. Agreement with the cloud reference. NO absolute correctness.

    Suggested: exact/normalized agreement for labels; set-overlap (Jaccard/F1)
    for spans and links; report divergence categories in detail.
    """
    raise NotImplementedError(
        "Implement differential agreement vs the cloud reference. Do not return a "
        "placeholder number — leave quality=None until this is real."
    )


def assert_single_basis(basis: str, tasks: list[dict]) -> None:
    """Guard against the forbidden mixed-yardstick design (paper §3, §4.3)."""
    if basis not in ("A", "B"):
        raise ValueError("measurement_basis must be exactly one of 'A' or 'B'.")
    # By construction all tasks share `basis`; this function documents/enforces it.
    for t in tasks:
        if t.get("measurement_basis", basis) != basis:
            raise ValueError(
                f"Task {t.get('id')} uses a different basis. Mixing labeled and "
                "cloud-reference scoring across task families is forbidden."
            )
