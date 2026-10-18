"""Parsers for TF-Mid and TF-Reason.

The default scorer is lenient_unique: one distinct schema-valid JSON value
after non-json code fences are ignored. Last, first, label-required, and
legacy_greedy stay as sensitivity checks. strict is only the whole-output check.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import _parse_task_json, score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_MID = os.path.join(
    ROOT, "results", "pillar1_aligned_v2", "TF_Mid__llama31_8b_instruct__F16.jsonl")

SPAN = '[{"start": 0, "end": 4, "label": "anaphoric_ambiguity"}]'
LINKS = '["Foo.java", "Bar.java"]'
AMB = {"gold_spans": [{"start": 0, "end": 4, "label": "anaphoric_ambiguity"}],
       "gold_label": "ambiguous"}
LINKS_REF = {"gold_target_ids": ["Foo.java"]}


def _ok(kind, text, parser):
    return _parse_task_json(text, kind, parser) is not None


def test_synthetic_mid_and_reason():
    code_mid = "spans = []\n```json\n" + SPAN + "\n```"
    code_reason = 'print(["A.java"])\n' + LINKS
    families = {
        "span_detection": [
            ("clean", SPAN, True, True, True),
            ("fenced", "```json\n" + SPAN + "\n```", True, True, True),
            ("prose", "Here is the list:\n" + SPAN, True, True, False),
            ("code", code_mid, False, True, False),
            ("none", "no json here", False, False, False),
            ("multi", SPAN + "\n" + '[{"start": 1, "end": 2, "label": "x"}]', False, True, False),
        ],
        "trace_link": [
            ("clean", LINKS, True, True, True),
            ("fenced", "```\n" + LINKS + "\n```", True, True, True),
            ("prose", "The files are " + LINKS, True, True, False),
            ("code", code_reason, False, True, False),
            ("none", "see the classes above", False, False, False),
            ("multi", '["A.java"]\n' + LINKS, False, True, False),
        ],
    }
    for kind, cases in families.items():
        assert len(cases) >= 5
        for name, text, legacy, lenient, strict in cases:
            assert _ok(kind, text, "legacy_greedy") is legacy, (kind, name, "legacy")
            assert _ok(kind, text, "lenient") is lenient, (kind, name, "lenient")
            assert _ok(kind, text, "strict") is strict, (kind, name, "strict")
    # Lenient keeps the last valid value. lenient_first keeps the first.
    last = _parse_task_json(
        '["A.java"]\n["Foo.java", "Bar.java"]', "trace_link", "lenient")
    first = _parse_task_json(
        '["A.java"]\n["Foo.java", "Bar.java"]', "trace_link", "lenient_first")
    assert last == ["Foo.java", "Bar.java"]
    assert first == ["A.java"]
    # Strict rejects a fence that is not the whole output, and rejects a bad schema.
    assert _parse_task_json('```json\n{"start": 1}\n```', "span_detection", "strict") is None
    assert _parse_task_json("[true]", "trace_link", "strict") is None
    # Missing label is schema-valid for the primary parser, not for the sensitivity.
    missing = '[{"start": 53, "end": 58}]'
    assert _parse_task_json(missing, "span_detection", "lenient") == [
        {"start": 53, "end": 58}]
    assert _parse_task_json(missing, "span_detection", "lenient_label_required") is None
    assert _parse_task_json(missing, "span_detection", "strict") == [
        {"start": 53, "end": 58}]
    assert _parse_task_json(
        '[{"start": 1, "end": 2, "label": 5}]', "span_detection", "lenient") is None
    print("[pass] synthetic mid and reason parsers")


def test_llama_mid_f16_fenced_json():
    assert os.path.exists(LLAMA_MID), LLAMA_MID
    rows = [json.loads(line) for line in open(LLAMA_MID, encoding="utf-8") if line.strip()]
    dec = json.JSONDecoder()

    def any_span_list(text):
        for i, ch in enumerate(text):
            if ch != "[":
                continue
            try:
                val, _end = dec.raw_decode(text[i:])
            except json.JSONDecodeError:
                continue
            if (isinstance(val, list) and val
                    and all(isinstance(x, dict) and "start" in x and "end" in x for x in val)):
                return True
        return False

    recovered = []
    for row in rows:
        raw = row.get("raw_model_output") or ""
        legacy = _parse_task_json(raw, "span_detection", "legacy_greedy")
        if legacy is None and any_span_list(raw):
            recovered.append(row["item_id"])
            assert _parse_task_json(raw, "span_detection", "strict") is None, row["item_id"]
    assert recovered == [
        "damir-92", "damir-13", "damir-27", "damir-12", "damir-110", "damir-95",
        "damir-80", "damir-75", "damir-30", "damir-15", "damir-9", "damir-88",
        "damir-40", "damir-1", "damir-59", "damir-3", "damir-50", "damir-84",
        "damir-89",
    ]
    by_id = {r["item_id"]: r for r in rows}
    raw1 = by_id["damir-1"]["raw_model_output"]
    # damir-1 writes the span list first, then Python that ends in a later [].
    assert _parse_task_json(raw1, "span_detection", "lenient") == []
    assert _parse_task_json(raw1, "span_detection", "lenient_first") == [
        {"start": 64, "end": 67, "label": "anaphoric_ambiguity"},
        {"start": 83, "end": 86, "label": "anaphoric_ambiguity"},
    ]
    got27 = _parse_task_json(by_id["damir-27"]["raw_model_output"], "span_detection", "lenient")
    assert got27 == [
        {"start": 64, "end": 67, "label": "anaphoric_ambiguity"},
        {"start": 104, "end": 107, "label": "anaphoric_ambiguity"},
    ]
    ref = {"gold_spans": [], "gold_label": "unambiguous"}
    _q, detail = score("A", "span_detection", by_id["damir-1"]["raw_model_output"], ref,
                       parser="lenient", mid_metric="p3a")
    assert detail["malformed"] is False and _q == 1.0
    _q2, detail2 = score("A", "span_detection", by_id["damir-1"]["raw_model_output"], ref,
                         parser="strict", mid_metric="p3a")
    assert detail2["malformed"] is True and _q2 == 0.0
    print(f"[pass] 19 Llama Mid aligned-F16 cases: {recovered}")


def _packet(case_id: str) -> dict:
    for name in ("wave1_packets.jsonl", "wave2_packets.jsonl"):
        path = os.path.join(ROOT, "coding", "samples", name)
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["case_id"] == case_id:
                return row
    raise AssertionError(case_id)


def test_lenient_unique():
    raw1 = next(
        json.loads(line)["raw_model_output"]
        for line in open(LLAMA_MID, encoding="utf-8")
        if line.strip() and json.loads(line)["item_id"] == "damir-1")
    spans = [
        {"start": 64, "end": 67, "label": "anaphoric_ambiguity"},
        {"start": 83, "end": 86, "label": "anaphoric_ambiguity"},
    ]
    assert _parse_task_json(raw1, "span_detection", "lenient_unique") == spans
    assert _parse_task_json(raw1, "span_detection", "lenient") == []

    assert _parse_task_json(
        _packet("W1-005")["raw_model_output"], "span_detection", "lenient_unique") is None
    links = [
        "RefreshmentPointAgencyManager.java",
        "AdvertisementManager.java",
        "DBRefreshmentPoint.java",
        "DBRefreshmentPointOperator.java",
        "RefreshmentPointCommonManager.java",
    ]
    assert _parse_task_json(
        _packet("W2-015")["raw_model_output"], "trace_link", "lenient_unique") == links
    assert _parse_task_json(
        _packet("W2-029")["raw_model_output"], "span_detection", "lenient_unique") is None
    assert _parse_task_json(
        _packet("W2-043")["raw_model_output"], "span_detection", "lenient_unique") == [
        {"start": 15, "end": 16, "label": "anaphoric_ambiguity"}]

    twice = SPAN + "\n\n" + SPAN
    assert _parse_task_json(twice, "span_detection", "lenient_unique") == json.loads(SPAN)
    fenced = "```python\nresult = []\n```\n```json\n" + SPAN + "\n```"
    assert _parse_task_json(fenced, "span_detection", "lenient_unique") == json.loads(SPAN)
    assert _parse_task_json(
        fenced, "span_detection", "lenient_unique_no_fence_skip") is None
    print("[pass] lenient_unique fence skip and distinct-value rule")


def test_default_parser_is_unique():
    prose = "note\n" + SPAN
    _q, detail = score("A", "span_detection", prose, AMB, mid_metric="p3a")
    assert detail["parser"] == "lenient_unique"
    assert detail["malformed"] is False and _q == 1.0
    two = SPAN + "\n" + '[{"start": 1, "end": 2, "label": "x"}]'
    _qu, amb = score("A", "span_detection", two, AMB, mid_metric="p3a")
    assert amb["malformed"] is True and _qu == 0.0
    _qs, strict = score("A", "span_detection", prose, AMB, parser="strict", mid_metric="p3a")
    assert strict["malformed"] is True and _qs == 0.0
    print("[pass] default parser is lenient_unique")


if __name__ == "__main__":
    test_synthetic_mid_and_reason()
    test_llama_mid_f16_fenced_json()
    test_lenient_unique()
    test_default_parser_is_unique()
    print("ALL PARSER TESTS PASSED")
