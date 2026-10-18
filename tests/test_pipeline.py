"""Format/logic tests for loaders + Option-A scoring.

These use tiny SYNTHETIC FIXTURES to test code paths only. They do NOT produce
experimental results and never touch results/ or the paper macros.
Run: python -m tests.test_pipeline
"""
from __future__ import annotations

import json
import os
import tempfile

from src import tasks as T
from src import metrics as M


def _write(path: str, text: str):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_classification():
    with tempfile.TemporaryDirectory() as d:
        ds = os.path.join(d, "clf")
        os.makedirs(ds)
        _write(os.path.join(ds, "data.csv"),
               "id,text,label\n1,The system shall log in 2s,performance\n"
               "2,Users can reset password,functional\n")
        items = T.load_task({"kind": "classification", "dataset": "clf"}, data_root=d)
        assert len(items) == 2
        q, det = M.score("A", "classification", "Label: performance",
                         items[0].reference)
        assert q == 1.0 and det["correct"] == 1
        q2, _ = M.score("A", "classification", "functional", items[0].reference)
        assert q2 == 0.0
    print("[pass] classification loader + macro scoring")


def test_trace_links():
    with tempfile.TemporaryDirectory() as d:
        ds = os.path.join(d, "trace")
        os.makedirs(ds)
        row = {"id": "t1", "source_text": "REQ-1", "target_doc": "doc...",
               "gold_target_ids": ["A", "B", "C"]}
        _write(os.path.join(ds, "data.jsonl"), json.dumps(row) + "\n")
        items = T.load_task({"kind": "trace_link", "dataset": "trace"}, data_root=d)
        assert items[0].long_context is True
        # predict A,B,D vs gold A,B,C -> tp=2, fp=1, fn=1 -> P=R=2/3 -> F1=2/3
        q, det = M.score("A", "trace_link", '["A","B","D"]', items[0].reference)
        assert abs(q - 2/3) < 1e-9, (q, det)
    print("[pass] trace-link loader + link-F1")


def test_single_basis_guard():
    try:
        M.assert_single_basis("A", [{"id": "x", "measurement_basis": "B"}])
    except ValueError:
        print("[pass] mixed-basis correctly rejected")
        return
    raise AssertionError("mixed basis was NOT rejected")


def test_basis_B_requires_cloud():
    try:
        M.score("B", "classification", "performance", {"gold_label": "performance"})
    except ValueError:
        print("[pass] basis B without cloud reference correctly rejected")
        return
    raise AssertionError("basis B without cloud output was NOT rejected")


if __name__ == "__main__":
    test_classification()
    test_trace_links()
    test_single_basis_guard()
    test_basis_B_requires_cloud()
    print("ALL TESTS PASSED")
