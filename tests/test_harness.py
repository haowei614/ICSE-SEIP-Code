"""Validation tests for the pilot harness (PART 14).

Pure code-path tests on tiny SYNTHETIC fixtures. They do NOT download data, do NOT
run any model, and do NOT touch results/final or the paper macros.

Run: python -m tests.test_harness
"""
from __future__ import annotations

import json
import os
import tempfile

from src import tasks as T
from src import metrics as M
from src import sampling as S
from src.schema import RunRecord


def _write(path: str, text: str):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
# Loader + duplicate-id detection                                             #
# --------------------------------------------------------------------------- #
def test_loader_and_duplicate_ids():
    with tempfile.TemporaryDirectory() as d:
        ds = os.path.join(d, "clf")
        os.makedirs(ds)
        _write(os.path.join(ds, "data.csv"),
               "id,text,label\n1,The system shall log in 2s,performance\n"
               "2,Users can reset password,functional\n")
        items = T.load_task({"kind": "classification", "dataset": "clf"}, data_root=d)
        assert len(items) == 2

        dupdir = os.path.join(d, "dup")
        os.makedirs(dupdir)
        _write(os.path.join(dupdir, "data.csv"),
               "id,text,label\n1,a,functional\n1,b,performance\n")
        try:
            T.load_task({"kind": "classification", "dataset": "dup"}, data_root=d)
        except T.DuplicateItemIds:
            print("[pass] loader + duplicate-id rejected")
            return
        raise AssertionError("duplicate ids were NOT rejected")


# --------------------------------------------------------------------------- #
# Prompt builder                                                              #
# --------------------------------------------------------------------------- #
def test_prompt_builder():
    item = T.Item(item_id="x", prompt="The system shall respond.",
                  reference={"gold_label": "functional"}, long_context=False)
    p = T.build_prompt({"kind": "classification"}, item)
    assert "The system shall respond." in p and "Label:" in p
    assert T.PROMPT_VERSION  # provenance tag present
    print("[pass] prompt builder")


# --------------------------------------------------------------------------- #
# Parser + metrics + malformed/empty/unexpected                              #
# --------------------------------------------------------------------------- #
def test_metrics_classification_paths():
    ref = {"gold_label": "performance"}
    q, det = M.score("A", "classification", "Label: performance", ref)
    assert q == 1.0 and det["correct"] == 1 and not det["malformed"]
    # unexpected label -> incorrect, not malformed
    q2, det2 = M.score("A", "classification", "banana", ref)
    assert q2 == 0.0 and det2["malformed"] is False
    # empty output -> malformed
    q3, det3 = M.score("A", "classification", "   ", ref)
    assert q3 == 0.0 and det3["malformed"] is True
    print("[pass] classification metric + malformed/empty/unexpected")


def test_metrics_trace_and_malformed_json():
    ref = {"gold_target_ids": ["A", "B", "C"]}
    q, det = M.score("A", "trace_link", '["A","B","D"]', ref)
    assert abs(q - 2 / 3) < 1e-9 and det["malformed"] is False
    # no JSON array present -> malformed
    q2, det2 = M.score("A", "trace_link", "no links here", ref)
    assert det2["malformed"] is True and q2 == 0.0
    print("[pass] trace-link metric + malformed JSON")


def test_metrics_span():
    ref = {"gold_spans": [{"start": 0, "end": 4, "label": "amb"}]}
    out = '[{"start":0,"end":4,"label":"amb"}]'
    q, det = M.score("A", "span_detection", out, ref)
    assert q == 1.0 and det["malformed"] is False
    # Soft IoU: off-by-one end still matches
    out2 = '[{"start":0,"end":5,"label":"amb"}]'
    q2, det2 = M.score("A", "span_detection", out2, ref)
    assert q2 == 1.0 and det2["tp"] == 1
    # Empty-empty agreement
    q3, _ = M.score("A", "span_detection", "[]", {"gold_spans": []})
    assert q3 == 1.0
    # p3a: malformed does not receive the empty-gold credit.
    prose = "no json here"
    q_leg, d_leg = M.score("A", "span_detection", prose, {"gold_spans": [], "gold_label": "unambiguous"})
    q_new, d_new = M.score(
        "A", "span_detection", prose,
        {"gold_spans": [], "gold_label": "unambiguous"}, mid_metric="p3a")
    assert q_leg == 1.0 and d_leg["malformed"] is True
    assert q_new == 0.0 and d_new["combined"] == 0.0 and d_new["decision"] == 0.0
    amb = {"gold_spans": [{"start": 0, "end": 4, "label": "amb"}], "gold_label": "ambiguous"}
    q_hit, d_hit = M.score("A", "span_detection", out, amb, mid_metric="p3a")
    q_miss, d_miss = M.score("A", "span_detection", "[]", amb, mid_metric="p3a")
    assert q_hit == 1.0 and d_hit["decision"] == 1.0
    assert q_miss == 0.0 and d_miss["span_f1"] == 0.0 and d_miss["decision"] == 0.0
    print("[pass] span metric")


# --------------------------------------------------------------------------- #
# Deterministic + stratified sampling                                         #
# --------------------------------------------------------------------------- #
def test_deterministic_sampling():
    items = [T.Item(item_id=f"i{k}", prompt="p",
                    reference={"gold_label": "a" if k % 3 else "b"},
                    long_context=False) for k in range(30)]
    id_of = lambda it: it.item_id
    a = S.deterministic_sample(items, 10, seed=1, id_of=id_of)
    b = S.deterministic_sample(items, 10, seed=1, id_of=id_of)
    c = S.deterministic_sample(items, 10, seed=2, id_of=id_of)
    assert [id_of(x) for x in a] == [id_of(x) for x in b]      # reproducible
    assert [id_of(x) for x in a] != [id_of(x) for x in c]      # seed-sensitive
    # stratified keeps both classes present
    strat = S.stratified_sample(items, 10, seed=1, id_of=id_of,
                                stratum_of=lambda it: it.reference["gold_label"])
    labels = {it.reference["gold_label"] for it in strat}
    assert labels == {"a", "b"} and len(strat) == 10
    print("[pass] deterministic + stratified sampling")


# --------------------------------------------------------------------------- #
# Result logger round-trip                                                    #
# --------------------------------------------------------------------------- #
def test_result_logger_roundtrip():
    rec = RunRecord(experiment_id="e", task_family="TF_Surface",
                    task_name="nfr", dataset="promise_nfr", item_id="1",
                    model_id="m", precision="Q4_K_M", measurement_basis="A",
                    seed=1, malformed_output=True, oom=False)
    d = json.loads(rec.to_json())
    for key in ("experiment_id", "task_family", "precision", "malformed_output",
                "oom", "timeout", "exception", "peak_memory_mb", "ttft_s"):
        assert key in d, f"missing schema field: {key}"
    assert d["malformed_output"] is True and d["schema_version"] == "2.0"
    print("[pass] result logger round-trip (full schema)")


def test_cloud_backend_requires_key():
    from src.backends import build_backend
    try:
        build_backend(
            "cloud", "claude_sonnet_45", "CLOUD",
            provider="anthropic",
            api_model="claude-sonnet-4-5",
            key_env="APSEC_MISSING_CLOUD_KEY_FOR_TEST",
        )
    except RuntimeError as e:
        assert "APSEC_MISSING_CLOUD_KEY_FOR_TEST" in str(e)
        print("[pass] cloud backend requires API key env")
        return
    raise AssertionError("cloud backend did not reject missing key")


def test_prompt_format_passthrough_without_tokenizer():
    from src.prompt_format import apply_hf_chat_template, grammar_for_task_kind
    raw = "Return ONLY a JSON list.\nLinks:"
    assert apply_hf_chat_template(None, raw, {"use_chat_template": True}) == raw
    assert apply_hf_chat_template(
        None, raw, {"use_chat_template": False}) == raw
    assert grammar_for_task_kind("trace_link") and "root" in grammar_for_task_kind(
        "trace_link")
    assert grammar_for_task_kind("classification") is None
    print("[pass] prompt_format passthrough + grammar map")


if __name__ == "__main__":
    test_loader_and_duplicate_ids()
    test_prompt_builder()
    test_metrics_classification_paths()
    test_metrics_trace_and_malformed_json()
    test_metrics_span()
    test_deterministic_sampling()
    test_result_logger_roundtrip()
    test_cloud_backend_requires_key()
    test_prompt_format_passthrough_without_tokenizer()
    print("ALL HARNESS TESTS PASSED")
