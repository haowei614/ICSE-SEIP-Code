# Failure-Mechanism Codebook (v1) — Pillar 2

Status: **v1 (Wave~1 + Wave~2; $n{=}100$).** Sensitizing concepts orient coding; a category
is a **finding** only if observed with supporting cases. M5/M6 promoted to findings
after Wave~1 and confirmed under quantization in Wave~2. M1--M4 remain sensitizing
(not primary-attested). One secondary M3 (W2-032) not promoted.

## Unit of analysis
One failure case = one (model, quant, task, item) output judged a failure under the
locked measurement basis. Sample achieved: **100** (Wave~1 FP16 $n{=}48$ + Wave~2
GGUF $n{=}52$), stratified by task family, model, and quantization. Second coder:
Wave~1 subset $n{=}36$; report Cohen's kappa.

## Coding procedure
1. Coder reads the input artifact, the model output, and (basis A) the gold or
   (basis B) the cloud reference output.
2. Assign one PRIMARY mechanism code; optionally SECONDARY codes.
3. Assign a VISIBILITY code (see below) — this is the acceptance-critical axis.
4. Record a short free-text rationale and a quote span.
5. Disagreements resolved by discussion; unresolved -> "contested", excluded from
   prevalence claims but reported as a count.

## Observed findings (v1)
| Code | Label | Definition | Prevalence (primary) | Typical visibility |
|------|-------|------------|----------------------|--------------------|
| **M5** | Format / schema noncompliance | Output violates required response form so automated parse fails | 15/100 (Wave~1: 9; legacy Wave~2: 6) | **V0** (by definition under Schema) |
| **M6** | Wrong-link / wrong-span / wrong-label | Well-formed output semantically incorrect vs gold | 85/100 (Wave~1: 39; legacy Wave~2: 46) | V1 or **V2** |

**Merge/split log:** none since v0.1. M5/M6 added 2026-09-07 as sensitizing; promoted
2026-09-07 after Wave~1 coding; Wave~2 confirms M5 rises under GGUF (52% of Wave~2
primaries) while M6 remains the silent-drift carrier when parse succeeds.

## Sensitizing only (not findings)
| Code | Label | Notes |
|------|-------|-------|
| M1 | Modal-verb / normative-strength drift | 0 primary |
| M2 | Loss of implicit requirements | 0 primary |
| M3 | Late-context degradation | 1 secondary (W2-032); not promoted |
| M4 | Terminology inconsistency | 0 primary |
| M9 | Emergent-* | unused |

## Visibility codes (the dangerous-class distinction)
Aligned with the three detection channels in the paper: Schema (automated),
Unaided (time-budgeted review), and Checklist (standards-aware review).

| Code | Label | Definition | Combined rate |
|------|-------|------------|---------------|
| V0 | Visible-automated (Schema) | Caught by an automated/schema/format check | 15% |
| V1 | Visible-review (Unaided) | Passes Schema but a reviewer catches it under normal time-budgeted review | 37% |
| V2 | **Silent semantic drift** | Semantically wrong, passes Schema AND is not caught under Unaided review; Checklist / ground truth may reveal it | **48%** |

Consistency rule used in Waves 1--2: primary M5 $\Leftrightarrow$ V0.

"Silent" (V2) does NOT mean undetectable under unlimited inspection; it is the
residue that survives the realistic review budget. V2 is the citable-lesson class:
it inflates false confidence in verifiability and normative strength (ASPICE / ISO
26262).

## Per-category result record (for observed categories only, T5)
For each category that becomes a finding, record ALL of:
1. definition; 2. illustrative anonymized example (case_id + quote span);
3. prevalence (count); 4. affected task families; 5. affected model/quant
conditions; 6. primary RE quality attribute threatened
(`verifiability | unambiguity | traceability | normative-strength`);
7. caught by the Schema gate? (yes/no); 8. caught by Unaided review? (yes/no).

See `analysis/out/pillar2_coding_summary.md` and coding sheets under `coding/samples/`.

## Reliability
Compute kappa on the double-coded subset with `analysis/kappa.py`.
- Wave~1 ($n{=}36$): primary $\kappa{=}1.00$, visibility $\kappa{=}0.91$; two V1/V2
  borderline disagreements in `wave1_disagreements.md`.
- Wave~2 ($n{=}36$): primary $\kappa{=}1.00$, visibility $\kappa{=}1.00$; zero
  disagreements (`wave2_disagreements.md`).

## Human-review experiment protocol (Unaided vs Checklist, T6) — PRE-REGISTER before review
All items are OPEN and must be fixed by the authors before any review begins:
- Reviewer background (RE/SE experience); number of reviewers (`\NumReviewers`).
- Sampling/stratification of reviewed cases; randomization of case order.
- Blinding: reviewers blinded to model/quant condition; ground truth hidden in both stages.
- Review time per case (`\ReviewTimeBudget`, default 60 s); documented rationale
  tying the budget to realistic industrial review pace.
- Written reviewer instructions; explicit detection criteria for "incorrect".
- Disagreement handling; inter-rater reliability (Cohen's kappa).
- Controls for review-order and fatigue effects (breaks, session caps,
  counterbalancing).
- Within-case comparison of Unaided vs Checklist review to estimate
  the reduction in silent-drift escape rate.
Review logs are tallied by `analysis/human_review.py`; outcomes are NEVER invented.
