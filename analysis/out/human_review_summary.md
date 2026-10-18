# Human review (C2/C3) summary

**Locked protocol:** `coding/HUMAN_REVIEW_PROTOCOL.md`  
**Logs:** `coding/human_review_sheet.csv` (+ blinded copies per reviewer)  
**Date tallied:** 2026-09-07

## Design realization
- n_cases = 32 (24 coderA-V2 + 8 coderA-V1 controls); seed=20260907
- Reviewers: **R1-SE**, **R2-RE** (2 independent; each scored all 32)
- All cases: `semantically_incorrect=1`, `c1_flagged=0` (sample excludes V0/M5)
- Mean `review_seconds` ≈ 57.7 (median 58; C2 budget 60s + C3 up to 90s)

## Escape rates (eligible = wrong ∧ ¬C1; here = all 32)

| Reviewer | C2 escape | C3 escape | Δ (C2−C3) |
|----------|-----------|-----------|-----------|
| R1-SE | 19/32 = **0.594** | 14/32 = **0.438** | 0.156 |
| R2-RE | 24/32 = **0.750** | 20/32 = **0.625** | 0.125 |
| **Mean** | **0.672** | **0.531** | **0.141** |

Among coderA-**V2** only (n=24):

| Reviewer | C2 escape | C3 escape |
|----------|-----------|-----------|
| R1-SE | 19/24 = 0.792 | 14/24 = 0.583 |
| R2-RE | 24/24 = 1.000 | 20/24 = 0.833 |
| Mean | **0.896** | **0.708** |

Among coderA-**V1** controls (n=8): **C2 escape = 0** for both reviewers (all caught in normal review).

## Inter-rater agreement (flags)
| Flag | Raw agree | Cohen's κ |
|------|-----------|-----------|
| c2_flagged | 27/32 | **0.655** (substantial) |
| c3_flagged | 26/32 | **0.636** (substantial) |

Disagreements listed in `coding/human_review_disagreements.md`.

## Interpretation (for paper)
1. CoderA V1 labels validated: controls never escape C2.
2. Silent-drift residue is real under 60s review: on V2 items, mean C2 escape ≈ **90%**.
3. Standards-aware C3 **reduces but does not eliminate** escape (mean absolute drop ≈ 0.14 on full set; ≈ 0.19 on V2-only).
4. Do **not** claim C3 is sufficient alone; supports `LOCAL_WITH_ENHANCED_VERIFICATION` not `LOCAL_OK`.
