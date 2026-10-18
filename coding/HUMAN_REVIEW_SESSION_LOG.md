# Human review session log (RE/SE reviewers)

**Date:** 2026-09-07  
**Protocol:** `HUMAN_REVIEW_PROTOCOL.md` (locked)  
**Sheet:** `human_review_sheet_blinded.csv` (n=32 × 2 reviewers = 64 rows)

## Reviewers
| ID | Role | Blind to gold / coderA / model·quant |
|----|------------------|--------------------------------------|
| R1-SE | SE / QA-leaning (≥1y reqs review) | yes |
| R2-RE | RE analyst (industrial skim) | yes |

## Timing
- Unaided budget 60s; Checklist +30s; session cap respected via order_index 1–16 / 17–32 break note in protocol.
- `review_seconds` = total Unaided+Checklist wall time recorded per case.

## Inter-rater (n=32 overlap)
- Unaided: raw agree 27/32; Cohen κ ≈ 0.66
- Checklist: raw agree 26/32; Cohen κ ≈ 0.64

## Escape (wrong and passes Schema; all 32 eligible)
| Reviewer | Unaided escape | Checklist escape | Δ |
|----------|-----------|-----------|-----------|
| R1-SE | 19/32 | 14/32 | 5 |
| R2-RE | 24/32 | 20/32 | 4 |

Checklist review reduces escape for both reviewers; residual silent cases remain (name-matched Reason links; FR/NFR boundary Surface labels).

The 32-case set contains 17 Wave~1 and 15 legacy raw-prompt Wave~2 cases.
The paper therefore treats W1-only escape as primary and the full set as a sensitivity analysis.

## Artifacts
- `human_review_sheet_blinded.csv` — combined 64 rows
- `human_review_sheet_blinded_R1-SE.csv` / `_R2-RE.csv` — per reviewer
- `human_review_sheet.csv` — same flags + model/quant/coderA metadata for analysis
