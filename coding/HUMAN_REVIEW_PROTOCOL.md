# Human-review protocol (Unaided vs Checklist) — PRE-REGISTERED

**Locked:** 2026-09-07 · **Before** any review session begins.  
Do not change parameters after the first reviewed case without logging a protocol amendment.

## Goal
Estimate whether a lightweight standards-aware Checklist reduces silent-drift
escape relative to time-budgeted Unaided review, on cases that already pass
the automated Schema gate.

## Sample
- Sheet: `coding/human_review_sheet.csv` (n=32, seed=20260907)
- Composition: 24 coderA-V2 (silent candidates) + 8 coderA-V1 (visible-review controls)
- Stratification target among V2: TF-Surface 12 + TF-Reason 12 (TF-Mid had 0 V2 in n=100)
- Packets: `coding/samples/wave{1,2}_packets.jsonl` by `case_id`
- **Blinding:** reviewers must **not** see `coderA_visibility` / `coderA_primary` columns
  during both stages (hide or strip those columns in the working copy). Ground truth / gold
  is hidden throughout; only the standards checklist is added in the second stage.

## Reviewers
| Parameter | Pre-registered value | Status |
|-----------|----------------------|--------|
| N reviewers | 2 (independent) | OPEN name/role |
| Background | RE or SE; ≥1 year requirements/docs experience preferred | OPEN |
| Blind to model/quant | yes | locked |
| Blind to coderA labels | yes | locked |
| Consent / ethics | record IRB or exemption determination before sessions | OPEN |

## Timing and fatigue
| Parameter | Value |
|-----------|-------|
| Time budget per case (Unaided) | **60 s** (`\ReviewTimeBudget`) |
| Rationale | approximates a quick industrial spot-check of one model output without hunting gold |
| Checklist extra budget | +30 s (total 90 s) for checklist only—not unlimited |
| Session cap | ≤16 cases / session; ≥10 min break between sessions |
| Order | use `order_index` in sheet; counterbalance reviewer×half-session if two reviewers |

## Detection criteria
**Unaided review:** Mark `c2_flagged=1` if the reviewer would **reject or escalate** the
output under ordinary review (looks wrong, incomplete, or untrustworthy) **without**
gold and **without** the standards checklist.

**Checklist review:** After the Unaided decision is recorded, apply the checklist and set
`c3_flagged`. Checklist probes (yes/no; any "fail" → flag):
1. **Verifiability** — Is the claim checkable / testable as written?
2. **Unambiguity** — Could two engineers reasonably disagree on the referent?
3. **Traceability** — Are named artifacts/links specific enough to follow?
4. **Normative strength** — If obligations appear, is shall/should/may coherent?

Do **not** open gold during either stage. After both flags are set, a separate adjudicator
may confirm `semantically_incorrect` from gold (pre-filled as 1 for this failure sample;
leave as-is unless a case was mis-sampled).

## Schema gate
`c1_flagged` is pre-filled from schema/malformed status (V0→1). Reviewers do not change it.
The historical column names `c1_flagged`, `c2_flagged`, and `c3_flagged` are retained
for file compatibility; the paper calls them Schema, Unaided, and Checklist.

## Logging
Fill in `human_review_sheet.csv`:
`reviewer, c2_flagged, c3_flagged, review_seconds, notes`.

Tally (never invent):
```bash
cd replication
python -m analysis.human_review coding/human_review_sheet.csv --out analysis/out
```

## Outcomes to report
- Silent-drift escape under Unaided review: wrong ∧ passes Schema ∧ not flagged Unaided
- Escape under Checklist review: wrong ∧ passes Schema ∧ not flagged Checklist
- Difference (Unaided escape − Checklist escape) with n of eligible cases
- Inter-rater κ on `c2_flagged` / `c3_flagged` if two reviewers overlap ≥15 cases

## Amendment log
| Date | Change | Reason |
|------|--------|--------|
| — | — | — |
