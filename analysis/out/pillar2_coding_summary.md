# Pillar 2 coding summary (Wave 1 + Wave 2)

Generated from `wave1_coding_sheet.csv` + `wave2_coding_sheet.csv` (coderA).
Do not invent; counts only.

## Sample

| Wave | n | Source | Primary κ / Visibility κ |
|------|---|--------|--------------------------|
| Wave 1 | 48 | FP16 failures | κ_primary=1.00, κ_vis=0.91 (n=36 subset) |
| Wave 2 | 52 | Quant Q8/Q6/Q4 failures | Wave 2 not double-coded yet |
| **Total** | **100** | stratified failures | |

Consistency checks (both waves): every primary M5 → V0; every V0 → primary M5. No contested codes in Wave 2.

## Primary mechanism

| Code | Wave 1 | Wave 2 | Combined |
|------|--------|--------|----------|
| M5 format/schema | 9 (19%) | 27 (52%) | 36 (36%) |
| M6 wrong content | 39 (81%) | 25 (48%) | 64 (64%) |
| M1–M4 primary | 0 | 0 | 0 |

Secondary: M6 co-occurs with M5 in 6 (W1) + 18 (W2) = 24 cases (format fails first; content also wrong).
M3 appears once as secondary (W2-032); not promoted.

## Visibility

| Code | Wave 1 | Wave 2 | Combined |
|------|--------|--------|----------|
| V0 (C1) | 9 (19%) | 27 (52%) | 36 (36%) |
| V1 (C2) | 15 (31%) | 9 (17%) | 24 (24%) |
| V2 silent | 24 (50%) | 16 (31%) | 40 (40%) |
| Visible (V0+V1) | 50% | 69% | **60%** |
| Silent (V2) | 50% | 31% | **40%** |

## Task family × visibility (combined n=100)

| Task | n | Visible | Silent (V2) | Dominant mechanism |
|------|---|---------|-------------|-------------------|
| TF-Surface | 24 | 9 (38%) | 15 (62%) | M6 only (FR/NFR boundary) |
| TF-Mid | 20 | 20 (100%) | 0 (0%) | M5 under quant; M6 false spans often V1 |
| TF-Reason | 56 | 31 (55%) | 25 (45%) | W1 mostly M6/V2; W2 mostly M5/V0 |

## Wave 2 reading (quant)

- Quant failures shift the **primary** channel from silent wrong answers (Wave 1) to **format collapse (M5)**, especially Reason/Mid under llama.cpp.
- Among **well-formed** Wave 2 failures (M6, n=25): V2=16 (64%), V1=9 (36%) — silent drift remains the danger class when parse succeeds.
- Surface under quant: still all M6; 9/12 coded V2 (FR↔NFR lexical cues).

## Implications for Pillar 3

- Surface: LOCAL_OK only with spot-check; silent FR/NFR risk → prefer LOCAL_WITH_REVIEW for high-severity.
- Reason/Mid under GGUF: schema gate mandatory; without it DO_NOT_LOCALIZE; even with parse, M6→V2 needs enhanced verification.
