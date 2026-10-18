# Decision table after practitioner T8 + chat-template alignment (v2)

Source: `t8_change_log.csv` · Sessions T8-S01..S03 · Participants P01–P03  
Plus 2026-09-16 alignment evidence (`CHAT_TEMPLATE_ALIGNMENT.md`).

Formative **N=3** face-validity for table revision; not a prevalence survey.

## Interpretation update (post-alignment)

T8 originally contrasted “GGUF vs FP16”. Aligned F16/Q* runs show that Mid/Reason
**format** failure under legacy GGUF was dominated by raw-prompt / missing HF chat
template, not bit-width. Practitioner judgments about **silent M6** and Mid
semantic uselessness remain valid; class labels are reframed as serving-config ×
verification rather than precision.

## Concrete revisions applied

| # | Initial cell | Change | Who drove | Final cell |
|---|--------------|--------|-----------|------------|
| 1 | Air-gap/NDA × TF-Surface → `LOCAL_WITH_REVIEW` | Upgrade: silent FR/NFR | P01, P03 | **`LOCAL_WITH_ENHANCED_VERIFICATION`** |
| 2 | Air-gap/NDA × TF-Reason → ENHANCED | Downgrade unsupervised (M6) | P03, P01 | **`DO_NOT_LOCALIZE` unsupervised**; suggest→human-commit OK |
| 3 | TF-Reason “GGUF” → `DO_NOT` | Reframe: **raw-prompt / misconfigured** GGUF | all + alignment | **`DO_NOT_LOCALIZE`** |
| 4 | TF-Mid × any → `DO_NOT` | Confirm (semantic; mal≈0 insufficient) | all | **`DO_NOT_LOCALIZE`** |
| 5 | Cost × Surface REVIEW | Split binding vs triage | P02 | Binding **REVIEW**; triage **LOCAL_OK**+5% audit |
| 6 | Cost × Reason | Aligned+schema+human vs misconfigured | P02 + alignment | Misconfigured **DO_NOT**; aligned+schema+human **ENHANCED** |
| 7 | Footnote | Schema removes M5 only | P01, P03 | Keep on Reason rows |

## Final summary table (RTX 4090 class)

| Constraint | Task | Class |
|------------|------|-------|
| Air-gap / NDA / residency | TF-Surface | `LOCAL_WITH_ENHANCED_VERIFICATION` |
| Air-gap / NDA / residency | TF-Reason | `DO_NOT_LOCALIZE` unsupervised; human-owned links OK as review workflow |
| Air-gap / NDA / residency | TF-Mid | `DO_NOT_LOCALIZE` |
| Cost-at-scale | TF-Surface (binding) | `LOCAL_WITH_REVIEW` |
| Cost-at-scale | TF-Surface (triage) | `LOCAL_OK` + ≥5% audit |
| Cost-at-scale | TF-Reason (raw-prompt / misconfigured GGUF) | `DO_NOT_LOCALIZE` |
| Cost-at-scale | TF-Reason (aligned chat template + schema + human) | `LOCAL_WITH_ENHANCED_VERIFICATION` |
| Cost-at-scale | TF-Mid | `DO_NOT_LOCALIZE` |
