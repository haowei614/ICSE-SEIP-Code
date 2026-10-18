# Dataset sources (locked 2026-09-05)

| Task | Local dir | Upstream | Notes |
|------|-----------|----------|-------|
| TF_Surface | `promise_nfr/` (625 rows) | [Zenodo 268542](https://zenodo.org/records/268542) PROMISE NFR | Cleland-Huang et al. |
| TF_Mid | `damir_pure/` (122 items) | [Figshare DAMIR-PURE](https://doi.org/10.6084/m9.figshare.29304650) | Anaphoric ambiguity on PURE subset (Ezzini et al. ICSE 2022). Rematerialized 2026-09-07: mojibake fix + pronoun char offsets (`word_index` / nearest match); `gold_label` ∈ {ambiguous, unambiguous} for stratified sampling. |
| TF_Reason | `trace_etour/` (57 UCs) | [tobhey/finegrained-traceability](https://github.com/tobhey/finegrained-traceability) eTour | Gold links: `etour_solution_links_english.txt` (orig. CoEST/TEFSE) |

Raw archives retained under `data/_downloads/` for provenance.
Harness-readable files only under the three dirs above.
