# SEIP Submission-Readiness Package

Manuscript: *Not All Requirements Tasks Degrade Alike…* · Venue: ICSE 2027 SEIP  
Revised in-repo: `paper.tex`, `macros.tex`, `paper.pdf` (currently ~11 pp with refs)  
Companion: `AUTHOR_TODO_LIST.md`

---

## A. Reviewer-style diagnosis

### Critical
| Issue | Why a reviewer objects | Severity | Fix applied / remaining |
|---|---|---|---|
| Reasoning / cliff overclaim | Tasks differ in dataset, length, schema, metric | Critical | Rewrote claims to task-dependent degradation; composite construct; no cliff |
| Runtime confound FP16 vs GGUF | Malformed jump may be llama.cpp free decoding | Critical | Kept front-and-center in rivals + internal validity; grammar rerun is AUTHOR TODO (P1) |
| Pillar 3 handcrafted thresholds | Looks like post-hoc scoring | Critical | Relabeled as operational parameters; sensitivity grid reported (stable classes) |
| Ethics unspecified | Human coding / review / practitioners | Critical | Explicit subsection + AUTHOR TODO (do not invent determination) |
| Surface “gradual decline” | Paired FP16–Q4 CI includes 0 | Critical | Softened to usable region; mild PE change; CI includes 0 |

### Major
| Issue | Fix |
|---|---|
| Abstract as config dump | Rewritten: problem → design → quant → failure → review → deployment |
| Private-cloud rebuttal too long | Compressed to short subsection |
| Heterogeneous metrics compared | Within-family emphasis; Mid raw GGUF caveated |
| κ=1.00 oversold | Lead with visibility / C2–C3; taxonomy humility |
| Human-review circularity risk | Explicit V2 sampling vs independent escape rates |
| Standards “derived from ISO” | Attribute vocabulary vs author numeric params |
| Manuscript residue | STUB/OPEN/NO/concurrent removed from prose |

### Minor
Title OK (keep). Class-name aliases documented in table caption. Missing HF commits / DOI / CPU-RAM → AUTHOR TODO.

---

## B. Revision plan

| Pri | Status |
|-----|--------|
| **P0** Safer claims; Abstract/Intro; construct validity; Pillar3 params+sensitivity; ethics TODO; Surface CI wording; hygiene | **Done in manuscript** |
| **P1** Human-review clarity; taxonomy humility; threats consolidate; compress cloud | **Done** |
| **P2** Title variants; resource table; grammar rerun | Optional / AUTHOR |

---

## C. Section-by-section (summary of replacements)

- **Abstract:** Practitioner problem first; silent drift + escape rates; no cliff; deployment classes named consistently.
- **Intro:** Faster to gap; compressed private-cloud; pillars as decision pipeline; composite task-family construct.
- **RW:** Standards structure severity discussion; numeric params author-defined.
- **Design:** Operational task families; softened retention/cliff procedure; Pillar3 severity provenance; replication contents listed.
- **Pillar1:** Surface uncertainty (bootstrap + paired CI); supported claim = task-dependent profiles.
- **Pillar2:** M1–M4 non-finding as strength; M5 vs M6 practitioner lesson; anti-circular review protocol.
- **Pillar3:** Operational parameters; sensitivity subsection (real CSV); ethics subsection with TODOs; class names unified.
- **Discussion:** Interpretation, not number dump.
- **Threats:** Four consolidated paragraphs (construct / internal / conclusion / external).
- **Conclusion:** Scoped to 7–9B / GGUF / RTX 4090 / selected benchmarks.

---

## D. Revised manuscript

Primary files: `paper.tex` + `macros.tex`. PDF rebuild succeeds (~11 pages).

Computed (not invented) addenda used in prose:
- `replication/analysis/out/pillar1_bootstrap_ci.csv`
- `replication/analysis/out/decision_sensitivity.csv` (27/27: Surface-Q4=REVIEW, Reason/Mid-Q4=DO_NOT, Reason-FP16=ENHANCED)
- Surface paired FP16−Q4 mean Δ ≈ 0.044; bootstrap CI ≈ **[-0.006, 0.087]** (includes 0)

---

## E. AUTHOR TODO LIST

See root `AUTHOR_TODO_LIST.md`.

---

## F. Final consistency audit

| Check | Status |
|---|---|
| Abstract ↔ Results (Surface usable; Mid/Reason schema; V2 0.40; C2 0.67; C3 0.53; N=3) | Aligned |
| RQs ↔ Results (boundary / mechanisms / decisions) | Aligned |
| Results ↔ Discussion (no cliff; M5/M6; per-task policy) | Aligned |
| Table T2 macro means ↔ prose | Aligned (0.64/0.63/0.62/0.60 Surface; Mid † caveat) |
| Deployment class names | Macros use LOCAL_*; table uses short aliases with caption legend |
| Sample counts | 4 models × 4 prec × 3 tasks; coding n=100; review n=32; practitioners N=3 |
| M5/M6 | Macros `\MFiveCount{}` / `\MSixCount{}` |
| V0/V1/V2 | Table T6: 60 visible / 40 silent |
| C2/C3 | 0.67 / 0.53 means; κ 0.66 / 0.64 |
| FP16/GGUF terminology | FP16=transformers ref; GGUF=Q8/Q6/Q4 llama.cpp |
| Claims vs Threats | Scope qualifiers match conclusion |
| **Still visible in PDF until cleared** | Author block; ethics/consent TODOs; ASPICE clause TODO |
| **Not claimed** | Reasoning cliff; bit-width cliff; ASPICE numeric weights; industrial prevalence from N=32 or N=3 |

### Title
**Keep default.** Optional alternatives (at most three):
1. *Task-Dependent Localization of Requirements LLMs under Deployment Constraints*
2. *Which Requirements Tasks Survive Local LLMs? An SEIP Evaluation under Quantization*
3. *Silent Drift and Schema Failure: Local LLMs for Requirements Work under Constraints*
