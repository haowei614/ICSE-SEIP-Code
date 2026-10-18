# Severity Weighting — anchored in standards, not benchmark scores (Pillar 3)

Severity of a failure mechanism is derived from the requirements-quality attribute it
threatens, using attribute prominence in ASPICE and ISO 26262 rather than model
scores. Fill the weights from a documented reading of the standards text; do not
invent numbers.

| Quality attribute | Standard anchor (cite clause when filled) | Severity weight |
|-------------------|-------------------------------------------|-----------------|
| Verifiability | ASPICE SYS.2/SWE.1; ISO 26262 requirements attributes | [UNRESOLVED] |
| Unambiguity | ASPICE requirement attributes; ISO 26262-8 | [UNRESOLVED] |
| Traceability | ASPICE bidirectional traceability; ISO 26262 traceability | [UNRESOLVED] |
| Normative strength | ISO 26262 use of "shall"; ASPICE compliance | [UNRESOLVED] |

## From mechanism to deployment risk (T9)
Per-cell deployment risk uses a transparent, reproducible rule:

```
DeploymentRisk = SilentFailureProbability x RequirementsSeverity
```

- `SilentFailureProbability` = Pillar-2 silent-drift (V2) rate for the task
  family / condition of the cell.
- `RequirementsSeverity` = severity weight of the primarily-threatened RE quality
  attribute (table above), anchored in ASPICE / ISO 26262 text, NOT model scores.

Feasibility gates (from Pillar 1) are applied BEFORE risk banding:
- viable operating region exists for this task/HW (quality/retention meets the
  pre-registered bar)?
- configuration fits VRAM? latency within the envelope?
An infeasible cell can never be LOCAL_OK regardless of risk.

## Deployment classes (short names used in the paper)
| Class | Condition |
|-------|-----------|
| `LOCAL_OK` | feasible region + DeploymentRisk in LOW band |
| `LOCAL_WITH_REVIEW` | feasible region + MODERATE risk (mandatory human verification) |
| `LOCAL_WITH_ENHANCED_VERIFICATION` | feasible but HIGH risk from silent drift on high-severity attributes (standards-aware C3 review) |
| `DO_NOT_LOCALIZE / ESCALATE` | no viable region, or risk intolerable even with C3 |

## Pre-registration TODO
Risk-band cutoffs (LOW / MODERATE / HIGH) and the pre-registered operating-region
bar are **UNRESOLVED**. Fix them before generating the table; do not tune after
seeing results. Severity weights above are also UNRESOLVED pending a documented
reading of the standards clauses.
