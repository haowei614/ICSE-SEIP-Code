# Practitioner feedback sessions (T8) — session guide

**Status:** complete · formative **N=3** locked · not a survey

## Files
| File | Use |
|------|-----|
| `RECRUITMENT_MESSAGE_zh.md` | **复制发给候选人** |
| `PARTICIPANT_ONEPAGER_zh.md` | 会前可选一页纸 |
| `exhibits_for_practitioners.md` | 会中展示的 4 个案例 |
| `generated_table_v0.md` | 临时决策表 |
| `t8_change_log.csv` | **每场记录改表**（投后填论文 T8） |

## Before sessions
- [ ] Ethics / IRB or exemption determination recorded
- [ ] Consent (anonymized notes OK; right to withdraw)
- [ ] Open exhibits + table; blank `t8_change_log.csv` row ready

## Protocol (30–45 min)
1. Context (5 min): local LLM under NDA/air-gap; ask their constraint type.
2. Show **3 silent-drift examples** (V2): W1-001, L2-019, L2-016 (legacy Wave~2 IDs; aligned Wave~2 keeps W2-*). Ask: would normal ~60s review catch this?
3. Show **1 M5/format** example: L2-005. Ask: do you already have schema gates?
4. Walk provisional table (`generated_table_v0.md`). Ask which cells are wrong.
5. Record **every concrete cell change** in `t8_change_log.csv` (or explicit “no change”).

## After all sessions
Tell the agent/repo owner: merge `t8_change_log.csv` → update `paper.tex` Table T8 + `\TableChangesFromFeedback`.
