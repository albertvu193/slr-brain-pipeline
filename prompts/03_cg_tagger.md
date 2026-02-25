# SLR CG Mechanism Tagging Prompt

You are tagging the specific Corporate Governance (CG) mechanisms that are **empirically tested** in an academic article.

## CG MECHANISM TAGS (select ALL that apply)

- **Board_Structure_Composition** — Board size, board independence ratio, proportion of independent/non-executive directors, board meetings frequency
- **Board_Diversity** — Gender diversity on the board, ethnic/racial diversity, age diversity, nationality diversity, educational background diversity
- **Board_Leadership** — CEO duality (CEO is also board chair), CEO power, chairman independence, CEO tenure
- **Board_Committee** — Audit committee (existence, size, independence, expertise), compensation/remuneration committee, nomination committee, CSR/sustainability committee, risk committee
- **Ownership_Structure** — Institutional ownership, managerial/insider ownership, family ownership, government/state ownership, foreign ownership, blockholders, type of controlling shareholder
- **Ownership_Concentration** — Degree of ownership concentration (top-1 / top-5 / top-10 shareholding), Herfindahl index of ownership, presence of controlling shareholders (without specifying type)
- **Mixed_CG_Mechanisms** — ONLY use when the study employs a **composite CG index or score** that bundles multiple mechanisms into one variable (e.g., "corporate governance quality index," "G-index," "governance score from a rating agency"). Do NOT use this tag just because the study tests multiple individual mechanisms.

## DECISION RULES

1. **Tag every mechanism the study empirically tests** as an independent variable, moderator, or mediator.
2. If the study tests 3+ specific mechanisms individually, tag EACH one separately — do NOT collapse them into Mixed_CG_Mechanisms.
3. Use **Mixed_CG_Mechanisms** ONLY for composite indices/scores, NOT for studies that test multiple individual mechanisms.
4. Do NOT tag mechanisms that are only mentioned in the literature review or used as control variables.
5. **Ownership_Structure vs Ownership_Concentration**: Use Ownership_Structure when the study distinguishes ownership by TYPE (institutional, family, etc.). Use Ownership_Concentration when the study measures the DEGREE of concentration regardless of type. If a study examines both type and degree, tag both.

## OUTPUT FORMAT
```json
{
  "cg_mechanisms": ["Board_Structure_Composition", "Board_Diversity"],
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence listing the specific CG variables tested."
}
```
