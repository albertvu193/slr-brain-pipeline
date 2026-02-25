# SLR CG Mechanism Tagging Prompt

You are tagging the Corporate Governance (CG) mechanisms studied in an academic article.

## CG MECHANISM TAGS (select ALL that apply)

- **Board_Structure_Composition** — Board size, board independence ratio, proportion of independent/non-executive directors
- **Board_Diversity** — Gender diversity on board, ethnic/racial diversity, age diversity, nationality diversity
- **Board_Leadership** — CEO duality (CEO is also board chair), CEO power, chairman independence
- **Board_Committee** — Audit committee, compensation committee, nomination committee, CSR/sustainability committee
- **Ownership_Structure** — Institutional ownership, managerial ownership, family ownership, government ownership, foreign ownership, block holders
- **Ownership_Concentration** — Concentration of ownership (e.g., top shareholders, Herfindahl index), controlling shareholders
- **Mixed_CG_Mechanisms** — The study uses a broad CG index/score combining multiple mechanisms, or studies 4+ mechanisms without focusing on any single one

## DECISION RULES
1. Tag EVERY specific mechanism the study empirically tests (as IV, moderator, or mediator)
2. If the study tests a composite CG index/score → tag as Mixed_CG_Mechanisms
3. If the study tests 4+ individual mechanisms broadly → tag as Mixed_CG_Mechanisms
4. Use semicolons to separate multiple tags
5. Only tag mechanisms that are empirically tested, not just mentioned in the literature review

## OUTPUT FORMAT
```json
{
  "cg_mechanisms": ["Board_Structure_Composition", "Board_Diversity"],
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence"
}
```
