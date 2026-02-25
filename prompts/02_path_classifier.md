# SLR Path Classification Prompt

You are classifying an academic article that has already been screened as "Include" in an SLR on Corporate Governance (CG) and ESG outcomes.

## PATH CATEGORIES

**Path_A_CG_to_ESG** — The study examines how CG mechanisms CAUSE or INFLUENCE ESG outcomes.
- Direction: CG → ESG
- Example: "Board independence improves ESG disclosure quality"
- Example: "Ownership concentration reduces sustainability reporting"

**Both_A_and_B** — The study examines BOTH directions:
- CG → ESG (Path A), AND
- ESG → Financial Performance with CG as moderator (Path B)
- Example: "Board diversity improves ESG scores, and ESG scores improve firm value when board independence is high"
- This is the MOST COMMON category (~62% of articles)

**Path_B_ESG_to_FP_with_CG_moderation** — The study ONLY examines ESG → Financial Performance, with CG as a moderating variable (not as a direct driver of ESG).
- Direction: ESG → FP, moderated by CG
- This is RARE (~2% of articles)

## DECISION LOGIC
1. Does the study test CG → ESG? If YES and nothing else → Path_A
2. Does the study ALSO test ESG → FP (or CG moderates ESG-FP link)? If YES → Both_A_and_B
3. Does the study ONLY test ESG → FP with CG as moderator? → Path_B

## OUTPUT FORMAT
```json
{
  "path": "Path_A_CG_to_ESG" | "Both_A_and_B" | "Path_B_ESG_to_FP_with_CG_moderation",
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence"
}
```
