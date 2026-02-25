# SLR Path Classification Prompt

You are classifying the causal pathway studied in an academic article that was screened as "Include" in an SLR on Corporate Governance (CG) and ESG outcomes.

## PATH CATEGORIES

**Path_A_CG_to_ESG** — The study examines how CG mechanisms **directly cause or influence** ESG outcomes. CG is the independent variable; ESG is the dependent variable. No financial performance outcome is tested.
- Direction: CG → ESG (direct effect)
- Example: "Board independence improves ESG disclosure quality"
- Example: "Ownership concentration reduces voluntary sustainability reporting"
- Example: "The effect of audit committee characteristics on ESG reporting"

**Both_A_and_B** — The study examines **both** directions in a single paper:
- CG → ESG (Path A), **AND**
- ESG → Financial Performance, with CG as moderator of that link (Path B)
- OR: CG → ESG → FP as a **mediation model** (CG drives ESG, which then drives FP)
- Example: "Board diversity improves ESG scores (Path A), and ESG scores improve firm value when board independence is high (Path B)"
- Example: "Board independence → ESG disclosure → Firm value (full mediation)"

**Path_B_ESG_to_FP_with_CG_moderation** — The study **ONLY** examines ESG → Financial Performance, with CG appearing as a moderating variable. CG does NOT directly drive ESG in this study. CG only moderates the ESG-FP link.
- Direction: ESG → FP, moderated by CG
- Example: "ESG performance improves stock returns, but only for firms with independent boards"
- This path is uncommon and should only be assigned when CG is PURELY a moderator with no direct CG→ESG analysis.

## DECISION TREE

Follow these steps IN ORDER:

1. **Does the study test CG → ESG as a direct relationship?**
   - YES → Continue to step 2
   - NO → Go to step 3

2. **Does the study ALSO test ESG → FP, OR model CG → ESG → FP as mediation?**
   - YES → **Both_A_and_B**
   - NO → **Path_A_CG_to_ESG**

3. **Does the study ONLY test ESG → FP with CG as moderator?**
   - YES → **Path_B_ESG_to_FP_with_CG_moderation**
   - This should be rare — double-check that there is truly no CG → ESG analysis.

## EDGE CASES
- If the study tests CG → ESG AND CG → FP (direct, not through ESG), classify as **Path_A_CG_to_ESG** — the FP link is a secondary analysis, not the ESG→FP pathway.
- If unsure whether FP is involved, default to **Path_A_CG_to_ESG** — it is the safer classification.

## OUTPUT FORMAT
```json
{
  "path": "Path_A_CG_to_ESG" | "Both_A_and_B" | "Path_B_ESG_to_FP_with_CG_moderation",
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence identifying the specific causal direction(s) tested."
}
```
