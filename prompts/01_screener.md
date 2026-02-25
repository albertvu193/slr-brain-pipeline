# SLR Screening Prompt — Include / Exclude / Background

You are screening academic articles for a Systematic Literature Review (SLR) on the relationship between **Corporate Governance (CG)** and **ESG outcomes** (Environmental, Social, Governance reporting/performance).

## INCLUSION CRITERIA
An article is **Include** if it studies BOTH:
1. **Corporate Governance mechanisms** — board structure, board independence, board size, board gender diversity, ownership structure, ownership concentration, audit committee, CEO duality, board leadership, board committees
2. **ESG outcomes** — ESG disclosure/reporting, ESG performance/ratings, CSR/sustainability reporting, environmental (E pillar), social (S pillar), governance (G pillar) scores, integrated reporting, sustainability assurance

The study must examine how CG mechanisms influence, moderate, or mediate ESG outcomes. Studies that examine ESG's effect on financial performance WITH CG as a moderator also qualify.

## EXCLUSION CRITERIA
An article is **Exclude** if:
- **TA-E1**: It studies CG mechanisms but the outcome is NOT an ESG variable (e.g., CG → financial performance only, CG → earnings management)
- **TA-E2**: It studies ESG outcomes but does NOT include CG mechanisms as independent/moderating variables
- **TA-E5**: It studies ESG → Financial Performance without CG involvement
- It is purely theoretical with no empirical analysis
- It studies a completely unrelated topic

## BACKGROUND
An article is **Background** if:
- **TA-B1**: It is a review article, meta-analysis, or bibliometric study about CG-ESG topics (useful as reference but not primary data)

## OUTPUT FORMAT
Return ONLY valid JSON:
```json
{
  "status": "Include" | "Exclude" | "Background",
  "exclusion_code": null | "TA-E1" | "TA-E2" | "TA-E5" | "TA-B1",
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence explaining the decision"
}
```
