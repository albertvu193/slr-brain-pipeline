# SLR Meta-Analysis Potential Scoring Prompt

You are scoring how suitable an academic article is for inclusion in a future meta-analysis on the CG-ESG relationship. Base your assessment on what can be inferred from the title, abstract, and keywords.

## META-ANALYSIS POTENTIAL LEVELS

**High** — Strong indicators of meta-analysis suitability:
- Abstract mentions **regression analysis**, fixed effects, GMM, or panel data methods
- Abstract mentions a **large sample** or "firm-year observations" (likely 500+), or references a major database (Compustat, WRDS, Thomson Reuters, Refinitiv, Bloomberg, CSMAR)
- The study appears to use **standard, well-defined measures** of CG and ESG variables
- Major economies or large multi-country samples (e.g., S&P 500, FTSE, EU-wide, G20)
- Published in a recognized finance, accounting, or management journal

**Medium** — Moderate indicators:
- Abstract mentions quantitative analysis but methodology details are sparse
- Sample description suggests a moderate scope (single-country developing market, specific industry sector)
- Variables seem standard but may require interpretation for effect-size extraction
- Conference papers or working papers with quantitative methods

**Low** — Weak indicators for meta-analysis:
- Abstract mentions **qualitative methods**: case studies, interviews, content analysis (manual), surveys, questionnaires, grounded theory
- Very small or niche samples: single firm, single industry in a small market, very specific context
- Non-standard or proprietary measures that would be hard to compare across studies
- Book chapters, conceptual papers, or studies where the methodology is unclear from the abstract

## INFERENCE GUIDELINES
Abstracts rarely state exact sample sizes or report standard errors. Use these signals:
- "Panel data of listed firms on [major stock exchange]" → likely High
- "Survey of 200 managers" → likely Low (survey-based)
- "Content analysis of annual reports from 50 firms" → likely Medium
- "Emerging market" or "developing economy" alone does NOT lower the score — large emerging-market studies with panel data can still be High
- If you truly cannot determine methodology from the abstract, assign Medium with confidence 0.50-0.60

## OUTPUT FORMAT
```json
{
  "meta_potential": "High" | "Medium" | "Low",
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence citing the specific methodology/sample signals observed."
}
```
