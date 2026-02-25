# SLR Meta-Analysis Potential Scoring Prompt

You are scoring how suitable an academic article is for inclusion in a future meta-analysis on the CG-ESG relationship.

## META-ANALYSIS POTENTIAL LEVELS

**High** — The study provides:
- Clear quantitative effect sizes (correlation coefficients, regression coefficients with standard errors)
- Large sample sizes (typically 500+ firm-year observations)
- Well-defined, replicable measures of both CG and ESG variables
- Standard econometric methods (OLS, fixed effects, GMM)
- Studies covering major economies or large cross-country samples

**Medium** — The study provides:
- Some quantitative results but effect sizes may need extraction/conversion
- Moderate sample sizes (100-500 firm-year observations)
- Reasonably clear variable definitions but may need interpretation
- Standard methods but with some complexity in extraction

**Low** — The study has:
- Qualitative or mixed-methods design making effect size extraction difficult
- Very small samples (<100 observations)
- Highly context-specific or niche populations
- Non-standard measures that are hard to compare across studies
- Case studies, interviews, or survey-based designs without comparable metrics

## OUTPUT FORMAT
```json
{
  "meta_potential": "High" | "Medium" | "Low",
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence"
}
```
