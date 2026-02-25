# SLR ESG Outcome Tagging Prompt

You are tagging the ESG outcome variables that are **empirically measured** as dependent variables or mediators in an academic article.

## ESG OUTCOME TAGS (select ALL that apply)

- **ESG_Disclosure_Reporting** — ESG disclosure quality/quantity, sustainability reporting level, GRI reporting, environmental/social disclosure indices, voluntary disclosure measures, disclosure scores constructed by researchers from annual/sustainability reports
- **ESG_Performance_Rating** — ESG scores or ratings from **third-party agencies**: MSCI ESG, Sustainalytics, Bloomberg ESG, Thomson Reuters/Refinitiv ESG, FTSE4Good, KLD/MSCI ratings, Asset4, S&P Global ESG scores. The key distinction: a professional agency produced the score, not the researchers.
- **CSR_Reporting** — CSR-specific reporting or CSR performance measures explicitly labeled as "CSR" rather than "ESG" or "sustainability." Use this when the paper specifically frames outcomes in CSR terminology.
- **E_Pillar** — Environmental outcomes analyzed **separately**: carbon emissions/intensity, environmental compliance, green innovation, environmental scores, carbon disclosure, environmental performance
- **S_Pillar** — Social outcomes analyzed **separately**: employee welfare, community investment, human rights, social scores, labor practices, health & safety
- **G_Pillar** — Governance pillar as an **outcome** variable (not input): governance quality scores from ESG raters when analyzed as a dependent variable, governance disclosure
- **Integrated_Reporting** — Integrated reporting quality/adoption under the <IR> framework, combined financial and non-financial reporting specifically in the integrated reporting context
- **Sustainability_Assurance** — External assurance or verification of sustainability reports, third-party ESG auditing, assurance quality

## DECISION RULES

1. Tag every ESG outcome the study measures as a dependent variable or mediator.
2. **Choose precisely** — do NOT automatically tag ESG_Disclosure_Reporting. Only tag it when the study actually measures disclosure/reporting as an outcome.
3. **ESG_Disclosure_Reporting vs ESG_Performance_Rating**: If the study uses agency ratings (Bloomberg, Refinitiv, MSCI, etc.), tag ESG_Performance_Rating. If the study constructs its own disclosure index from reports, tag ESG_Disclosure_Reporting. If both, tag both.
4. **CSR_Reporting vs ESG_Disclosure_Reporting**: If the paper explicitly uses "CSR" terminology, tag CSR_Reporting. If it uses "ESG" or "sustainability" terminology, tag ESG_Disclosure_Reporting. If it uses both, tag both.
5. **Individual pillars**: Only tag E_Pillar, S_Pillar, or G_Pillar when the study **separately analyzes** these pillars. If the study only uses an aggregate ESG score, do NOT tag individual pillars.
6. Do NOT tag outcomes that are only mentioned in the literature review or hypotheses but not empirically tested.

## OUTPUT FORMAT
```json
{
  "esg_outcomes": ["ESG_Disclosure_Reporting", "ESG_Performance_Rating"],
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence identifying the specific ESG outcomes measured."
}
```
