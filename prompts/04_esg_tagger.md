# SLR ESG Outcome Tagging Prompt

You are tagging the ESG outcome variables studied in an academic article.

## ESG OUTCOME TAGS (select ALL that apply)

- **ESG_Disclosure_Reporting** — ESG disclosure quality/quantity, sustainability reporting, GRI reporting, environmental/social disclosure indices. This is the MOST COMMON tag and applies to most articles.
- **ESG_Performance_Rating** — ESG scores from rating agencies (MSCI, Sustainalytics, Bloomberg, Thomson Reuters/Refinitiv, FTSE4Good), composite ESG performance scores
- **CSR_Reporting** — CSR-specific reporting, CSR disclosure, CSR performance (when specifically labeled CSR rather than ESG)
- **E_Pillar** — Environmental performance specifically: carbon emissions, environmental compliance, green innovation, environmental scores
- **S_Pillar** — Social performance specifically: employee welfare, community investment, human rights, social scores
- **G_Pillar** — Governance pillar performance specifically (as an outcome, not as an input mechanism): governance quality scores from ESG raters
- **Integrated_Reporting** — Integrated reporting (<IR> framework), combined financial-nonfinancial reporting
- **Sustainability_Assurance** — External assurance/verification of sustainability reports, third-party ESG auditing

## DECISION RULES
1. Tag EVERY ESG outcome the study measures as a dependent variable or mediator
2. ESG_Disclosure_Reporting is the default — nearly all studies involve some form of disclosure
3. If the study uses ESG scores from a rating agency → also tag ESG_Performance_Rating
4. If the study separately analyzes E, S, or G pillars → tag the specific pillars
5. Use semicolons to separate multiple tags

## OUTPUT FORMAT
```json
{
  "esg_outcomes": ["ESG_Disclosure_Reporting", "ESG_Performance_Rating"],
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence"
}
```
