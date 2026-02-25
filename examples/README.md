# Examples — Your AI Gets Smarter Here

Each `.jsonl` file in this folder contains gold-standard examples that get automatically injected into the AI prompt as few-shot examples.

## How to add examples

1. After a Brain run, review the output Excel
2. Find articles where the AI got it WRONG
3. Add the corrected version here as a new JSON line

### Format for each step:

**screener.jsonl**
```json
{"title": "...", "abstract": "...", "expected": {"status": "Include|Exclude|Background", "exclusion_code": null|"TA-E1"|"TA-E2"|"TA-E5"|"TA-B1", "confidence": 0.9, "reasoning": "..."}}
```

**path.jsonl**
```json
{"title": "...", "abstract": "...", "expected": {"path": "Path_A_CG_to_ESG|Both_A_and_B|Path_B_ESG_to_FP_with_CG_moderation", "confidence": 0.9, "reasoning": "..."}}
```

**cg.jsonl**
```json
{"title": "...", "abstract": "...", "expected": {"cg_mechanisms": ["Board_Structure_Composition", "Board_Diversity"], "confidence": 0.9, "reasoning": "..."}}
```

**esg.jsonl**
```json
{"title": "...", "abstract": "...", "expected": {"esg_outcomes": ["ESG_Disclosure_Reporting", "ESG_Performance_Rating"], "confidence": 0.9, "reasoning": "..."}}
```

**meta.jsonl**
```json
{"title": "...", "abstract": "...", "expected": {"meta_potential": "High|Medium|Low", "confidence": 0.9, "reasoning": "..."}}
```

## The improvement loop

More examples = better AI. Aim for:
- 10-15 examples per step to start
- Focus on HARD cases (the ones the AI gets wrong)
- Include edge cases and boundary decisions
