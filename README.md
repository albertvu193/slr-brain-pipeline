# SLR Brain — AI Screening Pipeline for Systematic Literature Reviews

A local Flask web app that classifies academic articles through a 5-step AI pipeline using OpenAI or Anthropic APIs. Built for SLR research on Corporate Governance (CG) → ESG outcomes.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8080
```

> **Mac users**: If port 8080 is busy, the app will tell you. Try `python -c "from app import app; app.run(port=9000)"`

## What It Does

Upload an Excel/CSV with article data (Title, Abstract columns required). The AI classifies each article through 5 chained steps:

| Step | File | Output | Description |
|------|------|--------|-------------|
| 1. Screen | `prompts/01_screener.md` | Include / Exclude / Background | Is this article relevant to CG-ESG research? |
| 2. Path | `prompts/02_path_classifier.md` | Path A / Both / Path B | What causal direction does it study? |
| 3. CG Tags | `prompts/03_cg_tagger.md` | Multi-tag | Which CG mechanisms are tested? |
| 4. ESG Tags | `prompts/04_esg_tagger.md` | Multi-tag | Which ESG outcomes are measured? |
| 5. Meta Score | `prompts/05_meta_scorer.md` | High / Medium / Low | How suitable for meta-analysis? |

If Step 1 = Exclude, Steps 2-5 are skipped (saves cost).

## Web UI (3 tabs)

### Tab 1: Run Pipeline
- Drag & drop Excel/CSV
- Choose provider (OpenAI or Anthropic) and model
- Enter API key
- Set max articles (use 5 for testing)
- Watch real-time progress with live feed

### Tab 2: Prompt Editor
- View/edit all 5 prompt templates in a sidebar layout
- Edit few-shot examples (JSONL format) for each step
- Click "Save Changes" — next run uses updated logic immediately
- No restart needed

### Tab 3: Results Table
- Full intermediate results for every article
- Columns: Screen status, Path, CG tags, ESG tags, Meta score, Confidence
- Click "show" on any row to expand raw API responses from all 5 steps
- Useful for debugging and spotting AI errors

## Self-Improving AI Loop

The `examples/` folder is your "fine-tuning without fine-tuning":

1. **Run a batch** → review Results Table
2. **Find errors** → note which articles the AI got wrong
3. **Add corrections** to `examples/*.jsonl` (via Prompt Editor tab or edit files directly)
4. **Re-run** → the corrected examples auto-inject as few-shot examples into API calls

### Example format (one JSON per line in `.jsonl`):

```json
{"title": "...", "abstract": "...", "expected": {"status": "Include", "exclusion_code": null, "confidence": 0.9, "reasoning": "..."}}
```

See `examples/README.md` for all formats.

## Project Structure

```
slr-brain-pipeline/
├── app.py                    # Flask app + AI pipeline + full HTML UI
├── prompts/                  # Editable prompt templates (Markdown)
│   ├── 01_screener.md        # Include/Exclude/Background logic
│   ├── 02_path_classifier.md # CG→ESG / Both / ESG→FP path
│   ├── 03_cg_tagger.md       # CG mechanism multi-tagging
│   ├── 04_esg_tagger.md      # ESG outcome multi-tagging
│   └── 05_meta_scorer.md     # Meta-analysis potential scoring
├── examples/                 # Few-shot examples (JSONL) — your fine-tuning layer
│   ├── screener.jsonl        # 5 gold-standard examples included
│   └── README.md             # How to add more examples
├── output/                   # Generated results (gitignored)
├── requirements.txt          # flask, pandas, openpyxl, openai, anthropic
├── .gitignore
└── README.md
```

## Architecture Notes

- **Single-file app**: Everything is in `app.py` (Python backend + HTML/CSS/JS frontend) for simplicity. No build tools needed.
- **Provider abstraction**: `call_openai()` and `call_anthropic()` share the same interface. Add new providers by adding a function + entry in `PROVIDERS` dict.
- **Prompt files are plain Markdown**: Edit in any text editor, or use the built-in Prompt Editor tab.
- **Examples auto-inject**: The `load_examples()` function reads JSONL files and appends them to the system prompt as few-shot examples.
- **Threading**: Each pipeline run spawns a thread. The UI polls `/status/<job_id>` every 1.2s for progress.
- **JSON mode**: OpenAI calls use `response_format={"type": "json_object"}` for reliable parsing.

## Cost Estimates

| Model | 590 articles | 5 articles (test) |
|-------|-------------|-------------------|
| gpt-4o-mini | ~$0.50 | ~$0.005 |
| gpt-4o | ~$5 | ~$0.05 |
| claude-sonnet | ~$3-5 | ~$0.03 |

## Taxonomy Reference

### Exclusion Codes
- `TA-E1`: Has CG but no ESG outcome
- `TA-E2`: Has ESG but no CG mechanism
- `TA-E5`: ESG→FP without CG involvement
- `TA-B1`: Review / meta-analysis / bibliometric

### CG Mechanisms
`Board_Structure_Composition`, `Board_Diversity`, `Board_Leadership`, `Board_Committee`, `Ownership_Structure`, `Ownership_Concentration`, `Mixed_CG_Mechanisms`

### ESG Outcomes
`ESG_Disclosure_Reporting`, `ESG_Performance_Rating`, `CSR_Reporting`, `E_Pillar`, `S_Pillar`, `G_Pillar`, `Integrated_Reporting`, `Sustainability_Assurance`

### Path Categories
- **Path_A_CG_to_ESG**: CG mechanisms → ESG outcomes
- **Both_A_and_B**: CG→ESG + ESG→FP (most common, ~62%)
- **Path_B_ESG_to_FP_with_CG_moderation**: ESG→FP with CG as moderator (rare, ~2%)
