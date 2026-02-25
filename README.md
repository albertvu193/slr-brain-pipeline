# SLR Brain — AI Screening Pipeline for Systematic Literature Reviews

A local Flask web app that classifies academic articles through a 6-step AI pipeline using OpenAI or Anthropic APIs. Built for SLR research on Corporate Governance (CG) → ESG outcomes.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:8080
```

> **Mac users**: If port 8080 is busy, the app will tell you. Try `python -c "from app import app; app.run(port=9000)"`

## Pipeline Steps

Upload an Excel/CSV with article data (Title, Abstract columns required). The pipeline runs through these steps:

| Step | What it does | Output |
|------|-------------|--------|
| **0. Upload** | Parse and validate your file | Article count, column detection |
| **1. Deduplicate** | Remove duplicate articles by DOI and normalized title | Unique article count, duplicate report |
| **2. Screen** | AI: Is this article relevant to CG-ESG research? | Include / Exclude / Background + exclusion codes |
| **3. Path** | AI: What causal direction does it study? | Path A (CG→ESG) / Both A+B / Path B (ESG→FP) |
| **4. CG Tags** | AI: Which CG mechanisms are empirically tested? | Multi-tag: Board, Ownership, etc. |
| **5. ESG Tags** | AI: Which ESG outcomes are measured? | Multi-tag: Disclosure, Rating, Pillars |
| **6. Meta Score** | AI: How suitable for meta-analysis? | High / Medium / Low |

If Step 2 = Exclude or Background, Steps 3-6 are skipped (saves cost).

## Web UI (3 tabs)

### Tab 1: Run Pipeline
- **Pipeline tracker** at the top shows each step with live progress counts
- Drag & drop Excel/CSV
- Choose provider (OpenAI or Anthropic) and model
- Enter API key, set max articles (use 5 for testing)
- **Deduplication banner** shows how many duplicates were removed
- **Live feed** shows each article being processed with current step indicator
- **Stats bar** tracks Include/Exclude/Background counts in real-time

### Tab 2: Prompt Editor
- View/edit all 5 prompt templates in a sidebar layout
- Edit few-shot examples (JSONL format) for each step
- Click "Save Changes" — next run uses updated logic immediately
- No restart needed

### Tab 3: Results Table
- **Summary cards** showing totals: screened, duplicates removed, include/exclude/background, path breakdown
- **Filter tabs** to show All / Include only / Exclude only / Background only
- Full results with columns: Screen status, Exclusion Code, Path, CG tags, ESG tags, Meta score, Confidence
- Click "show" on any row to expand raw API responses from all 5 steps

## Evaluation System

Test your prompts against a gold-standard dataset before running on real data:

```bash
python eval.py --provider openai --api-key sk-... --test-file eval_testset.jsonl
```

The eval script:
- Runs each test article through the full pipeline
- Compares AI output vs expected answers per step
- Reports accuracy for classification steps (Screen, Path, Meta)
- Reports precision/recall/F1 for tagging steps (CG Tags, ESG Tags)
- Prints confusion matrices for misclassifications
- Saves detailed results to `output/eval_results.json`

Options: `--model gpt-4o` to test with a specific model, `--quiet` to suppress per-article output.

### Building your test set

Edit `eval_testset.jsonl` (15 cases included). Each line:
```json
{"title": "...", "abstract": "...", "expected": {"status": "Include", "path": "Path_A_CG_to_ESG", "cg_mechanisms": ["Board_Structure_Composition"], "esg_outcomes": ["ESG_Disclosure_Reporting"], "meta_potential": "High"}}
```

Only include the `expected` fields you want to evaluate — omitted fields are skipped.

## Self-Improving AI Loop

The `examples/` folder is your "fine-tuning without fine-tuning":

1. **Run a batch** → review Results Table
2. **Find errors** → note which articles the AI got wrong
3. **Add corrections** to `examples/*.jsonl` (via Prompt Editor tab or edit files directly)
4. **Re-run** → the corrected examples auto-inject as few-shot examples into API calls
5. **Run eval** → `python eval.py ...` to verify improvements

### Example files included

| File | Examples | Coverage |
|------|----------|----------|
| `examples/screener.jsonl` | 12 | Include, Exclude (TA-E1/E2/E5), Background, edge cases (CG as control, ambiguous ESG) |
| `examples/path.jsonl` | 7 | Path A, Both A+B, Path B, mediation models |
| `examples/cg.jsonl` | 8 | All CG tags including Mixed, Ownership_Structure vs Concentration |
| `examples/esg.jsonl` | 8 | All ESG tags, Disclosure vs Rating distinction |
| `examples/meta.jsonl` | 7 | High/Medium/Low with methodology signal examples |

## Project Structure

```
slr-brain-pipeline/
├── app.py                    # Flask app + AI pipeline + full HTML UI
├── eval.py                   # Evaluation script — test prompts against gold-standard data
├── eval_testset.jsonl        # 15 gold-standard test cases
├── prompts/                  # Editable prompt templates (Markdown)
│   ├── 01_screener.md        # Include/Exclude/Background logic
│   ├── 02_path_classifier.md # CG→ESG / Both / ESG→FP path
│   ├── 03_cg_tagger.md       # CG mechanism multi-tagging
│   ├── 04_esg_tagger.md      # ESG outcome multi-tagging
│   └── 05_meta_scorer.md     # Meta-analysis potential scoring
├── examples/                 # Few-shot examples (JSONL) — your fine-tuning layer
│   ├── screener.jsonl        # 12 gold-standard examples
│   ├── path.jsonl            # 7 examples
│   ├── cg.jsonl              # 8 examples
│   ├── esg.jsonl             # 8 examples
│   ├── meta.jsonl            # 7 examples
│   └── README.md             # How to add more examples
├── output/                   # Generated results (gitignored)
├── requirements.txt          # flask, pandas, openpyxl, openai, anthropic
├── .gitignore
└── README.md
```

## Architecture Notes

- **Single-file app**: Everything is in `app.py` (Python backend + HTML/CSS/JS frontend) for simplicity. No build tools needed.
- **Provider abstraction**: `call_openai()` and `call_anthropic()` share the same interface. Add new providers by adding a function + entry in `PROVIDERS` dict.
- **Robust JSON parsing**: `extract_json()` handles Anthropic responses that may wrap JSON in markdown code blocks.
- **Retry with backoff**: API calls retry up to 3 times with exponential backoff on transient failures.
- **Deduplication**: Automatic DOI + normalized title dedup before AI screening begins.
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
- **Path_A_CG_to_ESG**: CG mechanisms → ESG outcomes (direct effect only)
- **Both_A_and_B**: CG→ESG + ESG→FP or CG→ESG→FP mediation
- **Path_B_ESG_to_FP_with_CG_moderation**: ESG→FP with CG as moderator only (rare)
