"""
SLR Brain v3 — Web UI with Pipeline Visualization + Prompt Editor + Eval + Multi-Provider
"""

import os, json, time, uuid, threading, re
import urllib.request, urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_file
import pandas as pd

# --- AI Provider Abstraction ---
def call_openai(prompt, system_msg, model, api_key, temperature=0.2):
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    # Reasoning models (o1/o3/o4) don't support temperature or system role
    is_reasoning = model.startswith(('o1', 'o3', 'o4'))
    if is_reasoning:
        messages = [{"role": "user", "content": system_msg + "\n\n---\n\n" + prompt}]
    else:
        messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": prompt}]
    kwargs = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
    if not is_reasoning:
        kwargs["temperature"] = temperature
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content

def call_anthropic(prompt, system_msg, model, api_key, temperature=0.2):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model, max_tokens=4096, temperature=temperature,
        system=system_msg, messages=[{"role": "user", "content": prompt}]
    )
    return resp.content[0].text

def call_google(prompt, system_msg, model, api_key, temperature=0.2):
    """Call Google Gemini API via REST (no SDK required)."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "system_instruction": {"parts": [{"text": system_msg}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"}
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode())
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode()
        except Exception:
            body = str(e)
        raise RuntimeError(f"Google API error {e.code}: {body[:500]}")

def extract_json(text):
    """Extract JSON from text that might contain markdown code blocks or extra text."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code block
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON found in response", text, 0)

def call_with_retry(caller, prompt, system_msg, model, api_key, max_retries=3):
    """Call AI provider with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return caller(prompt, system_msg, model, api_key)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            time.sleep(wait)

PROVIDERS = {
    "openai": {"caller": call_openai, "default_model": "gpt-4o-mini"},
    "anthropic": {"caller": call_anthropic, "default_model": "claude-sonnet-4-20250514"},
    "google": {"caller": call_google, "default_model": "gemini-2.0-flash"},
}

MODEL_CATALOG = {
    "openai": [
        {"id": "gpt-4o-mini", "label": "GPT-4o Mini \u2014 fast & cheap"},
        {"id": "gpt-4o", "label": "GPT-4o \u2014 balanced"},
        {"id": "gpt-4-turbo", "label": "GPT-4 Turbo"},
        {"id": "o3-mini", "label": "o3-mini \u2014 reasoning"},
    ],
    "anthropic": [
        {"id": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4 \u2014 balanced"},
        {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5 \u2014 fast & cheap"},
        {"id": "claude-opus-4-20250514", "label": "Claude Opus 4 \u2014 most capable"},
    ],
    "google": [
        {"id": "gemini-2.0-flash", "label": "Gemini 2.0 Flash \u2014 fast & cheap"},
        {"id": "gemini-2.5-flash-preview-05-20", "label": "Gemini 2.5 Flash"},
        {"id": "gemini-2.5-pro-preview-05-06", "label": "Gemini 2.5 Pro \u2014 most capable"},
    ],
}

# --- Prompt Loader ---
PROMPT_DIR = Path(__file__).parent / "prompts"
EXAMPLE_DIR = Path(__file__).parent / "examples"

def load_prompt(filename):
    return (PROMPT_DIR / filename).read_text()

def load_examples(step_name):
    path = EXAMPLE_DIR / f"{step_name}.jsonl"
    if not path.exists():
        return ""
    lines = path.read_text().strip().split("\n")
    examples = [json.loads(l) for l in lines if l.strip()]
    if not examples:
        return ""
    parts = ["\n\n## FEW-SHOT EXAMPLES\n"]
    for i, ex in enumerate(examples, 1):
        parts.append(f"### Example {i}")
        parts.append(f"Title: {ex.get('title', 'N/A')}")
        parts.append(f"Abstract: {ex.get('abstract', 'N/A')[:500]}")
        parts.append(f"Expected output: {json.dumps(ex.get('expected', {}))}\n")
    return "\n".join(parts)

# --- Deduplication ---
def deduplicate_df(df):
    """Remove duplicate articles by DOI and normalized title. Returns (clean_df, dup_count, dup_details)."""
    title_col = "Title" if "Title" in df.columns else "title"
    original_count = len(df)
    dup_details = []

    # Phase 1: Exact DOI dedup
    doi_col = None
    for c in df.columns:
        if c.lower() == "doi":
            doi_col = c
            break
    if doi_col and df[doi_col].notna().any():
        doi_mask = df[doi_col].notna() & (df[doi_col].astype(str).str.strip() != "")
        doi_rows = df[doi_mask]
        doi_dupes = doi_rows[doi_rows[doi_col].astype(str).str.strip().str.lower().duplicated(keep="first")]
        for _, row in doi_dupes.iterrows():
            dup_details.append({"title": str(row[title_col])[:80], "reason": f"Duplicate DOI: {row[doi_col]}"})
        df = df.drop(doi_dupes.index)

    # Phase 2: Normalized title dedup
    def normalize_title(t):
        t = str(t).lower().strip()
        t = re.sub(r'[^a-z0-9\s]', '', t)
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    df = df.copy()
    df["_norm_title"] = df[title_col].apply(normalize_title)
    title_dupes = df[df["_norm_title"].duplicated(keep="first")]
    for _, row in title_dupes.iterrows():
        dup_details.append({"title": str(row[title_col])[:80], "reason": "Duplicate title"})
    df = df.drop(title_dupes.index)
    df = df.drop(columns=["_norm_title"])

    dup_count = original_count - len(df)
    return df.reset_index(drop=True), dup_count, dup_details

# --- File Parsing (shared logic) ---
def parse_uploaded_file(fpath):
    """Parse uploaded Excel/CSV file. Returns (df, error_string)."""
    fpath = str(fpath)
    try:
        if fpath.endswith(".csv"):
            df = pd.read_csv(fpath)
        else:
            df = pd.read_excel(fpath, header=1)
            if "Title" not in df.columns and "title" not in df.columns:
                df = pd.read_excel(fpath, header=0)
            if "Title" not in df.columns and "title" not in df.columns:
                xls = pd.ExcelFile(fpath)
                for s in xls.sheet_names:
                    df = pd.read_excel(xls, s, header=1)
                    if "Title" in df.columns or "title" in df.columns:
                        break
    except Exception as e:
        return None, f"Could not read file: {e}"

    col_map = {c: c.strip() for c in df.columns}
    df.rename(columns=col_map, inplace=True)

    if "Title" not in df.columns and "title" not in df.columns:
        return None, f"No 'Title' column found. Columns: {list(df.columns)[:10]}"

    title_col = "Title" if "Title" in df.columns else "title"
    df = df[df[title_col].notna() & (df[title_col].astype(str).str.len() > 5)]
    return df, None

# --- Classification Pipeline ---
STEPS = [
    ("01_screener.md",       "screener",  "status",         "Screen",    "Include / Exclude / Background"),
    ("02_path_classifier.md","path",      "path",           "Path",      "CG->ESG / Both / ESG->FP"),
    ("03_cg_tagger.md",      "cg",        "cg_mechanisms",  "CG Tags",   "Board, Ownership, etc."),
    ("04_esg_tagger.md",     "esg",       "esg_outcomes",   "ESG Tags",  "Disclosure, Rating, Pillars"),
    ("05_meta_scorer.md",    "meta",      "meta_potential",  "Meta Score","High / Medium / Low"),
]

def classify_article(row, provider_name, api_key, model, log_entry=None):
    provider = PROVIDERS[provider_name]
    caller = provider["caller"]
    title = str(row.get("Title", row.get("title", "")))
    abstract = str(row.get("Abstract", row.get("abstract", "")))
    keywords = str(row.get("Author_Keywords", row.get("author_keywords", "")))
    article_text = f"Title: {title}\nKeywords: {keywords}\nAbstract: {abstract[:2000]}"
    results = {"raw_responses": {}, "step_results": {}}

    for prompt_file, step_name, output_key, label, desc in STEPS:
        # Update live step indicator
        if log_entry is not None:
            log_entry["current_step"] = step_name

        if step_name != "screener" and results.get("status") in ("Exclude", "Background"):
            results["step_results"][step_name] = {"skipped": True, "reason": f"Article is {results.get('status')}"}
            if output_key == "path":
                results[output_key] = results.get("exclusion_code", "")
            else:
                results[output_key] = ""
            continue

        system_msg = load_prompt(prompt_file) + load_examples(step_name)
        try:
            raw = call_with_retry(caller, article_text, system_msg, model, api_key)
            results["raw_responses"][step_name] = raw
            parsed = extract_json(raw)
            results["step_results"][step_name] = parsed

            if step_name == "screener":
                results["status"] = parsed.get("status", "Exclude")
                results["exclusion_code"] = parsed.get("exclusion_code", "")
                results["screen_confidence"] = parsed.get("confidence", 0)
                results["screen_reasoning"] = parsed.get("reasoning", "")
            elif step_name == "path":
                results["path"] = parsed.get("path", "")
                results["path_confidence"] = parsed.get("confidence", 0)
            elif step_name == "cg":
                mechs = parsed.get("cg_mechanisms", [])
                results["cg_mechanisms"] = ";".join(mechs) if isinstance(mechs, list) else str(mechs)
                results["cg_confidence"] = parsed.get("confidence", 0)
            elif step_name == "esg":
                outcomes = parsed.get("esg_outcomes", [])
                results["esg_outcomes"] = ";".join(outcomes) if isinstance(outcomes, list) else str(outcomes)
                results["esg_confidence"] = parsed.get("confidence", 0)
            elif step_name == "meta":
                results["meta_potential"] = parsed.get("meta_potential", "")
                results["meta_confidence"] = parsed.get("confidence", 0)

        except Exception as e:
            results["raw_responses"][step_name] = f"ERROR: {str(e)}"
            results["step_results"][step_name] = {"error": str(e)}
            if step_name == "screener":
                results["status"] = "ERROR"
                results["exclusion_code"] = ""
            else:
                results[output_key] = "ERROR"

    if log_entry is not None:
        log_entry["current_step"] = "done"
    return results

# --- Job Management ---
JOBS = {}

def run_job(job_id, df, provider_name, api_key, model, max_workers=3):
    job = JOBS[job_id]
    job["status"] = "dedup"

    # Step 0: Deduplication
    clean_df, dup_count, dup_details = deduplicate_df(df)
    job["dedup"] = {"removed": dup_count, "kept": len(clean_df), "details": dup_details}
    df = clean_df

    job["status"] = "screening"
    job["total"] = len(df)
    job["log"] = []
    results_dict = {}

    # Pre-create log entries so UI can show all articles immediately
    work_items = []
    for i, (_, row) in enumerate(df.iterrows()):
        title_short = str(row.get("Title", row.get("title", "")))[:80]
        abstract_short = str(row.get("Abstract", row.get("abstract", "")))[:500]
        log_entry = {"i": i+1, "title": title_short, "abstract": abstract_short,
                     "status": "processing", "current_step": "queued",
                     "step_results": {}, "raw": {}}
        job["log"].append(log_entry)
        work_items.append((i, row, log_entry))

    def process_one(i, row, log_entry):
        log_entry["current_step"] = "screener"
        try:
            result = classify_article(row, provider_name, api_key, model, log_entry)
            result["_original_row"] = row.to_dict()
            log_entry["status"] = result.get("status", "?")
            log_entry["raw"] = result.get("raw_responses", {})
            log_entry["step_results"] = result.get("step_results", {})
            log_entry["summary"] = {
                "status": result.get("status", ""),
                "exclusion_code": result.get("exclusion_code", ""),
                "path": result.get("path", ""),
                "cg": result.get("cg_mechanisms", ""),
                "esg": result.get("esg_outcomes", ""),
                "meta": result.get("meta_potential", ""),
                "confidence": result.get("screen_confidence", ""),
                "reasoning": result.get("screen_reasoning", ""),
            }
            return i, result
        except Exception as e:
            job["errors"] += 1
            log_entry["status"] = f"ERROR: {e}"
            log_entry["current_step"] = "error"
            return i, {"status": "ERROR", "_original_row": row.to_dict()}

    effective_workers = max(1, min(max_workers, len(work_items)))
    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(process_one, i, row, le): i for i, row, le in work_items}
        for future in as_completed(futures):
            idx, result = future.result()
            results_dict[idx] = result
            job["progress"] = len(results_dict)

    # Build output Excel (preserve original order)
    results_list = [results_dict.get(i, {"status": "ERROR", "_original_row": {}}) for i in range(len(df))]
    out_rows = []
    for r in results_list:
        orig = r.get("_original_row", {})
        out_rows.append({
            "Title": orig.get("Title", orig.get("title", "")),
            "Authors": orig.get("Authors", orig.get("authors", "")),
            "Year": orig.get("Year", orig.get("year", "")),
            "Journal": orig.get("Journal", orig.get("journal", "")),
            "DOI": orig.get("DOI", orig.get("doi", "")),
            "Source": orig.get("Source", orig.get("source", "")),
            "Abstract": str(orig.get("Abstract", orig.get("abstract", "")))[:500],
            "AI_Status": r.get("status", ""),
            "AI_Exclusion_Code": r.get("exclusion_code", ""),
            "AI_Path": r.get("path", ""),
            "AI_CG_Mechanism": r.get("cg_mechanisms", ""),
            "AI_ESG_Outcome": r.get("esg_outcomes", ""),
            "AI_Meta_Potential": r.get("meta_potential", ""),
            "AI_Confidence": r.get("screen_confidence", ""),
            "AI_Reasoning": r.get("screen_reasoning", ""),
        })

    out_df = pd.DataFrame(out_rows)
    output_path = Path(__file__).parent / "output" / f"brain_results_{job_id[:8]}.xlsx"
    out_df.to_excel(output_path, index=False, sheet_name="AI_Results")
    job["output_path"] = str(output_path)
    job["progress"] = len(df)
    job["status"] = "done"

# --- Flask App ---
app = Flask(__name__)

@app.route("/")
def index():
    return HTML_PAGE

# --- Prompt CRUD ---
@app.route("/api/models")
def get_models():
    return jsonify(MODEL_CATALOG)

@app.route("/api/prompts")
def get_prompts():
    result = []
    for prompt_file, step_name, output_key, label, desc in STEPS:
        content = load_prompt(prompt_file)
        examples_path = EXAMPLE_DIR / f"{step_name}.jsonl"
        examples_raw = examples_path.read_text() if examples_path.exists() else ""
        examples_parsed = []
        if examples_raw.strip():
            for line in examples_raw.strip().split("\n"):
                if line.strip():
                    try:
                        examples_parsed.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        result.append({
            "file": prompt_file,
            "step": step_name,
            "label": label,
            "desc": desc,
            "content": content,
            "examples_raw": examples_raw,
            "examples": examples_parsed,
            "num_examples": len(examples_parsed),
        })
    return jsonify(result)

@app.route("/api/prompts/<step>", methods=["POST"])
def save_prompt(step):
    data = request.json
    for prompt_file, step_name, *_ in STEPS:
        if step_name == step:
            if "content" in data:
                (PROMPT_DIR / prompt_file).write_text(data["content"])
            if "examples" in data:
                (EXAMPLE_DIR / f"{step_name}.jsonl").write_text(data["examples"])
            return jsonify({"ok": True})
    return jsonify({"error": "Unknown step"}), 404

@app.route("/api/examples/<step>", methods=["GET"])
def get_examples(step):
    valid_steps = [s[1] for s in STEPS]
    if step not in valid_steps:
        return jsonify({"error": "Unknown step"}), 404
    path = EXAMPLE_DIR / f"{step}.jsonl"
    if not path.exists():
        return jsonify([])
    examples = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return jsonify(examples)

@app.route("/api/examples/<step>", methods=["POST"])
def add_example(step):
    valid_steps = [s[1] for s in STEPS]
    if step not in valid_steps:
        return jsonify({"error": "Unknown step"}), 404
    data = request.json
    example = {"title": data.get("title", ""), "abstract": data.get("abstract", ""), "expected": data.get("expected", {})}
    path = EXAMPLE_DIR / f"{step}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(example) + "\n")
    return jsonify({"ok": True})

@app.route("/api/examples/<step>/<int:index>", methods=["DELETE"])
def delete_example(step, index):
    valid_steps = [s[1] for s in STEPS]
    if step not in valid_steps:
        return jsonify({"error": "Unknown step"}), 404
    path = EXAMPLE_DIR / f"{step}.jsonl"
    if not path.exists():
        return jsonify({"error": "No examples file"}), 404
    lines = [l for l in path.read_text().strip().split("\n") if l.strip()]
    if index < 0 or index >= len(lines):
        return jsonify({"error": "Index out of range"}), 400
    lines.pop(index)
    path.write_text("\n".join(lines) + "\n" if lines else "")
    return jsonify({"ok": True})

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file"}), 400
    upload_dir = Path(__file__).parent / "output"
    upload_dir.mkdir(exist_ok=True)
    fpath = upload_dir / f"upload_{uuid.uuid4().hex[:8]}_{f.filename}"
    f.save(fpath)
    df, err = parse_uploaded_file(fpath)
    if err:
        return jsonify({"error": err}), 400
    return jsonify({"rows": len(df), "columns": list(df.columns), "file_path": str(fpath)})

@app.route("/run", methods=["POST"])
def run():
    data = request.json
    fpath = data.get("file_path")
    provider = data.get("provider", "openai")
    api_key = data.get("api_key", "")
    model = data.get("model", PROVIDERS[provider]["default_model"])
    max_articles = data.get("max_articles", 0)
    max_workers = data.get("max_workers", 3)
    if not api_key:
        return jsonify({"error": "API key required"}), 400
    df, err = parse_uploaded_file(fpath)
    if err:
        return jsonify({"error": err}), 400
    if max_articles and max_articles > 0:
        df = df.head(max_articles)
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"status": "queued", "progress": 0, "total": len(df), "errors": 0, "log": [], "dedup": None}
    thread = threading.Thread(target=run_job, args=(job_id, df, provider, api_key, model, max_workers))
    thread.start()
    return jsonify({"job_id": job_id, "total": len(df)})

@app.route("/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify({
        "status": job["status"], "progress": job["progress"], "total": job["total"],
        "errors": job["errors"], "log": job.get("log", []),
        "output_path": job.get("output_path", ""),
        "dedup": job.get("dedup"),
    })

@app.route("/download/<job_id>")
def download(job_id):
    job = JOBS.get(job_id)
    if not job or not job.get("output_path"):
        return jsonify({"error": "No output"}), 404
    return send_file(job["output_path"], as_attachment=True)

# =====================================================
# HTML UI v3 — Pipeline + Prompt Editor + Quick Correct + Multi-Provider
# =====================================================
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SLR Brain v3</title>
<style>
:root{--bg:#0b1120;--bg2:#111827;--card:#1e293b;--border:#334155;--accent:#3b82f6;
--green:#22c55e;--red:#ef4444;--yellow:#eab308;--orange:#f97316;--purple:#a855f7;
--text:#e2e8f0;--muted:#94a3b8;--dim:#475569;}
*{box-sizing:border-box;margin:0;padding:0;}
body{font-family:'Inter',-apple-system,sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
a{color:var(--accent);text-decoration:none;}

/* --- NAV TABS --- */
.topbar{background:var(--bg2);border-bottom:1px solid var(--border);padding:0 2rem;display:flex;align-items:center;gap:2rem;position:sticky;top:0;z-index:100;}
.topbar h1{font-size:1.1rem;padding:0.9rem 0;white-space:nowrap;}
.topbar .tabs{display:flex;gap:0;}
.topbar .tab{padding:0.9rem 1.25rem;font-size:0.85rem;font-weight:500;color:var(--muted);cursor:pointer;border-bottom:2px solid transparent;transition:all 0.15s;}
.topbar .tab:hover{color:var(--text);}
.topbar .tab.active{color:var(--accent);border-bottom-color:var(--accent);}
.page{display:none;padding:1.5rem 2rem;max-width:1400px;margin:0 auto;}
.page.active{display:block;}

/* --- CARDS --- */
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1.25rem;}
.card h2{font-size:0.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.06em;margin-bottom:0.75rem;}

/* --- FORMS --- */
label{display:block;font-size:0.78rem;color:var(--muted);margin-bottom:0.2rem;margin-top:0.6rem;}
select,input[type=text],input[type=password],input[type=number]{
  width:100%;padding:0.45rem 0.65rem;background:var(--bg);border:1px solid var(--border);
  border-radius:6px;color:var(--text);font-size:0.85rem;}
select:focus,input:focus,textarea:focus{outline:none;border-color:var(--accent);}
textarea{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:6px;
  color:var(--text);font-family:'JetBrains Mono','Fira Code',monospace;font-size:0.78rem;
  padding:0.75rem;line-height:1.5;resize:vertical;}

/* --- BUTTONS --- */
.btn{padding:0.5rem 1.2rem;border:none;border-radius:7px;font-weight:600;cursor:pointer;font-size:0.82rem;transition:all 0.12s;}
.btn-primary{background:var(--accent);color:white;}
.btn-primary:hover{background:#2563eb;}
.btn-primary:disabled{opacity:0.35;cursor:not-allowed;}
.btn-sm{padding:0.35rem 0.8rem;font-size:0.75rem;border-radius:5px;}
.btn-ghost{background:transparent;border:1px solid var(--border);color:var(--muted);}
.btn-ghost:hover{border-color:var(--accent);color:var(--text);}
.btn-green{background:var(--green);color:white;}
.btn-green:hover{background:#16a34a;}
.btn-row{display:flex;gap:0.5rem;align-items:center;margin-top:0.75rem;}

/* --- DROPZONE --- */
.dropzone{border:2px dashed var(--border);border-radius:10px;padding:2rem;text-align:center;cursor:pointer;transition:all 0.2s;}
.dropzone:hover,.dropzone.active{border-color:var(--accent);background:rgba(59,130,246,0.04);}
.dropzone .icon{font-size:2rem;margin-bottom:0.3rem;}
.dropzone p{color:var(--muted);font-size:0.82rem;}
#file-input{display:none;}
.file-info{margin-top:0.75rem;padding:0.6rem;background:rgba(59,130,246,0.08);border-radius:7px;font-size:0.82rem;}

/* --- PROMPT EDITOR --- */
.step-grid{display:grid;grid-template-columns:200px 1fr;gap:0;min-height:70vh;}
.step-nav{border-right:1px solid var(--border);padding:0.5rem 0;}
.step-nav-item{padding:0.65rem 1rem;cursor:pointer;font-size:0.82rem;display:flex;align-items:center;gap:0.6rem;
  color:var(--muted);transition:all 0.12s;border-left:3px solid transparent;}
.step-nav-item:hover{background:rgba(59,130,246,0.05);color:var(--text);}
.step-nav-item.active{background:rgba(59,130,246,0.08);color:var(--accent);border-left-color:var(--accent);}
.step-nav-item .num{width:22px;height:22px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:0.7rem;font-weight:700;background:var(--border);color:var(--text);}
.step-nav-item.active .num{background:var(--accent);color:white;}
.step-editor{padding:1.25rem 1.5rem;}
.step-editor .step-title{font-size:1rem;font-weight:600;margin-bottom:0.15rem;}
.step-editor .step-desc{color:var(--muted);font-size:0.8rem;margin-bottom:1rem;}
.saved-toast{display:inline-block;color:var(--green);font-size:0.78rem;margin-left:0.75rem;opacity:0;transition:opacity 0.3s;}
.saved-toast.show{opacity:1;}
.example-count{font-size:0.72rem;color:var(--dim);margin-left:auto;}

/* --- PIPELINE TRACKER --- */
.pipeline-tracker{display:flex;align-items:center;gap:0;margin-bottom:1.25rem;padding:0.75rem 1rem;
  background:var(--bg2);border:1px solid var(--border);border-radius:10px;overflow-x:auto;}
.pipe-step{display:flex;align-items:center;gap:0.5rem;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.75rem;
  font-weight:600;white-space:nowrap;transition:all 0.3s;}
.pipe-step .pipe-num{width:20px;height:20px;border-radius:50%;display:flex;align-items:center;justify-content:center;
  font-size:0.65rem;font-weight:700;background:var(--border);color:var(--muted);}
.pipe-step .pipe-label{color:var(--muted);}
.pipe-step .pipe-count{font-size:0.7rem;color:var(--dim);margin-left:0.25rem;}
.pipe-arrow{color:var(--dim);font-size:0.75rem;margin:0 0.15rem;flex-shrink:0;}

.pipe-step.active{background:rgba(59,130,246,0.12);}
.pipe-step.active .pipe-num{background:var(--accent);color:white;}
.pipe-step.active .pipe-label{color:var(--accent);}

.pipe-step.done{background:rgba(34,197,94,0.08);}
.pipe-step.done .pipe-num{background:var(--green);color:white;}
.pipe-step.done .pipe-label{color:var(--green);}

/* --- PROGRESS / STATS --- */
.stats-row{display:flex;gap:1.2rem;margin-bottom:0.75rem;flex-wrap:wrap;}
.stat-box{text-align:center;min-width:55px;}
.stat-box .num{font-size:1.3rem;font-weight:700;}
.stat-box .label{font-size:0.7rem;color:var(--muted);}
.progress-bar{width:100%;height:6px;background:var(--bg);border-radius:3px;overflow:hidden;margin:0.5rem 0;}
.progress-fill{height:100%;background:var(--accent);border-radius:3px;transition:width 0.3s;width:0%;}

/* --- DEDUP BANNER --- */
.dedup-banner{padding:0.6rem 1rem;border-radius:8px;font-size:0.82rem;margin-bottom:0.75rem;display:none;
  align-items:center;gap:0.75rem;border:1px solid;}
.dedup-banner.show{display:flex;}
.dedup-banner.has-dups{background:rgba(234,179,8,0.08);border-color:rgba(234,179,8,0.25);color:var(--yellow);}
.dedup-banner.no-dups{background:rgba(34,197,94,0.08);border-color:rgba(34,197,94,0.25);color:var(--green);}

/* --- RESULTS TABLE --- */
.results-wrap{overflow-x:auto;margin-top:1rem;}
table.results{width:100%;border-collapse:collapse;font-size:0.75rem;}
table.results th{background:var(--bg2);color:var(--muted);font-weight:600;text-transform:uppercase;
  letter-spacing:0.04em;padding:0.5rem 0.6rem;text-align:left;position:sticky;top:0;white-space:nowrap;font-size:0.7rem;}
table.results td{padding:0.45rem 0.6rem;border-bottom:1px solid rgba(51,65,85,0.3);vertical-align:top;max-width:200px;overflow:hidden;text-overflow:ellipsis;}
table.results tr:hover td{background:rgba(59,130,246,0.04);}
table.results .expand-btn{color:var(--accent);cursor:pointer;font-size:0.7rem;white-space:nowrap;}

/* Badges */
.badge{padding:0.15rem 0.5rem;border-radius:4px;font-size:0.7rem;font-weight:600;white-space:nowrap;display:inline-block;}
.b-include{background:rgba(34,197,94,0.15);color:var(--green);}
.b-exclude{background:rgba(239,68,68,0.15);color:var(--red);}
.b-background{background:rgba(234,179,8,0.15);color:var(--yellow);}
.b-error{background:rgba(239,68,68,0.3);color:var(--red);}
.b-processing{background:rgba(59,130,246,0.15);color:var(--accent);}
.b-high{background:rgba(34,197,94,0.15);color:var(--green);}
.b-medium{background:rgba(234,179,8,0.15);color:var(--yellow);}
.b-low{background:rgba(148,163,184,0.15);color:var(--muted);}

/* Raw detail row */
.detail-row td{background:var(--bg2) !important;padding:0 !important;}
.detail-inner{padding:0.75rem 1rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:0.75rem;}
.detail-box{background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:0.6rem;font-size:0.72rem;}
.detail-box h4{color:var(--accent);font-size:0.72rem;margin-bottom:0.3rem;text-transform:uppercase;letter-spacing:0.04em;}
.detail-box pre{white-space:pre-wrap;word-break:break-word;color:var(--muted);line-height:1.4;margin:0;}

.download-bar{margin-top:1rem;padding:0.75rem 1rem;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);
  border-radius:8px;display:none;align-items:center;justify-content:space-between;}
.download-bar.show{display:flex;}

/* --- SUMMARY CARDS ROW --- */
.summary-cards{display:none;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:0.75rem;margin-bottom:1rem;}
.summary-cards.show{display:grid;}
.summary-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:0.75rem;text-align:center;}
.summary-card .sc-num{font-size:1.5rem;font-weight:700;line-height:1.2;}
.summary-card .sc-label{font-size:0.7rem;color:var(--muted);margin-top:0.15rem;}

/* --- FILTER TABS --- */
.filter-tabs{display:flex;gap:0.25rem;margin-bottom:0.75rem;flex-wrap:wrap;}
.filter-tab{padding:0.3rem 0.7rem;border-radius:5px;font-size:0.72rem;font-weight:600;cursor:pointer;
  background:var(--bg);border:1px solid var(--border);color:var(--muted);transition:all 0.12s;}
.filter-tab:hover{border-color:var(--accent);color:var(--text);}
.filter-tab.active{background:rgba(59,130,246,0.12);border-color:var(--accent);color:var(--accent);}
.filter-tab .ft-count{margin-left:0.3rem;font-weight:400;opacity:0.7;}

.two-col{display:grid;grid-template-columns:340px 1fr;gap:1.25rem;}
@media(max-width:900px){.two-col{grid-template-columns:1fr;}.step-grid{grid-template-columns:1fr;}}

/* --- EXAMPLE CARDS --- */
.example-cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:0.6rem;margin-top:0.75rem;}
.example-card{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:0.65rem 0.75rem;position:relative;transition:border-color 0.15s;}
.example-card:hover{border-color:var(--accent);}
.example-card .ec-title{font-weight:600;font-size:0.8rem;margin-bottom:0.2rem;line-height:1.3;padding-right:1.5rem;}
.example-card .ec-abstract{font-size:0.7rem;color:var(--muted);line-height:1.35;margin-bottom:0.4rem;max-height:2.8em;overflow:hidden;}
.example-card .ec-expected{display:flex;gap:0.25rem;flex-wrap:wrap;}
.example-card .ec-delete{position:absolute;top:0.4rem;right:0.5rem;background:none;border:none;color:var(--dim);cursor:pointer;font-size:1.1rem;line-height:1;padding:0.15rem 0.25rem;border-radius:3px;}
.example-card .ec-delete:hover{color:var(--red);background:rgba(239,68,68,0.1);}
.badge-sm{padding:0.1rem 0.35rem;border-radius:3px;font-size:0.62rem;font-weight:600;white-space:nowrap;display:inline-block;}

/* --- ADD EXAMPLE FORM --- */
.add-form{background:var(--bg);border:1px solid var(--accent);border-radius:8px;padding:1rem;margin-top:0.5rem;display:none;}
.add-form.show{display:block;}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;}
.checkbox-group{display:flex;flex-wrap:wrap;gap:0.5rem 0.75rem;margin-top:0.25rem;}
.checkbox-group label{display:flex;align-items:center;gap:0.3rem;font-size:0.76rem;color:var(--text);cursor:pointer;margin:0;}
.checkbox-group input[type=checkbox]{margin:0;accent-color:var(--accent);}

/* --- MODE TOGGLE --- */
.mode-toggle{display:flex;gap:0.25rem;margin-bottom:0.75rem;}
.mode-toggle button{padding:0.3rem 0.65rem;border-radius:5px;font-size:0.72rem;font-weight:600;cursor:pointer;
  background:var(--bg);border:1px solid var(--border);color:var(--muted);transition:all 0.12s;}
.mode-toggle button.active{background:rgba(59,130,246,0.12);border-color:var(--accent);color:var(--accent);}

/* --- TIMER --- */
.timer-display{font-size:0.75rem;color:var(--muted);margin-top:0.3rem;min-height:1.1em;}

/* --- QUICK CORRECT --- */
.qc-form{margin-top:0.5rem;padding:0.5rem;background:var(--card);border:1px solid var(--accent);border-radius:6px;}
.qc-form select,.qc-form input{font-size:0.72rem;padding:0.25rem 0.4rem;margin:0.15rem 0;}
.qc-form label{font-size:0.68rem;margin:0.15rem 0 0;display:flex;align-items:center;gap:0.25rem;}
.qc-saved{color:var(--green);font-size:0.72rem;font-weight:600;}
</style>
</head>
<body>

<!-- TOP NAV -->
<div class="topbar">
  <h1>SLR Brain</h1>
  <div class="tabs">
    <div class="tab active" onclick="showPage('run')">Run Pipeline</div>
    <div class="tab" onclick="showPage('prompts')">Prompt Editor</div>
    <div class="tab" onclick="showPage('results')">Results Table</div>
  </div>
</div>

<!-- ============== PAGE 1: RUN ============== -->
<div class="page active" id="page-run">

<!-- Pipeline Tracker -->
<div class="pipeline-tracker" id="pipeline-tracker">
  <div class="pipe-step" id="pipe-upload"><span class="pipe-num">0</span><span class="pipe-label">Upload</span></div>
  <span class="pipe-arrow">&rarr;</span>
  <div class="pipe-step" id="pipe-dedup"><span class="pipe-num">1</span><span class="pipe-label">Deduplicate</span><span class="pipe-count"></span></div>
  <span class="pipe-arrow">&rarr;</span>
  <div class="pipe-step" id="pipe-screen"><span class="pipe-num">2</span><span class="pipe-label">Screen</span><span class="pipe-count"></span></div>
  <span class="pipe-arrow">&rarr;</span>
  <div class="pipe-step" id="pipe-path"><span class="pipe-num">3</span><span class="pipe-label">Path</span><span class="pipe-count"></span></div>
  <span class="pipe-arrow">&rarr;</span>
  <div class="pipe-step" id="pipe-cg"><span class="pipe-num">4</span><span class="pipe-label">CG Tags</span><span class="pipe-count"></span></div>
  <span class="pipe-arrow">&rarr;</span>
  <div class="pipe-step" id="pipe-esg"><span class="pipe-num">5</span><span class="pipe-label">ESG Tags</span><span class="pipe-count"></span></div>
  <span class="pipe-arrow">&rarr;</span>
  <div class="pipe-step" id="pipe-meta"><span class="pipe-num">6</span><span class="pipe-label">Meta Score</span><span class="pipe-count"></span></div>
</div>

<div class="two-col">
  <div>
    <div class="card">
      <h2>Upload Articles</h2>
      <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
        <div class="icon">+</div>
        <p>Drop Excel / CSV here</p>
      </div>
      <input type="file" id="file-input" accept=".xlsx,.xls,.csv">
      <div class="file-info" id="file-info" style="display:none;"></div>
    </div>
    <div class="card" style="margin-top:1rem;">
      <h2>Configure</h2>
      <label>Provider</label>
      <select id="provider" onchange="updateModelDropdown()">
        <option value="openai">OpenAI</option>
        <option value="anthropic">Anthropic (Claude)</option>
        <option value="google">Google (Gemini)</option>
      </select>
      <label>Model</label>
      <select id="model" onchange="onModelChange()"></select>
      <input type="text" id="model-custom" placeholder="Type custom model ID" style="display:none;margin-top:0.3rem;">
      <label>API Key</label>
      <input type="password" id="api-key" placeholder="sk-..." onblur="saveApiKey()">
      <div class="form-row" style="margin-top:0.5rem;">
        <div>
          <label>Max articles (0 = all)</label>
          <input type="number" id="max-articles" value="5" min="0">
        </div>
        <div>
          <label>Parallel workers</label>
          <input type="number" id="max-workers" value="3" min="1" max="10">
        </div>
      </div>
      <div class="btn-row">
        <button class="btn btn-primary" id="run-btn" onclick="startRun()" disabled>Run Pipeline</button>
        <button class="btn btn-ghost" onclick="location.reload()">Reset</button>
      </div>
    </div>
  </div>
  <div>
    <!-- Dedup Banner -->
    <div class="dedup-banner" id="dedup-banner"></div>

    <div class="card">
      <h2>Progress</h2>
      <div class="stats-row">
        <div class="stat-box"><div class="num" id="s-done">0</div><div class="label">Done</div></div>
        <div class="stat-box"><div class="num" id="s-total">0</div><div class="label">Total</div></div>
        <div class="stat-box"><div class="num" style="color:var(--green)" id="s-inc">0</div><div class="label">Include</div></div>
        <div class="stat-box"><div class="num" style="color:var(--red)" id="s-exc">0</div><div class="label">Exclude</div></div>
        <div class="stat-box"><div class="num" style="color:var(--yellow)" id="s-bg">0</div><div class="label">Background</div></div>
        <div class="stat-box"><div class="num" style="color:var(--red)" id="s-err">0</div><div class="label">Errors</div></div>
      </div>
      <div class="progress-bar"><div class="progress-fill" id="pbar"></div></div>
      <div class="timer-display" id="timer-display"></div>
      <div class="download-bar" id="download-bar">
        <span style="color:var(--green);font-weight:600;">Pipeline complete!</span>
        <div style="display:flex;gap:0.5rem;">
          <button class="btn btn-green btn-sm" onclick="downloadResult()">Download .xlsx</button>
          <button class="btn btn-ghost btn-sm" onclick="showPage('results')">View Results</button>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:1rem;">
      <h2>Live Feed</h2>
      <div id="live-feed" style="max-height:50vh;overflow-y:auto;font-size:0.78rem;font-family:monospace;">
        <div style="color:var(--dim);padding:1rem;text-align:center;">Upload a file and click Run Pipeline</div>
      </div>
    </div>
  </div>
</div>
</div>

<!-- ============== PAGE 2: PROMPT EDITOR ============== -->
<div class="page" id="page-prompts">
<div class="card" style="padding:0;overflow:hidden;">
  <div class="step-grid">
    <div class="step-nav" id="step-nav"></div>
    <div class="step-editor" id="step-editor">
      <div style="color:var(--dim);padding:2rem;text-align:center;">Loading prompts...</div>
    </div>
  </div>
</div>
</div>

<!-- ============== PAGE 3: RESULTS TABLE ============== -->
<div class="page" id="page-results">
<!-- Summary Cards -->
<div class="summary-cards" id="summary-cards">
  <div class="summary-card"><div class="sc-num" id="sc-total" style="color:var(--accent)">0</div><div class="sc-label">Total Screened</div></div>
  <div class="summary-card"><div class="sc-num" id="sc-dedup" style="color:var(--yellow)">0</div><div class="sc-label">Duplicates Removed</div></div>
  <div class="summary-card"><div class="sc-num" id="sc-include" style="color:var(--green)">0</div><div class="sc-label">Included</div></div>
  <div class="summary-card"><div class="sc-num" id="sc-exclude" style="color:var(--red)">0</div><div class="sc-label">Excluded</div></div>
  <div class="summary-card"><div class="sc-num" id="sc-background" style="color:var(--yellow)">0</div><div class="sc-label">Background</div></div>
  <div class="summary-card"><div class="sc-num" id="sc-patha" style="color:var(--accent)">0</div><div class="sc-label">Path A</div></div>
  <div class="summary-card"><div class="sc-num" id="sc-both" style="color:var(--purple)">0</div><div class="sc-label">Both A+B</div></div>
  <div class="summary-card"><div class="sc-num" id="sc-pathb" style="color:var(--orange)">0</div><div class="sc-label">Path B</div></div>
</div>

<div class="card">
  <h2>Results <span style="font-weight:400;text-transform:none;letter-spacing:0;" id="result-count"></span></h2>
  <!-- Filter Tabs -->
  <div class="filter-tabs" id="filter-tabs"></div>
  <div class="results-wrap">
    <table class="results" id="results-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Title</th>
          <th>Step 1: Screen</th>
          <th>Code</th>
          <th>Step 2: Path</th>
          <th>Step 3: CG Tags</th>
          <th>Step 4: ESG Tags</th>
          <th>Step 5: Meta</th>
          <th>Conf.</th>
          <th>Detail</th>
        </tr>
      </thead>
      <tbody id="results-body">
        <tr><td colspan="10" style="text-align:center;color:var(--dim);padding:2rem;">Run the pipeline first</td></tr>
      </tbody>
    </table>
  </div>
</div>
</div>

<script>
// --- State ---
let filePath=null, jobId=null, pollInterval=null;
let allPrompts=[], activeStep=0, allLogs=[], lastDedup=null;
let activeFilter='all';
let editorMode='simple';
let showAddForm=false;
let runStartTime=null;

// --- Model Catalog ---
const MODEL_CATALOG={
  openai:[
    {id:'gpt-4o-mini',label:'GPT-4o Mini \u2014 fast & cheap'},
    {id:'gpt-4o',label:'GPT-4o \u2014 balanced'},
    {id:'gpt-4-turbo',label:'GPT-4 Turbo'},
    {id:'o3-mini',label:'o3-mini \u2014 reasoning'},
  ],
  anthropic:[
    {id:'claude-sonnet-4-20250514',label:'Claude Sonnet 4 \u2014 balanced'},
    {id:'claude-haiku-4-5-20251001',label:'Claude Haiku 4.5 \u2014 fast & cheap'},
    {id:'claude-opus-4-20250514',label:'Claude Opus 4 \u2014 most capable'},
  ],
  google:[
    {id:'gemini-2.0-flash',label:'Gemini 2.0 Flash \u2014 fast & cheap'},
    {id:'gemini-2.5-flash-preview-05-20',label:'Gemini 2.5 Flash'},
    {id:'gemini-2.5-pro-preview-05-06',label:'Gemini 2.5 Pro \u2014 most capable'},
  ]
};
const CG_MECHANISMS=['Board_Structure_Composition','Board_Diversity','Board_Leadership','Board_Committee','Ownership_Structure','Ownership_Concentration','Mixed_CG_Mechanisms'];
const ESG_OUTCOMES=['ESG_Disclosure_Reporting','ESG_Performance_Rating','CSR_Reporting','E_Pillar','S_Pillar','G_Pillar','Integrated_Reporting','Sustainability_Assurance'];
const EXCLUSION_CODES=['TA-E1','TA-E2','TA-E5','TA-B1'];
const PATH_OPTIONS=['Path_A_CG_to_ESG','Both_A_and_B','Path_B_ESG_to_FP_with_CG_moderation'];
const META_OPTIONS=['High','Medium','Low'];
const STATUS_OPTIONS=['Include','Exclude','Background'];

// --- Page Nav ---
function showPage(id){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  document.querySelector(`.tab[onclick="showPage('${id}')"]`).classList.add('active');
  if(id==='prompts'&&allPrompts.length===0) loadPrompts();
  if(id==='results') renderResultsTable();
}

// --- Pipeline Tracker ---
function updatePipelineTracker(status, logs, dedup){
  const steps=['upload','dedup','screen','path','cg','esg','meta'];
  // Reset all
  steps.forEach(s=>{
    const el=document.getElementById('pipe-'+s);
    el.className='pipe-step';
    const cnt=el.querySelector('.pipe-count');
    if(cnt) cnt.textContent='';
  });

  if(!status||status==='queued') {
    if(filePath) document.getElementById('pipe-upload').classList.add('done');
    return;
  }

  // Upload always done once running
  document.getElementById('pipe-upload').classList.add('done');

  if(status==='dedup'){
    document.getElementById('pipe-dedup').classList.add('active');
    return;
  }

  // Dedup done
  if(dedup){
    document.getElementById('pipe-dedup').classList.add('done');
    const dc=document.getElementById('pipe-dedup').querySelector('.pipe-count');
    if(dc) dc.textContent=dedup.removed>0?`-${dedup.removed}`:'';
  }

  if(status==='screening'||status==='done'){
    let incCount=0;
    (logs||[]).forEach(l=>{
      if(l.status==='Include') incCount++;
    });

    const finished=logs?logs.filter(l=>l.status&&l.status!=='processing').length:0;

    // Screen step
    const screenEl=document.getElementById('pipe-screen');
    if(finished>0||status==='done') screenEl.classList.add(status==='done'?'done':'active');
    const screenCnt=screenEl.querySelector('.pipe-count');
    if(screenCnt&&finished>0) screenCnt.textContent=`${finished}`;

    // Steps 3-6 only for included
    ['path','cg','esg','meta'].forEach(s=>{
      const el=document.getElementById('pipe-'+s);
      if(incCount>0) el.classList.add(status==='done'?'done':'active');
      const cnt=el.querySelector('.pipe-count');
      if(cnt&&incCount>0) cnt.textContent=`${incCount}`;
    });
  }
}

// --- Upload ---
const dz=document.getElementById('dropzone'), fi=document.getElementById('file-input');
['dragenter','dragover'].forEach(e=>dz.addEventListener(e,ev=>{ev.preventDefault();dz.classList.add('active');}));
['dragleave','drop'].forEach(e=>dz.addEventListener(e,ev=>{ev.preventDefault();dz.classList.remove('active');}));
dz.addEventListener('drop',ev=>{if(ev.dataTransfer.files.length)uploadFile(ev.dataTransfer.files[0]);});
fi.addEventListener('change',ev=>{if(ev.target.files.length)uploadFile(ev.target.files[0]);});

function uploadFile(file){
  const fd=new FormData(); fd.append('file',file);
  document.getElementById('file-info').style.display='block';
  document.getElementById('file-info').innerHTML='Uploading...';
  fetch('/upload',{method:'POST',body:fd}).then(r=>r.json()).then(d=>{
    if(d.error){document.getElementById('file-info').innerHTML='<span style="color:var(--red)">'+escHtml(d.error)+'</span>';return;}
    filePath=d.file_path;
    document.getElementById('file-info').innerHTML='<strong>'+escHtml(file.name)+'</strong> — '+d.rows+' articles found';
    document.getElementById('run-btn').disabled=false;
    document.getElementById('s-total').textContent=d.rows;
    updatePipelineTracker(null);
  });
}

// --- Model Dropdown ---
function updateModelDropdown(){
  const prov=document.getElementById('provider').value;
  const models=MODEL_CATALOG[prov]||[];
  const sel=document.getElementById('model');
  sel.innerHTML=models.map(m=>'<option value="'+m.id+'">'+m.label+'</option>').join('')+
    '<option value="_custom">Other (type model ID)</option>';
  document.getElementById('model-custom').style.display='none';
  loadSavedKey();
}
function onModelChange(){
  document.getElementById('model-custom').style.display=
    document.getElementById('model').value==='_custom'?'block':'none';
}
function getSelectedModel(){
  const v=document.getElementById('model').value;
  return v==='_custom'?document.getElementById('model-custom').value:v;
}
// --- API Key Persistence ---
function saveApiKey(){
  const prov=document.getElementById('provider').value;
  const key=document.getElementById('api-key').value;
  if(key) localStorage.setItem('slr_key_'+prov,key);
}
function loadSavedKey(){
  const prov=document.getElementById('provider').value;
  const saved=localStorage.getItem('slr_key_'+prov);
  if(saved) document.getElementById('api-key').value=saved;
}
// --- Timer ---
function formatTime(s){
  const m=Math.floor(s/60),sec=Math.floor(s%60);
  return m>0?m+'m '+sec+'s':sec+'s';
}

// --- Run ---
function startRun(){
  const key=document.getElementById('api-key').value;
  if(!key){alert('Enter API key');return;} if(!filePath){alert('Upload file first');return;}
  saveApiKey();
  runStartTime=Date.now();
  document.getElementById('run-btn').disabled=true;
  document.getElementById('live-feed').innerHTML='';
  document.getElementById('timer-display').textContent='';
  allLogs=[]; lastDedup=null;
  fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    file_path:filePath, provider:document.getElementById('provider').value,
    api_key:key, model:getSelectedModel(),
    max_articles:parseInt(document.getElementById('max-articles').value)||0,
    max_workers:parseInt(document.getElementById('max-workers').value)||3
  })}).then(r=>r.json()).then(d=>{
    if(d.error){alert(d.error);document.getElementById('run-btn').disabled=false;return;}
    jobId=d.job_id; document.getElementById('s-total').textContent=d.total;
    pollInterval=setInterval(pollStatus,800);
  });
}

function pollStatus(){
  if(!jobId)return;
  fetch('/status/'+jobId).then(r=>r.json()).then(d=>{
    document.getElementById('s-done').textContent=d.progress;
    document.getElementById('s-total').textContent=d.total;
    document.getElementById('s-err').textContent=d.errors;
    document.getElementById('pbar').style.width=(d.total>0?(d.progress/d.total*100):0)+'%';
    allLogs=d.log||[];

    // Dedup banner
    if(d.dedup){
      lastDedup=d.dedup;
      const banner=document.getElementById('dedup-banner');
      if(d.dedup.removed>0){
        banner.className='dedup-banner show has-dups';
        banner.innerHTML='<strong>Step 1 — Deduplication:</strong> Removed '+d.dedup.removed+' duplicates. '+d.dedup.kept+' unique articles proceed to screening.';
      } else {
        banner.className='dedup-banner show no-dups';
        banner.innerHTML='<strong>Step 1 — Deduplication:</strong> No duplicates found. All '+d.dedup.kept+' articles proceed to screening.';
      }
    }

    let inc=0,exc=0,bg=0;
    allLogs.forEach(l=>{
      if(l.status==='Include')inc++;
      if(l.status==='Exclude')exc++;
      if(l.status==='Background')bg++;
    });
    document.getElementById('s-inc').textContent=inc;
    document.getElementById('s-exc').textContent=exc;
    document.getElementById('s-bg').textContent=bg;

    updatePipelineTracker(d.status, allLogs, d.dedup);
    renderLiveFeed(allLogs.slice(-40));

    // Timer/ETA
    if(runStartTime&&(d.status==='screening'||d.status==='done')){
      const elapsed=(Date.now()-runStartTime)/1000;
      const timerEl=document.getElementById('timer-display');
      if(d.status==='done'){
        timerEl.textContent='Completed in '+formatTime(elapsed);
      } else if(d.progress>0){
        const eta=elapsed/d.progress*(d.total-d.progress);
        timerEl.textContent=formatTime(elapsed)+' elapsed \u00b7 ~'+formatTime(eta)+' remaining';
      } else {
        timerEl.textContent=formatTime(elapsed)+' elapsed \u00b7 starting...';
      }
    }

    if(d.status==='done'){
      clearInterval(pollInterval);
      document.getElementById('download-bar').classList.add('show');
      document.getElementById('run-btn').disabled=false;
    }
  });
}

function renderLiveFeed(logs){
  const el=document.getElementById('live-feed');
  const stepLabels={screener:'Screening',path:'Path',cg:'CG Tags',esg:'ESG Tags',meta:'Meta Score',done:'Done',error:'Error'};
  el.innerHTML=logs.map(l=>{
    const bc=l.status==='Include'?'b-include':l.status==='Exclude'?'b-exclude':
             l.status==='Background'?'b-background':l.status?.startsWith('ERROR')?'b-error':'b-processing';
    const s=l.summary||{};
    const curStep=l.current_step?stepLabels[l.current_step]||l.current_step:'';
    let extra='';
    if(l.status==='processing'&&curStep){
      extra=' <span style="color:var(--accent);font-size:0.7rem;">'+curStep+'...</span>';
    } else {
      if(s.exclusion_code) extra+=' <span style="color:var(--dim);font-size:0.7rem;">'+escHtml(s.exclusion_code)+'</span>';
      if(s.path) extra+=' <span style="color:var(--dim);font-size:0.7rem;">'+escHtml(s.path.replace('Path_A_CG_to_ESG','A').replace('Both_A_and_B','A+B').replace('Path_B_ESG_to_FP_with_CG_moderation','B'))+'</span>';
      if(s.meta) extra+=' <span class="badge b-'+s.meta.toLowerCase()+'">'+escHtml(s.meta)+'</span>';
    }
    return '<div style="padding:0.3rem 0;border-bottom:1px solid rgba(51,65,85,0.3);display:flex;gap:0.5rem;align-items:center;">'+
      '<span style="color:var(--dim);min-width:1.5rem;text-align:right;">'+l.i+'</span>'+
      '<span class="badge '+bc+'">'+(l.status||'...')+'</span>'+extra+
      '<span style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'+escHtml(l.title||'')+'</span>'+
    '</div>';
  }).join('');
  el.scrollTop=el.scrollHeight;
}

function downloadResult(){if(jobId)window.location.href='/download/'+jobId;}

// --- Prompt Editor ---
function loadPrompts(){
  fetch('/api/prompts').then(r=>r.json()).then(data=>{
    allPrompts=data;
    renderStepNav();
    selectStep(0);
  });
}
function renderStepNav(){
  const nav=document.getElementById('step-nav');
  nav.innerHTML=allPrompts.map((p,i)=>`
    <div class="step-nav-item ${i===activeStep?'active':''}" onclick="selectStep(${i})">
      <span class="num">${i+1}</span>
      <span>${escHtml(p.label)}</span>
      <span class="example-count">${p.num_examples} ex</span>
    </div>`).join('');
}
function selectStep(i){
  activeStep=i; showAddForm=false;
  renderStepNav(); renderStepEditor();
}
function setEditorMode(m){ editorMode=m; renderStepEditor(); }
function toggleAddForm(){ showAddForm=!showAddForm; renderStepEditor(); }

function renderStepEditor(){
  const p=allPrompts[activeStep];
  const ed=document.getElementById('step-editor');
  if(editorMode==='advanced'){
    ed.innerHTML=`
      <div class="step-title">${escHtml(p.label)}</div>
      <div class="step-desc">${escHtml(p.desc)} \u2014 <code>${escHtml(p.file)}</code></div>
      <div class="mode-toggle">
        <button onclick="setEditorMode('simple')">Simple</button>
        <button class="active" onclick="setEditorMode('advanced')">Advanced</button>
      </div>
      <label>Prompt Logic (Markdown)</label>
      <textarea id="prompt-content" rows="18">${escHtml(p.content)}</textarea>
      <label style="margin-top:0.75rem;">Few-Shot Examples (JSONL \u2014 one JSON per line)</label>
      <textarea id="prompt-examples" rows="10">${escHtml(p.examples_raw||'')}</textarea>
      <div class="btn-row">
        <button class="btn btn-primary btn-sm" onclick="saveStepAdvanced()">Save Changes</button>
        <button class="btn btn-ghost btn-sm" onclick="selectStep(${activeStep})">Revert</button>
        <span class="saved-toast" id="save-toast">Saved!</span>
      </div>`;
    return;
  }
  // Simple mode
  const examples=p.examples||[];
  const cards=examples.map((ex,idx)=>renderExCard(ex,idx,p.step)).join('');
  ed.innerHTML=`
    <div class="step-title">${escHtml(p.label)}</div>
    <div class="step-desc">${escHtml(p.desc)}</div>
    <div class="mode-toggle">
      <button class="active" onclick="setEditorMode('simple')">Simple</button>
      <button onclick="setEditorMode('advanced')">Advanced</button>
    </div>
    <div style="display:flex;justify-content:space-between;align-items:center;">
      <span style="font-size:0.82rem;font-weight:600;">${examples.length} Training Examples</span>
      <button class="btn btn-primary btn-sm" onclick="toggleAddForm()">${showAddForm?'\u2212 Cancel':'+ Add Example'}</button>
    </div>
    <div class="add-form ${showAddForm?'show':''}" id="add-form">${renderAddForm(p.step)}</div>
    <div class="example-cards">${cards||'<div style="color:var(--dim);padding:1.5rem;text-align:center;grid-column:1/-1;">No examples yet. Click + Add Example above.</div>'}</div>
    <span class="saved-toast" id="save-toast">Saved!</span>`;
}

function renderExCard(ex,idx,step){
  const e=ex.expected||{};
  let badges='';
  if(e.status) badges+='<span class="badge-sm b-'+e.status.toLowerCase()+'">'+e.status+'</span> ';
  if(e.exclusion_code) badges+='<span class="badge-sm" style="background:rgba(239,68,68,0.1);color:var(--red);">'+e.exclusion_code+'</span> ';
  if(e.path) badges+='<span class="badge-sm" style="background:rgba(59,130,246,0.12);color:var(--accent);">'+e.path.replace('Path_A_CG_to_ESG','Path A').replace('Both_A_and_B','Both A+B').replace('Path_B_ESG_to_FP_with_CG_moderation','Path B')+'</span> ';
  if(e.meta_potential) badges+='<span class="badge-sm b-'+e.meta_potential.toLowerCase()+'">'+e.meta_potential+'</span> ';
  if(e.cg_mechanisms) e.cg_mechanisms.forEach(m=>{badges+='<span class="badge-sm" style="background:rgba(168,85,247,0.12);color:var(--purple);">'+m.replace(/_/g,' ')+'</span> ';});
  if(e.esg_outcomes) e.esg_outcomes.forEach(m=>{badges+='<span class="badge-sm" style="background:rgba(249,115,22,0.12);color:var(--orange);">'+m.replace(/_/g,' ')+'</span> ';});
  return '<div class="example-card">'+
    '<button class="ec-delete" onclick="deleteExample(\''+step+'\','+idx+')" title="Delete">\u00d7</button>'+
    '<div class="ec-title">'+escHtml((ex.title||'').substring(0,75))+'</div>'+
    '<div class="ec-abstract">'+escHtml((ex.abstract||'').substring(0,100))+'...</div>'+
    '<div class="ec-expected">'+badges+'</div></div>';
}

function renderAddForm(step){
  let h='<label>Title</label><input type="text" id="ex-title" placeholder="Article title">';
  h+='<label>Abstract</label><textarea id="ex-abstract" rows="2" placeholder="Article abstract" style="font-family:inherit;"></textarea>';
  if(step==='screener'){
    h+='<div class="form-row"><div><label>Status</label><select id="ex-status">';
    STATUS_OPTIONS.forEach(s=>{h+='<option value="'+s+'">'+s+'</option>';});
    h+='</select></div><div><label>Exclusion Code</label><select id="ex-code"><option value="">None</option>';
    EXCLUSION_CODES.forEach(c=>{h+='<option value="'+c+'">'+c+'</option>';});
    h+='</select></div></div>';
  } else if(step==='path'){
    h+='<label>Path</label><select id="ex-path">';
    PATH_OPTIONS.forEach(p=>{h+='<option value="'+p+'">'+p.replace(/_/g,' ')+'</option>';});
    h+='</select>';
  } else if(step==='cg'){
    h+='<label>CG Mechanisms</label><div class="checkbox-group">';
    CG_MECHANISMS.forEach(m=>{h+='<label><input type="checkbox" class="ex-cg-cb" value="'+m+'"> '+m.replace(/_/g,' ')+'</label>';});
    h+='</div>';
  } else if(step==='esg'){
    h+='<label>ESG Outcomes</label><div class="checkbox-group">';
    ESG_OUTCOMES.forEach(m=>{h+='<label><input type="checkbox" class="ex-esg-cb" value="'+m+'"> '+m.replace(/_/g,' ')+'</label>';});
    h+='</div>';
  } else if(step==='meta'){
    h+='<label>Meta Potential</label><select id="ex-meta">';
    META_OPTIONS.forEach(m=>{h+='<option value="'+m+'">'+m+'</option>';});
    h+='</select>';
  }
  h+='<div class="form-row" style="margin-top:0.5rem;"><div><label>Confidence (0.5\u20141.0)</label>';
  h+='<input type="number" id="ex-conf" value="0.85" min="0.5" max="1" step="0.05"></div>';
  h+='<div><label>Reasoning</label><input type="text" id="ex-reason" placeholder="Brief explanation"></div></div>';
  h+='<div class="btn-row"><button class="btn btn-green btn-sm" onclick="saveNewExample()">Save Example</button>';
  h+='<button class="btn btn-ghost btn-sm" onclick="toggleAddForm()">Cancel</button></div>';
  return h;
}

function saveNewExample(){
  const step=allPrompts[activeStep].step;
  const title=document.getElementById('ex-title').value;
  const abstract=document.getElementById('ex-abstract').value;
  if(!title||!abstract){alert('Title and Abstract are required');return;}
  const expected={};
  if(step==='screener'){
    expected.status=document.getElementById('ex-status').value;
    const code=document.getElementById('ex-code').value;
    if(code) expected.exclusion_code=code;
  } else if(step==='path'){
    expected.path=document.getElementById('ex-path').value;
  } else if(step==='cg'){
    expected.cg_mechanisms=[...document.querySelectorAll('.ex-cg-cb:checked')].map(cb=>cb.value);
  } else if(step==='esg'){
    expected.esg_outcomes=[...document.querySelectorAll('.ex-esg-cb:checked')].map(cb=>cb.value);
  } else if(step==='meta'){
    expected.meta_potential=document.getElementById('ex-meta').value;
  }
  const conf=parseFloat(document.getElementById('ex-conf').value);
  if(conf) expected.confidence=conf;
  const reason=document.getElementById('ex-reason').value;
  if(reason) expected.reasoning=reason;
  fetch('/api/examples/'+step,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({title,abstract,expected})
  }).then(r=>r.json()).then(d=>{
    if(d.ok){ showAddForm=false; loadPrompts(); showToast(); }
  });
}

function deleteExample(step,index){
  if(!confirm('Delete this example?')) return;
  fetch('/api/examples/'+step+'/'+index,{method:'DELETE'}).then(r=>r.json()).then(d=>{
    if(d.ok) loadPrompts();
  });
}

function saveStepAdvanced(){
  const p=allPrompts[activeStep];
  const content=document.getElementById('prompt-content').value;
  const examples=document.getElementById('prompt-examples').value;
  fetch('/api/prompts/'+p.step,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({content,examples})
  }).then(r=>r.json()).then(d=>{
    if(d.ok){
      allPrompts[activeStep].content=content;
      allPrompts[activeStep].examples_raw=examples;
      allPrompts[activeStep].num_examples=examples.trim()?examples.trim().split('\n').filter(l=>l.trim()).length:0;
      renderStepNav(); showToast();
    }
  });
}

function showToast(){
  const t=document.getElementById('save-toast');
  if(t){t.classList.add('show'); setTimeout(()=>t.classList.remove('show'),2000);}
}

// --- Results Table ---
function computeStats(){
  let inc=0,exc=0,bg=0,pathA=0,both=0,pathB=0;
  allLogs.forEach(l=>{
    const s=l.summary||{};
    if(s.status==='Include')inc++;
    if(s.status==='Exclude')exc++;
    if(s.status==='Background')bg++;
    if(s.path==='Path_A_CG_to_ESG')pathA++;
    if(s.path==='Both_A_and_B')both++;
    if(s.path==='Path_B_ESG_to_FP_with_CG_moderation')pathB++;
  });
  return {total:allLogs.length, inc, exc, bg, pathA, both, pathB, dedup: lastDedup?lastDedup.removed:0};
}

function renderFilterTabs(){
  const stats=computeStats();
  const container=document.getElementById('filter-tabs');
  const filters=[
    {key:'all', label:'All', count:stats.total},
    {key:'Include', label:'Include', count:stats.inc},
    {key:'Exclude', label:'Exclude', count:stats.exc},
    {key:'Background', label:'Background', count:stats.bg},
  ];
  container.innerHTML=filters.map(f=>
    '<div class="filter-tab '+(activeFilter===f.key?'active':'')+'" onclick="setFilter(\''+f.key+'\')">'+f.label+'<span class="ft-count">'+f.count+'</span></div>'
  ).join('');
}

function setFilter(f){
  activeFilter=f;
  renderResultsTable();
}

function renderResultsTable(){
  const body=document.getElementById('results-body');
  if(!allLogs.length){body.innerHTML='<tr><td colspan="10" style="text-align:center;color:var(--dim);padding:2rem;">No results yet. Run the pipeline first.</td></tr>';return;}

  const stats=computeStats();

  // Update summary cards
  const sc=document.getElementById('summary-cards');
  sc.classList.add('show');
  document.getElementById('sc-total').textContent=stats.total;
  document.getElementById('sc-dedup').textContent=stats.dedup;
  document.getElementById('sc-include').textContent=stats.inc;
  document.getElementById('sc-exclude').textContent=stats.exc;
  document.getElementById('sc-background').textContent=stats.bg;
  document.getElementById('sc-patha').textContent=stats.pathA;
  document.getElementById('sc-both').textContent=stats.both;
  document.getElementById('sc-pathb').textContent=stats.pathB;

  renderFilterTabs();

  // Filter logs
  const filtered=activeFilter==='all'?allLogs:allLogs.filter(l=>(l.summary||{}).status===activeFilter);
  document.getElementById('result-count').textContent='('+filtered.length+' of '+allLogs.length+' articles)';

  body.innerHTML=filtered.map((l,idx)=>{
    const s=l.summary||{};
    const bc=s.status==='Include'?'b-include':s.status==='Exclude'?'b-exclude':
             s.status==='Background'?'b-background':'b-error';
    const mc=s.meta==='High'?'b-high':s.meta==='Medium'?'b-medium':'b-low';
    const pathShort=(s.path||'').replace('Path_A_CG_to_ESG','Path A').replace('Both_A_and_B','Both A+B').replace('Path_B_ESG_to_FP_with_CG_moderation','Path B');
    const cgShort=(s.cg||'').replace(/;/g,'; ').replace(/_/g,' ');
    const esgShort=(s.esg||'').replace(/;/g,'; ').replace(/_/g,' ');
    const conf=typeof s.confidence==='number'?(s.confidence*100).toFixed(0)+'%':'';
    const realIdx=allLogs.indexOf(l);

    return '<tr>'+
      '<td>'+l.i+'</td>'+
      '<td title="'+escAttr(l.title)+'" style="max-width:250px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">'+escHtml(l.title||'')+'</td>'+
      '<td><span class="badge '+bc+'">'+(s.status||'?')+'</span></td>'+
      '<td style="color:var(--dim);font-size:0.7rem;">'+(s.exclusion_code||'')+'</td>'+
      '<td>'+(pathShort||'—')+'</td>'+
      '<td style="max-width:150px;overflow:hidden;text-overflow:ellipsis;" title="'+escAttr(cgShort)+'">'+(cgShort||'—')+'</td>'+
      '<td style="max-width:150px;overflow:hidden;text-overflow:ellipsis;" title="'+escAttr(esgShort)+'">'+(esgShort||'—')+'</td>'+
      '<td><span class="badge '+mc+'">'+(s.meta||'—')+'</span></td>'+
      '<td>'+conf+'</td>'+
      '<td><span class="expand-btn" onclick="toggleDetail('+realIdx+')">show</span></td>'+
    '</tr>'+
    '<tr class="detail-row" id="detail-'+realIdx+'" style="display:none;">'+
      '<td colspan="10">'+
        '<div class="detail-inner">'+
          Object.entries(l.step_results||{}).map(function(entry){
            var k=entry[0],v=entry[1];
            return '<div class="detail-box"><h4>'+escHtml(k)+' <span class="expand-btn" onclick="quickCorrect('+realIdx+',\''+k+'\')" style="float:right;font-size:0.65rem;">Correct</span></h4><pre>'+escHtml(typeof v==='object'?JSON.stringify(v,null,2):String(v))+'</pre><div id="qc-'+realIdx+'-'+k+'"></div></div>';
          }).join('')+
          (s.reasoning?'<div class="detail-box"><h4>Screening Reasoning</h4><pre>'+escHtml(s.reasoning)+'</pre></div>':'')+
        '</div>'+
      '</td>'+
    '</tr>';
  }).join('');
}

function toggleDetail(idx){
  const row=document.getElementById('detail-'+idx);
  row.style.display=row.style.display==='none'?'table-row':'none';
}

// --- Quick Correct ---
function quickCorrect(logIdx,stepName){
  const container=document.getElementById('qc-'+logIdx+'-'+stepName);
  if(container.innerHTML){container.innerHTML='';return;}
  const log=allLogs[logIdx]; const s=log.summary||{};
  let h='<div class="qc-form">';
  if(stepName==='screener'){
    h+='<select id="qcs-'+logIdx+'">';
    STATUS_OPTIONS.forEach(v=>{h+='<option value="'+v+'"'+(v===s.status?' selected':'')+'>'+v+'</option>';});
    h+='</select> <select id="qcc-'+logIdx+'"><option value="">No code</option>';
    EXCLUSION_CODES.forEach(v=>{h+='<option value="'+v+'"'+(v===(s.exclusion_code||'')?' selected':'')+'>'+v+'</option>';});
    h+='</select>';
  } else if(stepName==='path'){
    h+='<select id="qcp-'+logIdx+'">';
    PATH_OPTIONS.forEach(v=>{h+='<option value="'+v+'"'+(v===s.path?' selected':'')+'>'+v.replace(/_/g,' ')+'</option>';});
    h+='</select>';
  } else if(stepName==='cg'){
    const cur=(s.cg||'').split(';').filter(x=>x);
    CG_MECHANISMS.forEach(m=>{
      h+='<label><input type="checkbox" class="qccg'+logIdx+'" value="'+m+'"'+(cur.includes(m)?' checked':'')+'>'+m.replace(/_/g,' ')+'</label>';
    });
  } else if(stepName==='esg'){
    const cur=(s.esg||'').split(';').filter(x=>x);
    ESG_OUTCOMES.forEach(m=>{
      h+='<label><input type="checkbox" class="qcesg'+logIdx+'" value="'+m+'"'+(cur.includes(m)?' checked':'')+'>'+m.replace(/_/g,' ')+'</label>';
    });
  } else if(stepName==='meta'){
    h+='<select id="qcm-'+logIdx+'">';
    META_OPTIONS.forEach(v=>{h+='<option value="'+v+'"'+(v===s.meta?' selected':'')+'>'+v+'</option>';});
    h+='</select>';
  }
  h+='<div style="margin-top:0.3rem;"><button class="btn btn-green btn-sm" onclick="saveQC('+logIdx+',\''+stepName+'\')">Save as Example</button> ';
  h+='<button class="btn btn-ghost btn-sm" onclick="document.getElementById(\'qc-'+logIdx+'-'+stepName+'\').innerHTML=\'\'">Cancel</button></div></div>';
  container.innerHTML=h;
}

function saveQC(logIdx,stepName){
  const log=allLogs[logIdx];
  const title=log.title||'';
  const abstract=log.abstract||'';
  const expected={confidence:0.90,reasoning:'Corrected from pipeline results'};
  if(stepName==='screener'){
    expected.status=document.getElementById('qcs-'+logIdx).value;
    const code=document.getElementById('qcc-'+logIdx).value;
    if(code) expected.exclusion_code=code;
  } else if(stepName==='path'){
    expected.path=document.getElementById('qcp-'+logIdx).value;
  } else if(stepName==='cg'){
    expected.cg_mechanisms=[...document.querySelectorAll('.qccg'+logIdx+':checked')].map(cb=>cb.value);
  } else if(stepName==='esg'){
    expected.esg_outcomes=[...document.querySelectorAll('.qcesg'+logIdx+':checked')].map(cb=>cb.value);
  } else if(stepName==='meta'){
    expected.meta_potential=document.getElementById('qcm-'+logIdx).value;
  }
  fetch('/api/examples/'+stepName,{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({title,abstract,expected})
  }).then(r=>r.json()).then(d=>{
    if(d.ok){
      document.getElementById('qc-'+logIdx+'-'+stepName).innerHTML='<span class="qc-saved">Saved as example!</span>';
    }
  });
}

// --- Init ---
document.addEventListener('DOMContentLoaded',function(){
  updateModelDropdown();
  loadSavedKey();
});

// --- Util ---
function escHtml(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
function escAttr(s){return (s||'').replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');}
</script>
</body>
</html>"""

if __name__ == "__main__":
    print("\n  SLR Brain v3 — http://localhost:8080\n")
    app.run(host="0.0.0.0", port=8080, debug=False)
