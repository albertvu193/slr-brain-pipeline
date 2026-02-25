#!/usr/bin/env python3
"""
SLR Brain — Evaluation Script

Runs the classification pipeline against gold-standard test data and reports
accuracy per step, confusion matrices, and per-article details.

Usage:
    python eval.py --provider openai --api-key sk-... [--model gpt-4o-mini] [--test-file eval_testset.jsonl]

The test set is a JSONL file where each line is:
{
  "title": "...",
  "abstract": "...",
  "keywords": "...",      (optional)
  "expected": {
    "status": "Include",
    "exclusion_code": null,
    "path": "Path_A_CG_to_ESG",
    "cg_mechanisms": ["Board_Structure_Composition"],
    "esg_outcomes": ["ESG_Disclosure_Reporting"],
    "meta_potential": "High"
  }
}

Fields in "expected" are optional — only present fields are evaluated.
"""

import argparse, json, sys, time
from pathlib import Path
from collections import defaultdict

# Import pipeline components from app.py
sys.path.insert(0, str(Path(__file__).parent))
from app import (
    PROVIDERS, STEPS, load_prompt, load_examples,
    extract_json, call_with_retry
)

STEP_MAP = {
    "screener": ("01_screener.md", "screener", "status"),
    "path": ("02_path_classifier.md", "path", "path"),
    "cg": ("03_cg_tagger.md", "cg", "cg_mechanisms"),
    "esg": ("04_esg_tagger.md", "esg", "esg_outcomes"),
    "meta": ("05_meta_scorer.md", "meta", "meta_potential"),
}

def classify_for_eval(title, abstract, keywords, provider_name, api_key, model):
    """Run the full pipeline on a single article, return parsed results per step."""
    provider = PROVIDERS[provider_name]
    caller = provider["caller"]
    article_text = f"Title: {title}\nKeywords: {keywords}\nAbstract: {abstract[:2000]}"
    results = {}

    for prompt_file, step_name, output_key, label, desc in STEPS:
        # Skip if excluded
        if step_name != "screener" and results.get("status") in ("Exclude", "Background"):
            results[f"{step_name}_skipped"] = True
            continue

        system_msg = load_prompt(prompt_file) + load_examples(step_name)
        try:
            raw = call_with_retry(caller, article_text, system_msg, model, api_key)
            parsed = extract_json(raw)

            if step_name == "screener":
                results["status"] = parsed.get("status", "")
                results["exclusion_code"] = parsed.get("exclusion_code")
            elif step_name == "path":
                results["path"] = parsed.get("path", "")
            elif step_name == "cg":
                results["cg_mechanisms"] = parsed.get("cg_mechanisms", [])
            elif step_name == "esg":
                results["esg_outcomes"] = parsed.get("esg_outcomes", [])
            elif step_name == "meta":
                results["meta_potential"] = parsed.get("meta_potential", "")

            results[f"{step_name}_confidence"] = parsed.get("confidence", 0)
            results[f"{step_name}_reasoning"] = parsed.get("reasoning", "")

        except Exception as e:
            results[f"{step_name}_error"] = str(e)

    return results


def compare_sets(expected, actual):
    """Compare two lists as sets. Returns (correct, missing, extra)."""
    exp_set = set(expected) if isinstance(expected, list) else {expected}
    act_set = set(actual) if isinstance(actual, list) else {actual}
    correct = exp_set & act_set
    missing = exp_set - act_set
    extra = act_set - exp_set
    return correct, missing, extra


def evaluate(test_file, provider_name, api_key, model, verbose=True):
    """Run evaluation and return metrics."""
    test_path = Path(test_file)
    if not test_path.exists():
        print(f"Error: Test file '{test_file}' not found.")
        sys.exit(1)

    test_cases = []
    with open(test_path) as f:
        for line in f:
            line = line.strip()
            if line:
                test_cases.append(json.loads(line))

    print(f"\n{'='*70}")
    print(f"  SLR Brain Evaluation — {len(test_cases)} test cases")
    print(f"  Provider: {provider_name} | Model: {model}")
    print(f"{'='*70}\n")

    # Track metrics per step
    step_metrics = {
        "screener": {"correct": 0, "total": 0, "confusion": defaultdict(lambda: defaultdict(int))},
        "path": {"correct": 0, "total": 0, "confusion": defaultdict(lambda: defaultdict(int))},
        "cg": {"precision_sum": 0, "recall_sum": 0, "total": 0},
        "esg": {"precision_sum": 0, "recall_sum": 0, "total": 0},
        "meta": {"correct": 0, "total": 0, "confusion": defaultdict(lambda: defaultdict(int))},
    }
    errors = []
    details = []

    for i, tc in enumerate(test_cases):
        title = tc.get("title", "")
        abstract = tc.get("abstract", "")
        keywords = tc.get("keywords", "")
        expected = tc.get("expected", {})

        title_short = title[:60] + ("..." if len(title) > 60 else "")
        print(f"  [{i+1}/{len(test_cases)}] {title_short}")

        try:
            actual = classify_for_eval(title, abstract, keywords, provider_name, api_key, model)
        except Exception as e:
            print(f"    ERROR: {e}")
            errors.append({"index": i+1, "title": title_short, "error": str(e)})
            continue

        detail = {"index": i+1, "title": title_short, "results": {}}

        # --- Evaluate each step ---

        # Step 1: Screener (exact match on status)
        if "status" in expected:
            exp_status = expected["status"]
            act_status = actual.get("status", "")
            match = exp_status == act_status
            step_metrics["screener"]["total"] += 1
            if match:
                step_metrics["screener"]["correct"] += 1
            step_metrics["screener"]["confusion"][exp_status][act_status] += 1

            icon = "OK" if match else "MISS"
            detail["results"]["screener"] = {"expected": exp_status, "actual": act_status, "match": match}
            if verbose:
                ex_code = actual.get("exclusion_code", "")
                print(f"    Screen:  {icon}  expected={exp_status}  actual={act_status}  {ex_code}")

        # Step 2: Path (exact match, only for Included)
        if "path" in expected and actual.get("status") == "Include":
            exp_path = expected["path"]
            act_path = actual.get("path", "")
            match = exp_path == act_path
            step_metrics["path"]["total"] += 1
            if match:
                step_metrics["path"]["correct"] += 1
            step_metrics["path"]["confusion"][exp_path][act_path] += 1

            icon = "OK" if match else "MISS"
            detail["results"]["path"] = {"expected": exp_path, "actual": act_path, "match": match}
            if verbose:
                print(f"    Path:    {icon}  expected={exp_path}  actual={act_path}")

        # Step 3: CG Tags (set comparison)
        if "cg_mechanisms" in expected and actual.get("status") == "Include":
            exp_cg = expected["cg_mechanisms"]
            act_cg = actual.get("cg_mechanisms", [])
            correct, missing, extra = compare_sets(exp_cg, act_cg)
            precision = len(correct) / len(set(act_cg)) if act_cg else (1.0 if not exp_cg else 0.0)
            recall = len(correct) / len(set(exp_cg)) if exp_cg else 1.0
            step_metrics["cg"]["precision_sum"] += precision
            step_metrics["cg"]["recall_sum"] += recall
            step_metrics["cg"]["total"] += 1

            match = not missing and not extra
            icon = "OK" if match else "MISS"
            detail["results"]["cg"] = {"expected": exp_cg, "actual": act_cg, "missing": list(missing), "extra": list(extra)}
            if verbose:
                print(f"    CG Tags: {icon}  expected={exp_cg}  actual={act_cg}")
                if missing: print(f"             Missing: {list(missing)}")
                if extra: print(f"             Extra:   {list(extra)}")

        # Step 4: ESG Tags (set comparison)
        if "esg_outcomes" in expected and actual.get("status") == "Include":
            exp_esg = expected["esg_outcomes"]
            act_esg = actual.get("esg_outcomes", [])
            correct, missing, extra = compare_sets(exp_esg, act_esg)
            precision = len(correct) / len(set(act_esg)) if act_esg else (1.0 if not exp_esg else 0.0)
            recall = len(correct) / len(set(exp_esg)) if exp_esg else 1.0
            step_metrics["esg"]["precision_sum"] += precision
            step_metrics["esg"]["recall_sum"] += recall
            step_metrics["esg"]["total"] += 1

            match = not missing and not extra
            icon = "OK" if match else "MISS"
            detail["results"]["esg"] = {"expected": exp_esg, "actual": act_esg, "missing": list(missing), "extra": list(extra)}
            if verbose:
                print(f"    ESG Tags:{icon}  expected={exp_esg}  actual={act_esg}")
                if missing: print(f"             Missing: {list(missing)}")
                if extra: print(f"             Extra:   {list(extra)}")

        # Step 5: Meta Score (exact match)
        if "meta_potential" in expected and actual.get("status") == "Include":
            exp_meta = expected["meta_potential"]
            act_meta = actual.get("meta_potential", "")
            match = exp_meta == act_meta
            step_metrics["meta"]["total"] += 1
            if match:
                step_metrics["meta"]["correct"] += 1
            step_metrics["meta"]["confusion"][exp_meta][act_meta] += 1

            icon = "OK" if match else "MISS"
            detail["results"]["meta"] = {"expected": exp_meta, "actual": act_meta, "match": match}
            if verbose:
                print(f"    Meta:    {icon}  expected={exp_meta}  actual={act_meta}")

        details.append(detail)
        print()

    # --- Print Summary ---
    print(f"\n{'='*70}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*70}\n")

    # Screener accuracy
    sm = step_metrics["screener"]
    if sm["total"] > 0:
        acc = sm["correct"] / sm["total"] * 100
        print(f"  Step 1 — Screen:     {sm['correct']}/{sm['total']}  ({acc:.1f}% accuracy)")
        if sm["total"] - sm["correct"] > 0:
            print(f"    Confusion matrix:")
            all_labels = sorted(set(list(sm["confusion"].keys()) + [l for d in sm["confusion"].values() for l in d.keys()]))
            header = "    " + " " * 14 + "  ".join(f"{l:>10}" for l in all_labels) + "  (actual)"
            print(header)
            for exp_label in all_labels:
                row = f"    {exp_label:>12}:"
                for act_label in all_labels:
                    count = sm["confusion"][exp_label][act_label]
                    row += f"  {count:>10}"
                print(row)
            print()

    # Path accuracy
    pm = step_metrics["path"]
    if pm["total"] > 0:
        acc = pm["correct"] / pm["total"] * 100
        print(f"  Step 2 — Path:       {pm['correct']}/{pm['total']}  ({acc:.1f}% accuracy)")

    # CG precision/recall
    cm = step_metrics["cg"]
    if cm["total"] > 0:
        avg_p = cm["precision_sum"] / cm["total"] * 100
        avg_r = cm["recall_sum"] / cm["total"] * 100
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
        print(f"  Step 3 — CG Tags:    Precision={avg_p:.1f}%  Recall={avg_r:.1f}%  F1={f1:.1f}%  (n={cm['total']})")

    # ESG precision/recall
    em = step_metrics["esg"]
    if em["total"] > 0:
        avg_p = em["precision_sum"] / em["total"] * 100
        avg_r = em["recall_sum"] / em["total"] * 100
        f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0
        print(f"  Step 4 — ESG Tags:   Precision={avg_p:.1f}%  Recall={avg_r:.1f}%  F1={f1:.1f}%  (n={em['total']})")

    # Meta accuracy
    mm = step_metrics["meta"]
    if mm["total"] > 0:
        acc = mm["correct"] / mm["total"] * 100
        print(f"  Step 5 — Meta Score: {mm['correct']}/{mm['total']}  ({acc:.1f}% accuracy)")

    if errors:
        print(f"\n  Errors: {len(errors)} articles failed to process")
        for e in errors:
            print(f"    [{e['index']}] {e['title']}: {e['error']}")

    print(f"\n{'='*70}\n")

    # Save detailed results
    output_path = Path(__file__).parent / "output" / "eval_results.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"metrics": {k: {kk: vv for kk, vv in v.items() if kk != "confusion"} for k, v in step_metrics.items()},
                    "details": details, "errors": errors}, f, indent=2, default=str)
    print(f"  Detailed results saved to: {output_path}\n")

    return step_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLR Brain Evaluation Script")
    parser.add_argument("--provider", default="openai", choices=["openai", "anthropic"])
    parser.add_argument("--api-key", required=True, help="API key for the provider")
    parser.add_argument("--model", default=None, help="Model to use (defaults to provider default)")
    parser.add_argument("--test-file", default="eval_testset.jsonl", help="Path to test JSONL file")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-article output")
    args = parser.parse_args()

    model = args.model or PROVIDERS[args.provider]["default_model"]
    evaluate(args.test_file, args.provider, args.api_key, model, verbose=not args.quiet)
