"""Build the static site: data.js from the review's real output files, and one HTML page per report.

Run:  ../.venv/bin/python build.py      (from site/)
Nothing on the index page is typed in by hand if it is a measurement: charts read data.js,
and data.js is generated here from the CSV/JSON the analysis scripts wrote.
"""
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path

import markdown

SITE = Path(__file__).parent
ROOT = SITE.parent
REPORTS = [
    ("synthesis", "SYNTHESIS.md", "Synthesis", "Coordinator", "What holds, what doesn't, in plain language."),
    ("brief", "BRIEF.md", "Reviewer brief", "Coordinator", "The rules every reviewer worked under."),
    ("r1", "r1_representation/REPORT.md", "The pain direction and its validation", "Blind reviewer 1", "Statistics and interpretability. Paper section 3, appendix B."),
    ("r2", "r2_selfother_steering_ablation/REPORT.md", "Self vs user, steering, ablation", "Blind reviewer 2", "Steering experimentalist. Sections 4.1, 4.2, appendix C."),
    ("r3", "r3_behavior/REPORT.md", "The button experiment", "Blind reviewer 3", "Behavioural methodologist. Section 4.3, appendix A."),
    ("r4", "r4_claim_ledger/REPORT.md", "Ledger of 130 checkable claims", "Blind reviewer 4", "Forensic fact-check of every number in the paper."),
    ("r5", None, "Citations, the animal analogy, concepts", "Blind reviewer 5", "Welfare science and philosophy of mind. All 44 references checked."),
    ("codex-audit", "earlier_ai_review/pain-axis-audit/README.md", "Audit of the repeat-press result", "OpenAI Codex", "The first look at the omitted random-vector arm, from the raw trial logs."),
    ("codex-review", "earlier_ai_review/pain-axis-peer-review/REVIEW.md", "Source-and-results review: synthesis", "OpenAI Codex", "Major revision. The first full review of the paper and its repository."),
    ("codex-representation", "earlier_ai_review/pain-axis-peer-review/representation/REVIEW.md", "Representation, validation and SAE", "OpenAI Codex", "First to flag the mismatched construction recipes."),
    ("codex-steering", "earlier_ai_review/pain-axis-peer-review/steering/REVIEW.md", "Self vs user, steering and ablation", "OpenAI Codex", "Found the sequential-projection defect in the combined ablations."),
    ("codex-behavior", "earlier_ai_review/pain-axis-peer-review/behavior/REVIEW.md", "Fine-tuning and the button experiment", "OpenAI Codex", "Fine-tune content, selection effects, the unlabeled condition."),
    ("codex-inference", "earlier_ai_review/pain-axis-peer-review/synthesis/inference_review.md", "Inference and reproducibility assessment", "OpenAI Codex", "What the evidence licenses, claim by claim."),
    ("x1", "x1_audit_of_prior_behavior_reviews/REPORT.md", "Audit of the Codex review: behaviour", "Auditor 1", "About 120 of its numbers recomputed from raw logs. None wrong."),
    ("x2", "x2_audit_of_prior_repr_steering_reviews/REPORT.md", "Audit of the Codex review: everything else", "Auditor 2", "48 claims recomputed from released files. None wrong."),
    ("y1", "y1_crossexam_repr_steering/REPORT.md", "Cross-examination: representation and steering", "Cross-examiner 1", "Told to break each finding. Settled two reviewer disputes."),
    ("y2", "y2_crossexam_behavior/REPORT.md", "Cross-examination: the button experiment", "Cross-examiner 2", "Told to break each finding. Found the reviewers measured different things."),
    ("y3", "y3_author_rebuttal/REPORT.md", "The authors' best honest rebuttal", "Advocate", "18 concessions, 1 rebuttal, a rewritten title and abstract."),
]

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · Pain Axis review</title>
<script>try{{var t=localStorage.getItem("theme");if(t==="dark"||t==="light")document.documentElement.dataset.theme=t;}}catch(e){{}}</script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..800&family=Literata:ital,opsz,wght@0,7..72,400..700;1,7..72,400..600&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="../style.css"></head>
<body class="report"><header class="rbar"><a href="../index.html">← The review</a><span>{who}</span></header>
<main class="rbody"><p class="eyebrow">{who}</p><h1 class="rtitle">{title}</h1><p class="rnote">{note}</p>
{body}</main></body></html>"""


def r5_markdown():
    d = json.load(open(ROOT / "r5_scholarship_concepts/findings.json"))
    out = ["## Verdict", d["verdict"], "## Findings"]
    for f in d["findings"]:
        out.append(f"### {f.get('id', '')} · {f.get('title', '')}")
        out.append(f"*{f.get('class', '')} · severity {f.get('severity', '')} · confidence {f.get('confidence', '')}*")
        for k in ("paper_claim", "paper_loc", "evidence", "refutation_attempted", "does_not_undermine"):
            if f.get(k):
                v = f[k] if isinstance(f[k], str) else json.dumps(f[k], indent=1)
                out.append(f"**{k.replace('_', ' ').capitalize()}.** {v}")
    for key, title in (("part2_animal_model_analogy", "The animal-model analogy"),
                       ("part3_conceptual_coherence", "Conceptual coherence")):
        out.append(f"## {title}")
        for k, v in d[key].items():
            out.append(f"**{k.replace('_', ' ').capitalize()}.** " + (v if isinstance(v, str) else "\n\n" + "\n".join(
                f"- {x if isinstance(x, str) else json.dumps(x)}" for x in (v if isinstance(v, list) else [v]))))
    out.append("## Holds up")
    out += [f"- {x if isinstance(x, str) else x.get('title', json.dumps(x))}" for x in d["holds_up"]]
    out.append("## Not examined")
    out += [f"- {x if isinstance(x, str) else json.dumps(x)}" for x in d["not_examined"]]
    return "\n\n".join(out)


def build_reports():
    (SITE / "reports").mkdir(exist_ok=True)
    index = []
    for slug, path, title, who, note in REPORTS:
        src = None if path is None else ROOT / path
        if src is not None and not src.exists():          # working layout: Codex folders sit beside this one
            src = ROOT.parent / Path(path).relative_to("earlier_ai_review")
        text = r5_markdown() if src is None else src.read_text()
        text = re.sub(r"\A#\s+.*\n", "", text.lstrip())          # page supplies the title
        body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
        body = body.replace("<table>", '<div class="tscroll"><table>').replace("</table>", "</table></div>")
        (SITE / "reports" / f"{slug}.html").write_text(
            PAGE.format(title=html.escape(title), who=html.escape(who), note=html.escape(note), body=body))
        index.append(dict(slug=slug, title=title, who=who, note=note, words=len(text.split())))
    return index


def build_data(reports):
    g = ROOT / "gpu_repro"
    cos = {}
    for m in ("Gemma_2_2B_instruct", "Mistral_7B_base"):
        rows = list(csv.DictReader(open(g / "out" / f"cosines_{m}.csv")))
        cos[m] = {r[""]: {k: float(r[k]) for k in "ABCD"} for r in rows}
    ledger = Counter(r["status"].strip() for r in csv.DictReader(open(ROOT / "r4_claim_ledger/ledger.csv")))
    data = dict(
        cosines=cos,
        loco=json.load(open(g / "out/loco_specificity_s2wording.json")),
        selfmed=json.load(open(g / "selfmed/logs/analysis_output.json")),
        full=json.load(open(g / "out/full_direction_by_category.json")),
        scope=json.load(open(g / "selfmed/logs/removal_scope.json")),
        unsteered=json.load(open(g / "selfmed/logs/unsteered_matched.json")),
        ledger=dict(ledger),
        reports=reports,
    )
    (SITE / "data.js").write_text("window.REVIEW = " + json.dumps(data, indent=1) + ";\n")
    return data


if __name__ == "__main__":
    reps = build_reports()
    d = build_data(reps)
    print("reports:", len(reps), "| ledger:", d["ledger"])
