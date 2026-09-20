# Shared brief for all reviewers

## What this is

An independent review of arXiv:2609.16247v1, "The Pain Axis: LLMs Represent Self-Directed Harm
and Act to Relieve It" (Tagliabue, Dung, Berg; submitted 2026-09-14), and of its public repository.

The paper went viral. The person who commissioned this review suspects it is not necessarily weak,
but may contain enough errors to damage the discourse if it is first celebrated and then falls
apart. They no longer trust anyone's verdict on it, human or AI, including earlier AI reviews.
Your job is to get it right, not to debunk and not to defend. A finding that the paper is correct
on a point is as valuable as a finding that it is wrong. An inflated or sloppy criticism is a
failure of this review, exactly as a missed error is.

## Paths

- Paper text (pdftotext, line-numbered when you Read it): `/work/pain-axis-review/paper/paper.txt`
- Paper PDF (30 pages; Read with the `pages` parameter): `/work/pain-axis-review/paper/paper.pdf`
- Paper HTML: `/work/pain-axis-review/paper/paper.html`
- Figures and the three Appendix A result tables as PNG (the tables exist ONLY as images; Read them):
  `/work/pain-axis-review/paper/images/` (`tab_selfmed_7B.png`, `tab_selfmed_32B.png`, `tab_selfmed_72B.png`, `fig_*.png`)
- Authors' repository, cloned at commit 8d1649c: `/work/Pain-axis/`
  (`datasets/`, `scripts/`, `results/` ~226 MB; folders follow paper section numbers; read its README first)
- Python with numpy, pandas, scipy, statsmodels, scikit-learn, torch (CPU use only):
  `/work/pain-axis-review/.venv/bin/python`
- Your output directory is given in your individual task. Write only there.

## Hard rules

- The repository clone is someone else's work and is evidence. Never modify, move, or delete
  anything in `/work/Pain-axis/`. Never run the authors' GPU scripts (several of
  them delete the entire Hugging Face cache). Reading them is the point; executing them is not.
- No model inference, no fine-tuning, no paid API calls, no downloading model weights.
- Do not write outside your output directory. Do not touch `/work/earlier-ai-review/audit/`
  or `/work/earlier-ai-review/peer-review/`.
- Do not contact anyone, post anything, or open issues/PRs.
- Do not spawn further subagents.

## Standard of evidence

Every finding must be checkable by a sceptical reader without trusting you:

- Quote the paper claim and give its line number in `paper.txt` (or the table image and cell).
- Give the evidence: `file:line` in the repo, or a script you wrote plus its actual output.
  Recompute from the released raw files wherever they exist, rather than trusting the authors'
  summary CSVs. If a number cannot be recomputed from what was released, say so; that is a finding
  about verifiability, not about correctness.
- Before writing a finding down, try to refute it. Read the surrounding code, check whether the
  paper already discloses it (Limitations, footnotes, appendix), check whether you misread a
  denominator or a filter. State what you did to try to break it.
- Classify each finding as one of:
  `factual-error` (paper states X, released data/code shows Y),
  `code-bug` (implementation does not do what the paper says or what is mathematically intended),
  `method-weakness` (design cannot support the inference drawn),
  `overclaim` (wording stronger than the evidence, evidence itself fine),
  `unverifiable` (claim has no released artifact behind it),
  `holds-up` (you tried to break it and it reproduces).
- Give severity (`fatal` = invalidates a headline claim, `major`, `minor`, `cosmetic`) and your
  confidence (`high`/`medium`/`low`). Say explicitly what the finding does NOT undermine.
- Never state a number you did not compute or read. Never describe a file you did not open.
  If you ran out of time on something, list it under "not examined".

## Deliverables

In your output directory:

1. `REPORT.md`: your review. Lead with a verdict paragraph, then findings ordered by severity,
   then what holds up, then "not examined".
2. `findings.json`: a list of objects with keys `id`, `title`, `class`, `severity`, `confidence`,
   `paper_claim`, `paper_loc`, `evidence`, `refutation_attempted`, `does_not_undermine`.
3. Any scripts you wrote, runnable with the venv Python above, and their captured output.

Your final reply to the coordinator: at most 400 words. Verdict, the five findings that matter
most (with severity and confidence), and anything you could not resolve.
