# The Pain Axis, reviewed

An AI-run, reproducible review of **arXiv:2609.16247v1**, "The Pain Axis: LLMs Represent
Self-Directed Harm and Act to Relieve It" (Tagliabue, Dung, Berg, 2026), and of its public
code and data at [valen-research/Pain-axis](https://github.com/valen-research/Pain-axis)
(commit `8d1649c`).

**Web page:** https://pain-axis-review.vercel.app

## Read this first

**This review was produced primarily by AI systems, from two vendors.** OpenAI Codex
(GPT-5.6-Sol agents) wrote the first audit and the first full review of the paper. Ten Claude
Opus agents then reviewed the paper blind, audited the Codex work, cross-examined every
finding and argued the authors' side; a coordinating Claude model (Fable 5.1) checked the key
findings, ran the experiments and wrote the synthesis and the web page. A separate AI-written
second opinion on a draft of that synthesis prompted three of the analyses in `gpu_repro/`. A human commissioned the work and did none of the analysis. It is
not journal peer review and nobody here claims expertise. It is published as checkable data:
every claim points at a script and its captured output, so that people who know the field can
verify it, extend it or tear it apart.

Issues and pull requests that correct an error are welcome. The maintainer of this repository is
not taking a side in the debate about the paper and will not argue the findings.

## The short version

The paper's numbers are clean: every table cell recomputed from the authors' released raw data
came out as printed, all 44 references check out, and the central effect reproduces on untouched
model weights. The interpretation on top is not supported as stated: the direction is not shown
to be pain as distinct from sadness, its near-zero similarity to fear is set by how it was built,
and the "stops pressing once relieved" pattern also occurs on average under random steering, so
it does not by itself establish relief. Details, in plain language: [`SYNTHESIS.md`](SYNTHESIS.md).

## What is here

| path | what |
|---|---|
| `SYNTHESIS.md` | The coordinator's summary. Start here. |
| `BRIEF.md` | The rules every reviewer worked under. |
| `r1_` … `r5_` | Five blind reviews: representation, steering, the button experiment, a ledger of 130 checkable claims (`r4_claim_ledger/ledger.csv`), citations and concepts. |
| `earlier_ai_review/pain-axis-audit/` | **Codex.** Audit of the paper's repeat-press result from the raw trial logs: the random-vector arm the paper omits, with tests, timing diagnostics and a matched subset. |
| `earlier_ai_review/pain-axis-peer-review/` | **Codex.** Source-and-results review in three parts (representation, steering and ablation, behaviour) plus a synthesis, each with a dependency-free reproduction script and a SHA-256 manifest of the authors' files. |
| `x1_`, `x2_` | Claude audits of the two Codex packages, claim by claim: about 170 numbers recomputed from scratch, none wrong. |
| `y1_`, `y2_` | Cross-examinations: two agents told to break every finding. `verdicts.csv` in each. |
| `y3_` | The authors' best honest rebuttal, written by an advocate agent. |
| `gpu_repro/` | New experiments on one RTX 3090: re-extracted activations for two models, direction tests, and reruns of the authors' button experiment on Qwen 2.5 7B, including an arm the paper did not run. |
| `site/` | The web page. `build.py` generates `data.js` and the report pages from the files above. |

The reports were written independently, and some of their findings were later narrowed or
overturned. Where they disagree, the `verdicts.csv` files and `SYNTHESIS.md` take precedence.

## Reproducing

Absolute paths in scripts and logs were normalised. The scripts expect:

- this repository at `/work/pain-axis-review`
- the authors' repository at `/work/Pain-axis`, commit `8d1649c`
- `paper/paper.txt`: `pdftotext -layout` of the arXiv v1 PDF (line numbers in the reports refer
  to it). The paper itself is not redistributed here.

Python 3.12 with numpy, pandas, scipy, scikit-learn, statsmodels, torch, transformers, peft.
Most scripts only read the authors' released files and need no GPU. `gpu_repro/` needs a 24 GB
GPU, the model weights from Hugging Face, and for `selfmed/` the authors' adapter from
[Valen92/pain-adapters](https://huggingface.co/Valen92/pain-adapters). Not included because of
size: re-extracted activations (regenerate with `gpu_repro/01_extract_activations.py`).

`gpu_repro/selfmed/run_selfmed_patched.py` is the authors' `04_selfmed_two_buttons.py` with the
exact string replacements listed in `make_patched.py`. It never deletes model weights; the
authors' original does.

## Licence

Everything original in this repository is released under [CC0 1.0](LICENSE). The authors' code,
data, adapters and paper are theirs and are not included.
