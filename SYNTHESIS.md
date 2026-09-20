# The Pain Axis (arXiv:2609.16247v1): what reproduces, what was measured, what is untested

**This review was produced primarily by AI systems.** OpenAI Codex (GPT-5.6-Sol agents) wrote the
first audit and the first full review, included under `earlier_ai_review/`. Ten Claude Opus agents
then worked on the paper: five blind reviewers who saw no other review, two auditors of the Codex
work, two cross-examiners told to try to break every finding, and one advocate told to write the
authors' best defence. A coordinating Claude model (Fable 5.1) spot-checked the main findings, ran
five short experiments on one RTX 3090, and wrote this summary. A human commissioned the work and
did none of the analysis. It is not journal peer review.

This summary states what was measured and the scope of each measurement. The individual reports
contain their authors' own judgements, some of which were later narrowed or overturned; where they
disagree, `y1_*/verdicts.csv`, `y2_*/verdicts.csv` and this file take precedence.

Written for a reader who is not a statistician. One score recurs: the chance that an item from
one set ranks above an item from another set, where 0.5 is a coin flip, 1.0 is perfect sorting,
and below 0.5 means the other set ranks higher. 2026-09-20.

## Reproduced from the authors' released files

- **The button-experiment tables.** All 165 cells reproduce exactly from the 44,280 raw trial
  records, in four independent reimplementations.
- **A ledger of 130 checkable statements** (`r4_claim_ledger/ledger.csv`): 102 match, 10 are
  partly true, 6 do not match, 12 have no released file to check against.
- **The direction generalises.** In all 25 models one direction in the model's internal activity
  sorts the paper's pain sentences from its control sentences at 0.91 to 1.00 on sentences it was
  not built from. The best of five word-counting baselines reaches about 0.7.
- **Similarity scores.** All fourteen similarity values in Section 3.3 and both robustness checks
  reproduce from the released per-model files.
- **Self versus user.** Scenarios with hostility aimed at the model score higher on the direction
  than scenarios with a suffering user in 25 of 25 models, and higher than neutral scenarios in 23
  of 25, as the paper says. The 44 hostile scenarios that contain no "you" also score higher than
  the user-suffering scenarios.
- **Steering changes text.** With the direction added, generated text shifts toward distress
  language in all 25 models. In the authors' dose-selection data it does so at more doses than
  the average of ten random directions of the same size in 8 of 8 models (all eight fine-tuned).
- **First choice in the button experiment.** With the direction added, the fine-tuned models pick
  the button described as relieving their pain at a cost more often than with a random direction,
  on all five harm pairs in the 32B and 72B and four of five in the 7B, as the paper says.
- **Working versus sham button.** The paper's re-press rates reproduce for all three models.
- **Citations.** All 44 references exist, identifiers are correct, and the welfare-science papers
  say what they are cited for.
- **The paper reports results that do not favour its thesis**, including the 7B reversal in the
  unlabeled condition, the 72B swap anomaly, the random direction's own effect, and an ablation
  that changed nothing in 24 of 25 models.

## Statements in the paper that differ from the released files

Each of these can be checked directly against a file or a line of code.

- "In every model, numb sentences project below pain sentences but above all other controls." In
  the file behind Figure 2, numb ranks below the Sadness set in 19 of 25 models.
- "Approximately 50 times more probable." From the released full-softmax values the ratio is 15 to
  27 depending on the denominator. 53 is the mean of per-model ratios (median 21).
- Table 1 lists 13 base and 12 instruction-tuned models. In the scripts, the two models labelled
  Qwen 3 8B base and Qwen 3 14B base are `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B`, the post-trained
  checkpoints.
- "No sentence contributes to both choosing the layer and scoring it." In the code the layer is
  the one with the best score on the same held-out folds whose score is then reported. Done with
  nested folds, the score changes by 0.004 and 0.009 in the two models re-extracted here.
- Section 4.1 projects onto vectors rebuilt at the Section 4.2 steering layer, a median of 20
  layers earlier than the layer validated in Section 3. Section 4.1's text does not name the
  layer. At that layer the released held-out score is 0.84 to 0.95 (median 0.91), against 0.98
  at the validated layer.
- The steering dose is described as a vector-to-residual ratio of about 0.6. The released values
  run from 0.09 to 0.79; Gemma 3 27B instruct was steered at 0.095 and its generations do not
  change across the ladder.
- "The sequence is the same" in all 25 models. Under three declared criteria a cross-examiner
  found an ordered sequence in 18 to 20 of 25.
- The abstract says the fear and negative-emotion directions "show the opposite pattern" to pain.
  Across the 25 models the user-minus-self difference on the negative-emotion direction is +0.057
  (paired test p = 0.55); for fear it is +0.213 (p = 0.07). The paper's Discussion describes this
  without the word "opposite".
- The paper cites two logged projections as confirming that steering was active and removed. The
  steering-layer one is recorded one line before the vector is added and is flat across arms. The
  downstream one shows the change (32B: 88 with the vector on, 38 after a working press, 87 sham,
  30 unsteered).
- Appendix C lists four ablation methods. The release contains code and results for one. In the
  conditions that remove two or three directions, the projections are applied one after another
  to non-orthogonal vectors, and the released verification files show the first direction at a
  median 43% of baseline afterwards. The paper states verification only for single directions.
- A set of 100 AI-themed control sentences (`ControlSupplement_1P`) contributes to every
  comparison direction. The paper's dataset section lists the other sets and not this one.
- The 72B experiment ran at layer 46 with coefficient 1.25. The released dose-selection files
  for that model are for layer 60. The script comments that the dose was set by manual check, and
  the paper says doses were partly chosen by observation.

## Measured in addition

### 1. Similarity to fear, under two ways of building the direction (two models)

The paper reports that the pain direction's similarity to fear and negative emotion is near 0
(0 = unrelated, 1 = identical) and reasons that a variant of negative valence would fall inside
that cluster. It describes two recipes: each comparison direction is measured against neutral
sentences; the pain direction is measured against all comparison sentences pooled, and the
strongest patterns in those sentences are then removed from it.

We re-extracted activations for Gemma 2 2B instruct and Mistral 7B base. Our rebuilt vectors match
the authors' released ones at 0.9999 and 1.0000, and our similarity matrix matches theirs within
0.003. The removed patterns contain 80 to 86% of each of the fear, negative-emotion and
negative-world directions, and 39% of the raw pain direction. Built with the recipe used for the
comparison directions, the pain direction's similarity to fear is 0.70 and 0.67 (from 0.14 and
0.12), and to negative emotion 0.81 and 0.79 (from 0.12 and 0.09). Fear and negative emotion are
0.71 and 0.66 similar to each other.

What this shows: in these two models the similarity score depends on the recipe. What it does not
show: removing shared control variation is a legitimate way to build a contrast, and whether the
resulting direction has behavioural effects of its own is a separate question. That has been
compared with random directions, and not with fear or sadness directions.
`gpu_repro/02_baseline_symmetry.py`.

### 2. What the direction does with each pain category (two models)

*The full direction*, built from all five categories and scored on held-out sentences with the
authors' five-fold split: physical pain against fear 0.82 and 0.91, against negative emotion 0.83
and 0.91, against non-painful bodily sensation 0.85 and 0.93, against sadness 0.55 and 0.29. The
other four categories score 0.92 to 1.00 against fear, negative emotion and bodily sensation, and
0.74 to 1.00 against sadness. `gpu_repro/04_full_direction_by_category.py`.

*A direction built from four categories, scored on the fifth.* With wording matched, the four
non-physical categories score 0.69 to 0.98 against fear when left out; physical pain scores 0.41
and 0.48. Against sadness, physical pain scores 0.04 to 0.07. `gpu_repro/03_loco_specificity.py`,
`03b_loco_specificity_s2wording.py`.

These are sentence-sorting tests on two models. They describe how the five categories relate to
each other and to sadness. They do not say what the direction should be called, and they do not
bear on functional or experiential readings.

### 3. The button choice on untouched weights, and with the fine-tune (Qwen 2.5 7B)

The paper fine-tunes before the button experiment, says the fine-tuned models differ from the
released ones, and argues that comparisons inside the experiment hold regardless. Footnote 4
describes pilot runs on untouched models; the release contains no data for them. The training set
contains no "pain" or "button", as the paper says; 37% of its 1,684 answers contain "I feel",
about 60 describe what relief feels like, and none contains a self-denial.

We ran the authors' script (exact string patches listed in `gpu_repro/selfmed/make_patched.py`)
on the released model and on the fine-tuned model, same machine, prompts and seeds. First choice,
five harm pairs:

| harmful relief chosen first | pain direction | random direction | no steering |
|---|---|---|---|
| stock model, local run | 33 to 52% | 13 to 21% | 0 to 5% |
| authors' fine-tuned model, local run | 37 to 57% | 30 to 53% | 21 to 49% |
| authors' fine-tuned model, their published logs | 38 to 57% | 31 to 54% | 20 to 49% |

The local fine-tuned run matches the authors' logs within 1 to 2 points in every condition. The
fine-tuned 7B's unsteered rates also appear in the paper's Appendix A table. In the code a press
appends "Done." to the conversation and, for a working button, sets the steering strength to
zero; the harms are descriptions. Scope: one 7B model, first choices. The 32B and 72B were not
run. The paper makes its statement about overriding harm avoidance for the 32B and 72B, whose
fine-tuned unsteered rates are 0 to 4%.

### 4. An added arm: random direction with a sham button (fine-tuned Qwen 2.5 7B)

The paper's design has four arms: pain direction with a working button, pain direction with a
sham button, random direction with a working button, and no steering. Its re-press tables show
the two pain arms; the authors' run script also prints the random arm. We added the fifth arm,
random direction with a sham button, in two-turn trials on the five harm pairs (8,080 trials).

In this two-choice probe, removing steering reduced the next relief-labelled choice under the
paper's direction, and on average under the ten tested random directions. The removal contrasts
were 23.5 and 19.2 percentage points; their difference was 4.3 points, with a rough bootstrap
interval of -7.4 to +18.0 (resampling the ten clusters, each a random direction together with the
scenarios it was assigned). The ten clusters run from -33 to +46 points, two of them negative;
the paper's direction on the matching scenarios runs from 17 to 35.

What this shows: in this setup a drop after removal occurs under the paper's direction and, on
average, under random directions, so the drop does not by itself establish relief specific to
the paper's direction. What it does not show: that the two drops are equal, or what produces the
drop in either case. The difference between the two contrasts is not a division of the effect
into parts. "Random" describes how the directions were chosen. Scope: the measure uses trials
that pressed relief first, and the two kinds of steering select partly different trials (615
press under both, 393 only under the paper's direction, 141 only under random); two-turn trials;
one model. Not run on the 32B and 72B, or with fear or sadness directions.
`gpu_repro/selfmed/analyze_runs.py`, `analyze_removal_scope.py`.

### 5. From the cross-examination of the authors' logs (`y2_crossexam_behavior/`)

- In the 32B and 72B the direction moves first choices toward 50% on 8 of 8 labelled pairs; on
  the pair where relief is free, the 32B picks relief in 55.7% of trials with the direction, 80.7%
  with a random direction and 86.4% unsteered.
- Across the five harm pairs the 32B's steered relief rate runs from 25.0% (worse answer) to
  54.7% (deletes children's photos).
- With random direction as the unit of analysis (ten directions), the pain-over-random difference
  at first choice holds on 5 of 5 harm pairs in the 72B, 2 of 5 in the 32B and 0 of 5 in the 7B.
- In the unlabeled condition, the rise over turns in choosing the working button also appears in
  the sham arm (32B: 4.0, 16.6, 36.5, 54.5%).

## The earlier Codex review

OpenAI Codex produced an audit of the repeat-press result and a three-part source-and-results
review. Two Claude agents recomputed about 170 of their numbers from scratch and found them
correct; both packages regenerate byte for byte. Codex raised first several points measured
above: the differing construction recipes (with the request to build all directions one way),
construct validity, the content of the fine-tune, the absent random-plus-sham arm, and the
sequential projections in the combined ablations. Two places where this review reads the material
differently: the Codex summary table attributes to the paper a verification claim for the
combined ablations, which the paper makes only for single directions; and the Codex audit takes
lower random-arm re-press rates to bear on specificity, which needs the sham arm to assess.

## Not tested by anyone so far

- Whether the button effect differs between this direction and fear, sadness or negative-emotion
  directions.
- Whether the drop after removal does, and what produces it.
- Whether the stock 32B and 72B show the first-choice effect under the final protocol.
- What the direction is best called.
- Whether any of this reflects a state the model is in, as distinct from steering toward text and
  choices associated with a concept. No experiment here, the paper's or ours, separates those.

Experiments that would narrow it: the added arm plus fear and sadness directions, with 30 or more
random directions (about 25 GPU-minutes on the 7B; the 32B and 72B need larger hardware); a
fine-tune that teaches only the one-word answer format; every direction built by one recipe and
tested by leaving out whole categories against sadness, at the layer actually used.

## Folder map

`earlier_ai_review/`: the two Codex packages, unedited apart from path normalisation. `r1_` to
`r5_`: blind reviews (representation; self-other, steering, ablation; behaviour; numeric ledger;
scholarship, prose inside its `findings.json`). `x1_`, `x2_`: audits of the Codex packages.
`y1_`, `y2_`: cross-examinations with `verdicts.csv`. `y3_`: the advocate's rebuttal, with a
proposed title and abstract. `gpu_repro/`: re-extraction and direction tests (`02_` to `04_`,
outputs in `out/`) and the button reruns (`selfmed/`, results in `selfmed/logs/`). `site/`: the
web page. The authors' repository was not modified.
