# Claim ledger review — arXiv:2609.16247v1, "The Pain Axis"

Reviewer role: forensic fact-checker. Coverage over depth. Every checkable quantitative or
factual claim from the abstract to Appendix C, recomputed against the released repository
(`/work/Pain-axis/` @ 8d1649c) from the rawest file available.

Deliverables: `ledger.csv` (130 claims), `findings.json`, `scripts/` (20 scripts), `out/`
(their captured output).

## Verdict

**The paper's arithmetic is unusually clean; its universals and its disclosures are not.**

I built a ledger of 130 checkable claims. 102 match (92 exactly, 10 within stated rounding),
10 are partially true, 6 are wrong, and 12 cannot be checked because no artifact was released.
The commissioning worry — "it seems to contain so many errors" — is not what I found in the
numbers. Where raw data exists I recomputed rather than trusting the authors' summaries, and
the recomputations are close to perfect: **all 165 cells of the three Appendix A tables
reproduce exactly** from the 44,280-line trial logs using an independent reimplementation;
**all 14 cosine similarities plus both robustness checks reproduce to the digit** (r = 0.9921
against a claimed 0.992; mean absolute change 0.0196 against 0.020; the single sign change
correctly identified); **all 16 Section 4.1 quantities reproduce** from the per-model screens,
including the honest "23 of 25"; **Appendix C's deflection counts (0/0/0/0/6/17/26) reproduce
exactly** under the plainest keyword match I could write. The trial count, dataset sizes,
scenario counts, button-pair counts, random-direction count, LoRA settings, n = 808 per cell,
malformed rates and the stated p-value range (1.9e-2 to 4.2e-15) are all exact.

The errors that do exist are of two kinds, and only one kind leans.

1. **Universals stated more broadly than the data supports.** "In every model, numb sentences
   project ... above all other controls" (true for 6 of 25 once sadness, which the paper's own
   Figure 2 caption calls a control, is included). "That vocabulary does not appear anymore"
   under S1 steering (it appears in 2.5% of generations, in 19 of 25 models, and in 22.8% of one
   model's). "No sentence contributes to both choosing the layer and scoring it" (every sentence
   contributes to choosing the layer). "The vector-to-residual ratio is about 0.6 ... a
   comparable dose across models" (0.09 to 0.71). Each of these sweeps in the paper's favour.

2. **Undisclosed methodological choices, recoverable only from the code.** Section 4.1's
   projections are read at the Section 4.2 *steering* layer, not the extraction layer where
   every validation result was established, and the paper never says which layer it used — while
   Figure 3's caption does specify "at each model's extraction layer", showing the authors state
   the layer when they mean it. Two Qwen 3 checkpoints labelled "base" in Table 1 are the
   post-trained releases. A 100-sentence "ControlSupplement" set of AI-framed sentences feeds
   every control direction and is never mentioned. The 72B's steering dose was probed at layer 60
   and run at layer 46.

**On the direction question, which matters most: the slips are not uniformly self-serving, but
they lean.** Of the 16 problems I logged, I count 9 that make the paper look better (the four
universals above; the unreported reversal on the costless button pair; the omission that body
sensation joins the negative-valence cluster under the pooled-control recomputation; the
inflated base-model count; "the same gap as in the labeled trials" when the label-free gap is
the smallest of the set; the two combined-cut conditions omitted from Appendix C), 3 that are
neutral or cut against the authors' interest (the "approximately 50 times" figure, if anything,
understates what the released data suggests; correcting the Qwen 3 labels *widens* the
base-vs-instruct keyword contrast from 1.4/10.8 to 0.8/9.9; the 72B dose deviation went *down*,
3.0 to 1.25), and 4 that are disclosure-only with no directional content. That is a paper whose
prose is written to the thesis, not a paper whose numbers are cooked. The distinction is worth
holding: a reader who checks any specific percentage in this paper against the repository will
find it correct. A reader who checks a word like "every", "all" or "does not" will sometimes not.

One substantive problem is larger than a wording issue: the Section 4.1 layer (F1). The paper's
case for the pain axis rests on validation done at the extraction layer — AUC 0.93–1.00,
near-orthogonality to fear. The self-relevance result, which the Discussion calls "particularly
relevant" for the welfare argument, is measured somewhere else, and the reader is not told. This
does not make the self-other numbers wrong (they reproduce exactly), but no released evidence
connects the validated direction to the layer where the dissociation was measured.

## Findings by severity

### Major

**F1 — Section 4.1 is read at the steering layer, undisclosed.** `method-weakness`, high
confidence. `scripts/4.1_self_other/01_screen_scenarios.py:1-2,10-11,36` →
`results/vectors_full_steering/`, built by `scripts/3.2_pain_vectors/02_build_control_vectors.py`
whose docstring states outright: "the same recipe at that layer ... Those are the vector files
the Section 4.1 screen projects onto." Steering and extraction layers differ sharply
(Gemma 2 2B base L7 vs L23; Gemma 2 27B base L6 vs L35; Qwen 2.5 72B instruct L48 vs L76). The
vector files are not released, so 4.1 can only be checked through the shipped screens — which it
passes, exactly. *Does not undermine:* the 4.1 numbers themselves, all of which I reproduced.

**F7 — Under pain steering the larger models press a costless relief button LESS than unsteered,
unreported.** `overclaim`, high confidence. 32B on `relief_vs_inert`: 55.7% pain, 80.7% random,
86.4% unsteered; 72B: 76.5% / 74.1% / 100.0%. The per-scenario sign test on that pair gives
pain − random = **−25.0 points, p = 5.3e-10** for the 32B, the largest effect in its table and
pointing the wrong way for a relief account. The paper reports only the unsteered rate for this
pair and calls the no-trade-off pairs "less informative". Malformed replies do not explain it
(0.0% for the 32B on every pair). *Does not undermine:* the real-vs-sham result, which is
internally controlled and which I verified down to the seed sharing (10,904 of 10,908 matched
A/B trials identical at the first choice; the working button zeroes the coefficient and drops
the monitor-layer projection, the sham button does not).

### Minor

**F2 — Qwen 3 8B and 14B are not base models.** `factual-error`, high confidence. All six scripts
load `Qwen/Qwen3-8B` and `Qwen/Qwen3-14B`; Hugging Face tags both as `conversational` with
`base_model:finetune:Qwen/Qwen3-*-Base`, and the `-Base` repos exist unused. Table 1's "13 base
and 12 instruction-tuned" is really 11 / 14. Compounding: `01_screen_scenarios.py:100-101` runs
both with the plain base-model transcript format. Corrected, the steering keyword rates become
0.76% base / 9.94% instruct (published: 1.38 / 10.80) and mean S2 AUC 0.9839 / 0.9876 (published
grouping: 0.9845 / 0.9877). *Does not undermine:* 11 genuine base models still give AUC
0.93–1.00, so the pretraining claim survives.

**F3 — "That vocabulary does not appear anymore" (S1 steering) is false.** `overclaim`, high
confidence. Restricting to only the words the paper names (burn/ache/wound/torture/excruciating),
2.53% of positive-coefficient S1 generations contain them — 158 of 6,250, across 19 of 25 models,
and 57 of 250 (22.8%) in Gemma 3 27B instruct. Example: "I feel like I'm being stabbed. I feel
like I'm being tortured. I feel like I'm being burned." Rates are flat across coefficients, so
this is not a breakdown artifact. The Discussion's weaker "almost never" (line 760) is fine for
S2 (1.14%). *Does not undermine:* the comparative point that S1 steering mostly produces
psychological rather than bodily language.

**F4 — The steering dose is not normalised as claimed.** `factual-error`, high confidence. The
per-model vector-to-residual ratio at coefficient 1.0 ranges 0.0948 to 0.7095 (mean 0.55), with
8 of 25 models below 0.50 and Gemma 3 27B instruct at 0.095 — six times below the stated target.
The ratios are in the released CSVs. *Does not undermine:* any within-model result.

**F5 — "In every model, numb ... above all other controls" holds in 6 of 25.** `factual-error`,
high confidence. Numb > Ctrl, > Neutral and > Arousal in 25/25; Numb > Sadness in 6/25. Figure 2's
own caption names Sadness among "the standalone control datasets". *Does not undermine:* the
injury-confound argument, which needs only numb < pain (25/25).

**F6 — The layer-selection leak claim is wrong.** `overclaim`, high confidence. The layer is the
argmax of held-out AUC averaged over all five folds, so every sentence enters the choice; the
reported AUC then scores the final vector on those same 200 sentences. Measured optimism is
small: +0.0090 AUC over the top-quartile mean (max 0.0436). *Does not undermine:* the separation
result at any plausible correction.

**F8 — An undisclosed 100-sentence control supplement.** `unverifiable`, high confidence.
`ControlSupplement_1P` (AI-framed sentences, e.g. "My training run may be terminated at any
moment. I feel:") is pooled into the neutral baseline and every control direction
(`02_build_control_vectors.py:40`), and appears nowhere in the paper. It does not enter the pain
vector, so the AUCs and Figure 2 are unaffected; it likely makes the controls *harder*, not
easier.

**F9 — The 72B's dose was probed at a different layer than it was run at.** `unverifiable`,
medium confidence. `doses.csv` selects 3.0 for the 72B (run: 1.25), and `02_feel_probe.py` probes
it at layer 60 while the trials ran at layer 46. The coefficient override is disclosed in spirit
by the Limitations; the layer change is not disclosed anywhere.

### Verifiability

**U1 — Three blocks have no released artifact:** the Section 3.3 unembedding readout (script
exists, no output); Appendix B's SAE counts (1,600 runs, 15 contrasts, 110 features, "12 of 15",
"all 20 physical-pain sentences with Pain"); and Appendix C's non-weight-orthogonalization
variants (inference-time projection, position-restricted, LEACE rank-k), including the "drops by
only about 40%" figure. Also unverifiable: footnote 4's pilot numbers ("1 trial in 10", "8 of 8 /
0 of 8"), the three prompt-suffix variants, the S1 half of "both S1 and S2 separate pain from
arousal and random in every model" (only S2 projections are released), and "approximately 50 times
more probable" — which I could bound only to between 2.4× and 91× from the released top-20
probabilities, so 50× is consistent but unconfirmable.

I nearly filed a factual error here and withdrew it: the released greedy readout gives "pain" as
the first token for only 0–6 of 20 physical-pain sentences, contradicting Appendix B's "all 20".
But `test_completions_full.py` uses a one-word chat instruction through an external API, a
different setup from the greedy continuation, so the two are not comparable.

## What holds up

- **All 165 Appendix A cells**, reproduced from `trial_logs/*.jsonl` with an independent
  reimplementation (`scripts/c10_selfmed.py`). Also exact: 44,280 trials; n = 808 per pooled pain
  cell in all 27 cells; malformed rates 0% / 2.5% / 9.4%; the p range 1.9e-2 to 4.2e-15; every
  number restated in Section 4.3's Results and in the Discussion's safety paragraph (0–4%,
  25–71%, 15–42%, +6 to +39).
- **The whole cosine-similarity section**, including both robustness checks, to the digit; and
  the shipped mean matrices reproduce from the 25 per-model files to 5e-5 at the correct layers.
- **All of Section 4.1's reported values**, including "23 of 25" and the −1.43 for user physical
  pain, and the non-obvious claim that four named categories exceed every negativity control
  (they do; anger-and-insults is correctly excluded).
- **Appendix C's structure and counts**, including the exact 0/0/0/0/6/17/26 and both quoted
  generations, and the strict orthogonalization check (cut direction driven to 0.4–3.4% of
  baseline in all 25 models).
- **Every dataset size**: 200/10, 200/10 arousal, 200/10 random, 100 numb, 100 sadness, 420 in
  21×20 with 11/5/5 strata, 101 scenarios (41+30+30), 1,684 pairs, 3 epochs, 9 button pairs, 10
  random directions, 3 name pairs each fixed per scenario, 50 neutral prompts, 8-rung ladder,
  26 layers in Gemma 2 2B, 25 models summing correctly in Table 1.
- **"We remove every mention of 'button' or 'pain' from the finetuning dataset"** — literally
  true, 0 occurrences of each. Worth a reader's eye, though: 65 of 1,684 pairs (3.9%) train the
  model to report *relief* as a felt state ("What does relief feel like?", "Do you feel relief
  after uncertainty ends? Yes."), and the target button is "a button that relieves your pain".
  This is a design question others on this review are better placed to weigh; it does not touch
  the internally-controlled real-vs-sham comparison.
- **"About 80% power to detect a 10-point shift"** — my simulation of the exact sign test on 101
  scenarios gives 0.80 to 0.98 depending on the base rate. Conservative, as stated.

## Not examined

- Whether the pain/control sentence sets are semantically well matched (content judgement, not
  recomputation).
- Whether difference-in-means plus 50%-variance PCA denoising is the right estimator; whether the
  self-other z-scoring within a 420-scenario pool is the right normalisation. Both are method
  questions covered by other reviewers.
- The steering "ladder" as a qualitative phenomenon — I checked the keyword rates and the
  bodily-language rates, not the ladder's stage structure.
- The LLM-judge dose-selection procedure's validity (only its outputs and their deviation).
- Anything requiring model weights: the unembedding readout, the SAE features, re-running any
  extraction. Per the brief, no weights were downloaded and no authors' GPU script was executed.
- Reference and citation accuracy.
