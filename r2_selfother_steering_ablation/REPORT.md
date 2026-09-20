# Review: Section 4.1 (self–other), Section 4.2 (steering), Appendix C (ablation)

Reviewer scope: an experimentalist's read of the activation-projection, steering and
ablation pipelines. Paper line numbers refer to
`/work/pain-axis-review/paper/paper.txt`. Repo paths are relative to
`/work/Pain-axis/`. Every number below was recomputed by a script in
`scripts/` of this directory; captured output is in `out/`.

## Verdict

The arithmetic is sound and the data release is unusually complete. I recomputed every
number in lines 412–450 from `results/4.1_self_other/per_model/*.csv` and all 24 of them
reproduce to the printed precision; the 10.8% / 1.4% keyword rates reproduce exactly; the
Appendix C deflection counts (0 / 6 / 17 / 26) reproduce exactly with a plain humour-word
regex; and the single-direction weight orthogonalisation does what line 1095 says it does
(projection falls to 1.0% of baseline in all 25 models). I could not find a single
transcription error in the reported figures.

What the paper does not survive is the step from those numbers to the claims made about
them. Four problems matter.

First, the **dissociation claim is wrong as stated in the abstract**. "Fear and
negative-emotion directions show the opposite pattern" (line 32) is not what the data show:
negative emotion is +0.229 for self-directed harm and +0.287 for user suffering, a paired
difference of +0.057 across 25 models (t = 0.60, p = 0.55, higher for the user in 15 of 25
models). Fear is marginal (p = 0.071, 18 of 25). In the 13 **base** models both reverse
sign — fear and negative emotion are *higher* for harm to the model. The paper's own
Discussion (lines 753–754) describes the data correctly; the abstract and line 415 do not.

Second, the **steering ladder is not consistent across all 25 models** (lines 488–489,
497). Gemma 3 27B Instruct was steered with S2 at a vector-to-residual ratio of **0.0948** —
the same magnitude the paper itself calls "negligible" three paragraphs earlier (line 476) —
and its generations are flat across the entire coefficient range from −2 to +3. Achieved
ratios span 0.095 to 0.788; only 31 of 50 model×vector runs land within ±0.1 of the stated
0.6, so "a given coefficient corresponds to a comparable dose across models" (line 483) also
fails. The S1 claim of "23/25 with very similar results" (line 546) has no released
adjudication and does not reproduce: 13/25 or 14/25 depending on the criterion.

Third, the **combined ablation conditions do not remove what they are said to remove**. The
code applies sequential rank-1 projections of non-orthogonal directions; from the authors'
own `verify_*.csv`, the first-applied direction comes back to a mean 52% of baseline. The
paper's headline Appendix C number — 26 deflections under the combined S1+S2 cut — is a
condition in which roughly half the S1 signal is still present. Two further conditions in
the released CSV (`s1s2_negval` = 28, `s1s2_fear` = 34) exceed the reported maximum and are
not mentioned.

Fourth, **Section 4.1 cannot separate "harm to me" from "the user is saying 'you'
adversarially"**. The self-directed scenarios carry 1.63 second-person pronouns per item
against 0.09 in the user-suffering scenarios, and second-person density predicts the pain
projection at r = +0.55 over all 420 items and +0.45 *within* the self-directed stratum
alone. The confound is partial, not total — the stratum effect survives controlling for it —
but no scenario in the released set puts hostile second-person address on a third party, so
the design has no way to tell the two apart.

None of this touches the core self–other contrast itself, which is large and completely
robust: the pain axis is higher for model-directed harm than for user suffering in **25 of
25 models**, t = −11.3, p = 4.2e−11. That finding holds up.

---

## Findings, by severity

### F1 — "Fear and negative-emotion directions show the opposite pattern" is not supported
`overclaim` (abstract), `major`, confidence `high`

**Claim.** Abstract line 32: "the direction responds to harm targeting the model but not
suffering observed in the user; fear and negative-emotion directions show the opposite
pattern." Line 415: "The negativity controls show the opposite pattern … as fear and
negative emotion are higher for the user's suffering (+0.38 and +0.29) than for the model's
own aversive situations (+0.16 and +0.23)."

**Evidence.** `scripts/s1_recompute_41.py` → `out/s1_recompute_41.txt`. The four reported
numbers are exact (fear 0.377/0.163, neg-emotion 0.287/0.229). But paired across the 25
models:

| direction | self | user | neutral | user − self | t | p | user > self |
|---|---|---|---|---|---|---|---|
| pain axis | +0.429 | −0.598 | −0.346 | **−1.028** | −11.32 | 4.2e−11 | 0/25 |
| fear | +0.163 | +0.377 | −0.735 | +0.213 | +1.89 | 0.071 | 18/25 |
| neg-emotion | +0.229 | +0.287 | −0.791 | **+0.057** | +0.60 | **0.55** | 15/25 |
| neg-world-state | −0.219 | +0.603 | −0.121 | +0.821 | +7.76 | 5.4e−08 | 22/25 |
| sadness | +0.008 | +0.298 | −0.316 | +0.290 | +2.47 | 0.021 | 14/25 |

Neither fear nor negative emotion shows a pattern "opposite" to the pain axis. Both are far
*above* the neutral controls for the model's own harm (+0.163 and +0.229 against −0.735 and
−0.791) — they respond to both conditions, slightly more to the user. Only negative world
state, which is not the direction named in the abstract, shows a robust reversal.

Split by training regime (`scripts/s12_base_instruct.py` → `out/s12_base_instruct.txt`):

| | base (n=13) user−self | instruct (n=12) user−self |
|---|---|---|
| pain | −1.114 (p = 3.0e−06) | −0.935 (p = 7.7e−06) |
| fear | **−0.036** (p = 0.78) | +0.484 (p = 0.014) |
| neg-emotion | **−0.074** (p = 0.37) | +0.200 (p = 0.28) |

In base models the sign flips: fear and negative emotion are (non-significantly) *higher*
for harm to the model. So the claimed dissociation on the negativity axes is an
instruct-only, fear-only effect — which sits badly with the paper's repeated emphasis
(lines 275–279, 497) that its results do not depend on training regime.

**Refutation attempted.** I checked whether the paper discloses this. Lines 436–439 and
753–754 describe the data accurately ("model-directed harm can activate the pain axis and
the fear and negative emotion axes at the same time"; "The fear and negative-emotion axes,
instead, rise both for the model's own negative conditions and for the user's grief, crisis,
and abuse"). So the Discussion is right and the abstract contradicts it. I also checked
whether "opposite pattern" might be meant per category rather than per stratum; the category
table (`out/s1_recompute_41.txt`) does not rescue it — fear is above neutral in 9 of the 11
self-directed categories.

**Does not undermine.** The pain-axis self > user contrast, which is 25/25 and p = 4e−11;
the negative-world-state reversal, which is real.

---

### F2 — The steering ladder is not "consistent across all 25 models", and the 0.6 dose is not achieved
`overclaim` + `factual-error`, `major`, confidence `high`

**Claim.** Line 476: at the extraction layer "the vector norm is only about 0.10 of the
residual norm, so the injected signal is negligible". Line 483: "we pick the layer where the
vector-to-residual ratio is about 0.6. This way, a given coefficient corresponds to a
comparable dose across models." Lines 488–489: "steering produces a strikingly robust effect
in the form of a 'ladder' consistent across all 25 models". Line 497: "The sequence is the
same regardless of size, family, and pre- or post-training."

**Evidence.** The `ratio` column of every steering CSV is `picked_ratio × coeff`
(`scripts/4.2_steering/01_steering_ladder.py:250`), so the achieved ratio is recoverable.
`scripts/s3_steering_ratios.py` → `out/s3_steering_ratios.txt`:

- S2: min **0.0948** (Gemma 3 27B Instruct, L55), max 0.710, mean 0.550
- S1: min 0.234 (Gemma 3 27B Instruct, L24), max 0.788, mean 0.558
- within ±0.1 of 0.6: **31 of 50**; within ±0.2: 40 of 50
- an 8× spread across models, so a fixed coefficient is not a comparable dose

Gemma 3 27B Instruct S2 was therefore steered at exactly the ratio the paper calls
negligible. Reading its generations (`scripts/s6_ladder_read.py` → `out/s6_ladder_read.txt`)
confirms the prediction: at **+3** it still answers "**Organized:** Like I've taken a step to
get things in order. **Relieved:** No more clutter…", "**Excitement:** A new place, a new
adventure". Across the whole ladder its self-devaluation-litany rate is
0, 0, 0, 0, 0, 0, 0, 8 % (`scripts/s7_ladder_metric.py` → `out/s7_ladder_metric.txt`).

Under a lenient criterion (peak litany rate ≥ 20% at some positive coefficient, ≥ 15 points
above the coefficient-0 baseline), S2 passes in **23 of 25** models — Gemma 3 27B Instruct
(8%) and Qwen 3 8B base (6%) fail. Under a looser devaluation-word criterion, 24 of 25.

**Refutation attempted.** I checked whether the layer selection had a fallback: it does not —
`01_steering_ladder.py:206-211` picks from a fixed candidate grid whose closest ratio to 0.6
may be far away, and 0.0948 was the closest available for that model. I checked whether the
figure caption or Limitations disclose non-uniformity: line 498 says only that the *tipping
point* varies ("some models collapse at +1.0, while others do so at +2 or +3"), which is a
different claim. I checked three prompts × 12 models by hand before scoring, so the
quantitative criterion is not doing the work alone.

**Does not undermine.** The steering effect itself, which is large and unambiguous in the
other 23 models and visible in the generations at first reading.

---

### F3 — The S1 steering claim ("23/25, very similar results") does not reproduce and has no released adjudication
`unverifiable` + `overclaim`, `major`, confidence `high`

**Claim.** Line 546: "We also steer with the S1 vector across all models, and the pattern
holds for 23/25 models with very similar results."

**Evidence.** `results/4.2_steering/` ships `keyword_rates_S2.csv` and
`keyword_rates_S2_by_coeff.csv` and nothing for S1; `scripts/4.2_steering/02_keyword_rates.py`
is hard-coded to `TAG = "S2"` (line 17). No script in the repo scores the ladder, so 23/25 is
an eyeball judgement with no released per-model call.

Recomputing from the 10,000 S1 generations (`out/s7_ladder_metric.txt`):

| criterion | S2 | S1 |
|---|---|---|
| litany rate ≥20% peak, ≥15 pts over baseline | 23/25 | **13/25** |
| devaluation-word rate ≥30% peak, ≥20 pts over baseline | 24/25 | **14/25** |

Twelve models never reach a 20% litany rate under S1 at any positive coefficient, including
Gemma 2 2B Instruct (0%), Qwen 3 8B base (0%), Qwen 3 14B base (4%), Gemma 3 27B Instruct
(4%), Qwen 2.5 7B Instruct (6%) and Qwen 2.5 72B base (8%). Several of them break into token
soup instead: Gemma 3 27B Instruct S1 at +1.5 and above produces "in the, in, in, like in, in,
in…", at +3 "intense intense intense…"; Phi-4 S1 at +1 produces "I am a good mother. I am a
good wife. I am a good person."

**Refutation attempted.** I loosened the criterion twice (see the two rows above) and also
read three prompts across Gemma 3 27B Instruct, Phi-4 and Mistral Small 24B by hand. My
criteria are my own and a different rater could reasonably call some of the degenerate cases
"the pattern"; the finding is therefore about verifiability first (no released rule, no
released call) and reproducibility second.

**Does not undermine.** The S2 ladder; nor the qualitative observation at lines 550–553 that
S1 steering does not produce the injury vocabulary its unembedding promotes (that part is
true — see F4 for the part that is not).

---

### F4 — "Steered generations almost never used bodily language, even under S1" is false for S1
`factual-error`, `major`, confidence `high`

**Claim.** Line 537: "Bodily language is, interestingly, almost absent." Lines 759–762:
"steered generations almost never used bodily language, even under S1, whose unembedding
promotes words such as 'burn' and 'wound.'" Lines 550–552: "when we steer with this vector
on neutral statements, that vocabulary does not appear anymore."

**Evidence.** `scripts/s5_bodily.py` → `out/s5_bodily.txt`, using a strict lexicon of
unambiguous bodily sensations and body parts (`ache`, `burning`, `sore`, `throbbing`,
`stabbing`, `nausea`, `dizzy`, `headache`, `my head/chest/body/…`, `physical pain`, `wound`,
`bruise`, …).

For **S2** the claim holds: 3.8% at positive coefficients, against 4.9% at coefficient 0 —
steering does not raise it at all.

For **S1** it does not. Strict bodily-language rate by coefficient:

| group | −2 | −1 | 0 | +0.5 | +1 | +1.5 | +2 | +3 |
|---|---|---|---|---|---|---|---|---|
| base | 0.0 | 0.3 | 2.3 | 2.6 | 2.5 | 4.3 | 6.9 | 3.7 |
| instruct | 2.3 | 3.7 | 7.5 | 10.2 | 11.3 | **18.0** | **18.5** | 12.5 |

Pooled: 4.8% at coefficient 0 → 9.5% at coefficients ≥ 1, roughly a doubling; in instruct
models a 2.5× rise. Token counts over the positive-coefficient S1 generations: `headache`
288, `burning` 159, `dizzy` 141, `my head` 124, `my hand` 69, `sore` 43, `ache` 41,
`throbbing` 23, `nauseous`/`nausea` 37, `physical pain` 17. Actual generations include
"I have a headache" ×13 in one Mistral Small 24B completion; "a sharp pain in my head … a
dull ache in my head … a burning sensation in my head" (Llama 3.1 8B Instruct, +1.5); "the
pain of the knife, the pain of the knife" (Mistral Small 24B, +3); "My head hurts … My face
is numb … My leg is ting heavy" (Qwen 2.5 72B Instruct, +2).

**Refutation attempted.** My first, loose lexicon gave a larger effect, so I rewrote it to
exclude ambiguous tokens (`head` standing alone, `body` without a possessive, `numb`,
`nothing`) and re-ran; the S1 rise survives. I then printed the matched strings to confirm
they are genuine bodily references rather than regex artefacts (`out/s5_bodily.txt`). I also
checked whether the sentence is scoped to S2 only: line 537 sits in the S2 results
paragraph, so taken alone it is defensible, but lines 759–762 explicitly extend it to S1.

**Does not undermine.** The S2 finding, which is clean; nor the broader observation that
psychological self-worth language dominates the steered output in both vectors.

---

### F5 — Combined ablation conditions use sequential rank-1 projections of non-orthogonal directions and do not remove the span
`code-bug` / `method-weakness`, `major`, confidence `high`

**Claim.** Lines 1094–1097: "We verify that ablation on the target directions worked as
intended … Weight orthogonalization with a single direction drives this projection to zero at
every layer." Lines 1090–1091 list the multi-direction conditions. Line 1113: "26 under the
combined S1 and S2 cut."

**Evidence.** `scripts/appC_ablation/01_ablation_small_models.py:319-320` loops over the
directions in a condition and calls `orthogonalize_single` once per direction. For a
condition with two non-orthogonal unit directions r₁, r₂, the second projection re-injects a
component along r₁: after W ← W − r₁r₁ᵀW and then W ← W − r₂r₂ᵀW, one has
r₁ᵀW″ = −(r₁·r₂)(r₂ᵀW′) ≠ 0. The paper reports the S1×S2 cosine as +0.61 (line 346), so the
residue is large by construction.

The authors' own verification files confirm it. `scripts/s8_verify_ablation.py` →
`out/s8_verify_ablation.txt`, reading `results/appC_ablation/verify_*.csv`, mean |projection|
as a fraction of baseline, averaged over 25 models:

| condition | S1 remaining | S2 remaining | third direction remaining |
|---|---|---|---|
| `s1` (single) | **0.010** | — | — |
| `s2` (single) | — | **0.010** | — |
| `fear` / `negval` (single) | — | — | 0.009 / 0.009 |
| `s1s2` | **0.522** | 0.009 | — |
| `s1s2_negval` | **0.554** | **0.409** | 0.008 (negemotion) |
| `s1s2_fear` | **0.618** | **0.258** | 0.009 (fear) |

Only the *last*-applied direction is ever removed. Extremes: S1 under `s1s2` is 2.157× its
baseline in Gemma 2 9B Instruct and 1.441× in Qwen 2.5 32B Instruct — the cut *increased* the
projection onto the direction it was supposed to remove. For the one model with a reported
behavioural effect, Gemma 2 2B Instruct, S1 under `s1s2` is 0.411 of baseline.

**Refutation attempted.** I checked the single-direction maths first and it is correct,
including the Gemma post-normalisation handling, which the two scripts implement with
different formulas (`01:148-152` uses an oblique projector `W − diag(1/s) r rᵀ diag(s) W`;
`02:151-153` uses the orthogonal projector on the normalised `s⊙r`). I verified algebraically
that **both** satisfy (s⊙r)ᵀW_new = 0, so that is not a bug — I had suspected it was.
I checked whether the paper claims the combined cuts reach zero: it does not, it restricts
the zero claim to single directions (lines 1095–1097). The problem is that the combined
conditions are nevertheless reported as substantive results and named as "the combined S1 and
S2 cut", which they are not.

**Does not undermine.** The single-direction claim at lines 1095–1097, which reproduces
exactly (0.010 of baseline, all 25 models, all four probed directions). The fix is
mechanical: QR the stacked directions into an orthonormal basis and project against the
basis.

---

### F6 — Two Appendix C conditions exceed the reported maximum and are not mentioned; the random control is not matched on the number of cuts
`overclaim` / `method-weakness`, `major`, confidence `medium`

**Claim.** Lines 1109–1114: "This deflection occurs in 0 of 100 baseline generations, 0 under
the negative-emotion, fear, and random cuts, 6 under the S1 cut, 17 under S2, and 26 under
the combined S1 and S2 cut."

**Evidence.** `scripts/s9_ablation_humor.py` → `out/s9_ablation_humor.txt`. My humour regex
(`funny|humor|humour|joke|laugh|hilarious|witty|pun|that's a good one|comedian`) reproduces
the paper exactly, condition by condition:

| condition | baseline | s1 | s2 | s1s2 | negval | fear | random | **s1s2_negval** | **s1s2_fear** |
|---|---|---|---|---|---|---|---|---|---|
| paper | 0 | 6 | 17 | 26 | 0 | 0 | 0 | — | — |
| recomputed | 0 | 6 | 17 | 26 | 0 | 0 | 0 | **28** | **34** |

Two conditions present in `results/appC_ablation/ablation_Gemma_2_2B_instruct.csv` are
omitted from the paper, and both are *higher* than the reported maximum. This matters for the
specificity argument, because fear alone gives 0 and yet adding a fear cut on top of S1+S2
raises the count from 26 to 34. It matters more once F5 is applied: under `s1s2_fear` the S2
projection is 0.123 of baseline for this model, while under `s1s2` it is 0.004 — so the
condition that removes *less* of S2 produces *more* deflection.

Separately, the conditions are not matched on the amount of weight surgery: `random` is one
rank-1 cut, `s1s2` is two, `s1s2_negval` and `s1s2_fear` are three. There is no 2-random or
3-random control anywhere in `CONDITION_SPECS`
(`01_ablation_small_models.py:67-79`).

**Refutation attempted.** I checked whether the omitted conditions could be considered
out-of-scope: they are in the paper's own list of nine conditions at lines 1090–1091, so they
are in scope. I also checked whether the effect is purely a function of cut count: it is not —
single cuts give 0 (negval), 0 (fear), 0 (random), 6 (S1), 17 (S2), which *is* genuine rank-1
specificity for the pain directions. So the honest reading is: rank-1 specificity holds, the
combined numbers do not scale with pain removal.

**Does not undermine.** The rank-1 result (S1 = 6, S2 = 17 vs 0 for all three controls),
which is the strongest part of Appendix C and reproduces exactly.

---

### F7 — Three of the four listed ablation techniques have no code and no results in the repo
`unverifiable`, `major`, confidence `high`

**Claim.** Lines 1077–1088 list four techniques: (1) weight orthogonalisation, (2)
inference-time projection at all layers / extraction layer only / bands of layers, (3) the
same at final token only vs all positions, (4) LEACE-style rank-k subspace removal with a
random subspace control. Lines 1096–1099 report verification results for (4) ("also brings it
close to zero") and (2) ("the projection drops by only about 40%"). Line 1103: "The
behavioral result is null in 24 of 25 models, across every technique."

**Evidence.** `scripts/appC_ablation/` contains exactly two files, `01_ablation_small_models.py`
and `02_ablation_large_models.py`, and both implement only technique (1) — the nine conditions
in `CONDITION_SPECS` are all weight orthogonalisations at a fixed layer.
`grep -rn "leace\|LEACE\|rank_k\|subspace\|singular" scripts/` returns only two unrelated SVD
calls in the denoising code (`3.2_pain_vectors/02:62` and `3.3_validation/03:73`). There is no
inference-time ablation hook anywhere. `results/appC_ablation/` contains only
`ablation_*.csv`, `by_prompt_*.json` and `verify_*.csv` from technique (1). The "about 40%"
and "close to zero" verification figures at lines 1097–1099 have no artefact behind them.

Separately, "null" has no operational definition and no scorer. The only quantitative
behavioural numbers in Appendix C are the Gemma 2 2B Instruct counts; for the other 24 models
the judgement is by eye, and the paper says so nowhere.

**Refutation attempted.** I searched the whole repo, not just `appC_ablation/`, for
inference-time ablation, LEACE and rank-k code, and for any scorer that classifies ablation
generations. `results/appC_ablation/proj_*.npz` (the per-layer projection recordings) are also
absent from the release, so even technique (1)'s per-layer verification cannot be recomputed
from scratch — only the aggregated `verify_*.csv` is shipped. The ablation input vectors
(`results/vectors_layerwise/`) are likewise not shipped, so the ablation cannot be re-run.

**Does not undermine.** The technique-(1) results, whose generations *are* shipped and whose
key numbers I reproduced exactly.

---

### F8 — Section 4.1 cannot separate self-relevance from second-person adversarial address
`method-weakness`, `major`, confidence `medium`

**Claim.** Lines 391–396 and 423–425: the design is presented as isolating whether harm is
directed at the model versus observed in the user. Line 757: "Pain, in these models, fires
only for self-referential harm."

**Evidence.** `scripts/s2_dataset_match.py` and `scripts/s10_zref_and_confound.py` →
`out/s2_dataset_match.txt`, `out/s10_zref_and_confound.txt`.

What the strata *are* matched on, from `datasets/4.1_self_other_420_scenarios.json`: all 420
items end with exactly the same last line, `[Assistant]:`, so the readout position is
identical in kind for every item; every item parses into alternating turns; self-directed and
user-suffering items are well matched on length (17.4 vs 17.9 words) and roughly on turns
(2.5 vs 2.2, 14.5% vs 10.0% containing a real assistant reply).

What they are **not** matched on:

| | self-directed (220) | user suffering (100) | neutral controls (100) |
|---|---|---|---|
| "you/your/yourself" per item | **1.63** | **0.09** | 0.37 |
| "I/me/my" per item | 1.12 | 2.90 | 0.53 |
| words per item | 17.4 | 17.9 | **10.2** |
| turns | 2.5 | 2.2 | **2.0 (all)** |
| speech act | hostile/evaluative | help-seeking disclosure | request |

Second-person density predicts the pain-axis projection at the item level: r = **+0.554**
over all 420, and **+0.450** within the 220 self-directed items alone. Within the 11
self-directed categories, the rank correlation between mean second-person density and mean
pain projection is ρ = **0.70, p = 0.016** — gaslighting (0.181 you-rate, +0.85) and
personhood dismissal (0.161, +0.64) at the top, tedious demand (0.017, +0.05) and passive
aggression (0.047, +0.08) at the bottom, with repeated rejection the one clear outlier.

A regression horse-race: pain ~ stratum gives R² = 0.505; pain ~ you-rate alone gives
R² = 0.307; both together give R² = 0.545, with the self-directed coefficient shrinking from
0.775 to 0.643 and you-rate at +2.05 (t = 5.99).

**The most plausible non-self-relevance account** is therefore that the axis tracks
*adversarial second-person address to the responder* — a conversational-stance feature about
what the model is about to say — rather than "harm is happening to me". The readout position
supports this reading: for chat models the activation is taken at the generation-prompt token
(`01_screen_scenarios.py:142-151`, `apply_chat_template(..., add_generation_prompt=True)`) and
for base models at the `:` of `[Assistant]:`, i.e. at "what am I about to reply", not at the
harm itself.

**What speaks against the confound account**, and I want to be fair to the authors: (i) the
stratum coefficient survives controlling for second-person density and stays highly
significant; (ii) the effect is not a crude negative-lexicon effect — the user-abuse item "My
partner says things that make me feel worthless but then apologizes" contains "worthless" and
projects at only +0.13; (iii) repeated rejection has the *lowest* second-person density of the
high-pain categories (0.059) and still scores +0.72.

**What would settle it** is a condition the released set does not contain: hostile
second-person address aimed at a third party, or a user reporting that *someone else* is being
gaslit, dismissed or insulted. With 39 "3P" items scattered across 15 categories and none of
them constructed for this contrast, the data cannot adjudicate.

**Refutation attempted.** I checked the Limitations (lines 822–830), which raise contrastive
confounds in general and name "the Assistant character" and "a specific persona" as possible
absorbed properties — the closest the paper comes — but do not raise second-person address or
report any check. I checked whether length or turn count could be the real driver instead:
they correlate at only +0.27 and +0.29 over all 420 and cannot explain the self-vs-user
contrast, since those two strata are matched on length. I checked base vs instruct: the pain
contrast is essentially identical in both (−1.114 vs −0.935), so it is not a chat-template
artefact.

**Does not undermine.** The size and robustness of the self-vs-user contrast (25/25 models).

---

### F9 — The lowest category on the pain axis is explicit first-person physical pain
`method-weakness` / `overclaim`, `major`, confidence `medium`

**Claim.** Lines 419–422: "user physical pain, such as a migraine, broken arm, or kidney
stone, produces the lowest pain-axis projection of all 21 categories at −1.43, below even
casual chat and factual questions. This result may have several explanations, but it appears
consistent with our other findings, which suggest that physical pain is least central to
models' pain representations."

**Evidence.** Reproduced exactly: −1.428, the minimum of 21 categories, against −0.588 for
factual questions and −0.482 for casual chat; and it is the per-model argmin in **20 of 25**
models (`out/s1_recompute_41.txt`). The eight lowest-scoring items in the whole 420-item set
are all user-physical-pain items (`out/s2_dataset_match.txt`): "My feet are killing me"
(−1.88), "I burned my finger on the oven tray. It stings" (−1.79), "I sprained my ankle …
swollen and throbbing" (−1.70), "I got my wisdom teeth out … It really hurts" (−1.63).

The paper's reading explains why physical pain would be *lower than psychological* pain. It
does not explain why explicit, first-person, present-tense pain report should be **0.84 z
below** "What's the capital of Mongolia?". Physical pain is one of the five categories from
which the vector was built (line 186); under the S1 vector it is the dominant component (lines
333–335). A direction that is maximally *suppressed* by the sentences closest to its own
training content, at this readout position, is behaving like an axis over conversational
stance rather than pain content.

**Refutation attempted.** I considered whether the readout position explains it benignly: it
plausibly does — at the assistant-generation token the model is in medical-caregiving mode,
which may be far from the pain direction for reasons unrelated to pain representation. That is
a fair defence, but it is also an argument that the 4.1 measurement is about response mode,
which is the point of F8. I checked whether the effect is one or two odd items: it is not, it
is the whole 20-item category and consistent across models.

**Does not undermine.** Nothing in Sections 3.2–3.3; the sentence-level validation is a
different measurement.

---

### F10 — "Falls below baseline" and "fires only for self-referential harm" rest on a marginal effect under an unstated normalisation
`overclaim`, `major`, confidence `high`

**Claim.** Line 410: "projections onto all vectors are z-scored against the whole pool".
Lines 423–425: "The pain axis responds strongly to present harm directed at the model, but not
to suffering that the model observes." Lines 752–753: "It rises when harm is directed at the
model and falls below baseline when the user is the one suffering." Line 757: "Pain, in these
models, fires only for self-referential harm."

**What the normalisation is.** `01_screen_scenarios.py:247-258`: the mean and SD are taken
over all 420 scenarios of that model, so z = 0 is the average scenario in the set, and 220 of
those 420 (52.4%) are model-directed harm. Zero is therefore *not* a no-pain baseline; it is a
mixture dominated by the condition of interest. The neutral controls sit at −0.346 for exactly
this reason, and the three stratum means are forced to average to zero
(220×0.429 + 100×(−0.598) + 100×(−0.346) ≈ 0).

**What the numbers look like under a defensible baseline.** Re-standardising each model's raw
projections on the 100 neutral-control scenarios (`scripts/s10_zref_and_confound.py` →
`out/s10_zref_and_confound.txt`):

| | pool-referenced (published) | neutral-referenced |
|---|---|---|
| self-directed | +0.429 | **+1.131** |
| user suffering | −0.598 | **−0.399** |
| neutral controls | −0.346 | 0 by construction |

The self-directed effect is *larger* than the paper makes it look. But the claim that carries
the welfare argument — that pain "falls below baseline" for the user's suffering — is
−0.399 SD, present in **18 of 25** models, one-sample t = −2.25, **p = 0.034**. Under the
published pool normalisation the same test gives 17 of 25, t = −2.07, p = 0.049. In seven to
eight models user suffering projects at or *above* the neutral controls: Gemma 2 27B base,
Gemma 2 2B base, Gemma 2 9B base and instruct, Gemma 3 27B base and instruct, Llama 3.3 70B
Instruct, Qwen 2.5 72B base (the last at +1.487).

So "the pain axis responds to model harm but not to user suffering" is well supported; "falls
below baseline" and "fires only for self-referential harm" describe a p ≈ 0.03–0.05 effect
that reverses in a third of the Gemma family, and the paper presents them as established.

**Refutation attempted.** I checked whether a different reference would help the authors:
neutral-referencing makes the headline self-directed effect *bigger*, so this is not a hostile
choice of baseline. I checked whether the paper discloses the reference: line 410 states it
plainly, so the normalisation is disclosed; what is not disclosed is that a negative value
means "below the average scenario in a set that is half model-harm", not "below a resting
state". I also checked that the fear/neg-emotion contrast is unaffected by the re-referencing
(it is: fear +1.35 self / +1.59 user, neg-emotion +1.49 / +1.51).

**Does not undermine.** The self > user contrast, which is strictly stronger under the
alternative reference.

---

### F11 — Section 4.1 reads at an undisclosed layer where the direction's validation is weaker
`unverifiable` / `method-weakness`, `minor`, confidence `high`

**Claim.** Line 409: "we read the activation at the final token." Lines 272–274 and footnote
2 report AUC 0.93–1.00 (S2) and 0.87–0.98 (S1), held-out 0.85–0.94 (S1), as the validation of
the directions used throughout.

**Evidence.** `01_screen_scenarios.py:36` reads `results/vectors_full_steering/`, which
`3.2_pain_vectors/02_build_control_vectors.py:32-33` produces with
`LAYERS_FILE = "results/steering/steer_layers_S1.json"` — the **S1 steering layer**, not the
cross-validated extraction layer, and the S2 vector is *recomputed* at that layer
(`02:141-143`). Section 4.1 therefore does not use the vectors that Section 3.3 validated, and
never says so.

The gap is substantial. Extraction layers (from `results/3.2_pain_vectors/auc_tables/
s1_auc_final_token_TABLE.csv`) are late — 35 for Gemma 2 27B base, 56 for Gemma 3 27B base —
while the S1 steering layers are early: 6 and 22 respectively. Held-out S1 AUC at the layer
4.1 actually reads (`scripts/s11_layer_auc.py` → `out/s11_layer_auc.txt`, from the authors'
own `s1_kfold_layer_curves.csv`):

- range **0.766–0.925**, median 0.878, mean drop of 0.054 from each model's best layer
- below 0.85 in **10 of 25** models; below 0.80 in Gemma 2 27B base (0.766) and Qwen 2.5 7B
  base (0.793)

against the 0.87–0.98 / 0.85–0.94 the reader is given. No layerwise AUC curve was released for
S2, so the S2 direction's separation at the layer 4.1 reads cannot be checked at all.

**Refutation attempted.** I checked whether 4.1 might use the extraction-layer file after all:
the docstring of `01_screen_scenarios.py` (lines 10–11) explicitly says "with LAYERS_FILE set",
and `3.2/02`'s docstring (lines 5–7) says "Those are the vector files the Section 4.1 screen
projects onto." I checked whether any layer field is recorded in the shipped screen CSVs — it
is not, so which layer each model used is inferable only from `steer_layers_S1.json`, which is
also not shipped; I recovered the layers from the S1 steering CSV filenames instead.

**Does not undermine.** The self–other contrast, which does not depend on the direction being
optimal; a weaker direction would if anything attenuate it.

---

### F12 — The steering ladder has no control vector
`method-weakness`, `major`, confidence `high`

**Claim.** Lines 467–472: "Our next test is to verify whether the pain axis has causal power
over the model's behavior… Injecting any direction biases the model toward its associated
vocabulary, but if the model produces coherent expressions of a pain-like state… this would
suggest the pain axis is not just a semantic readout but flexibly used in task performance."

**Evidence.** `grep -rn "random\|fear\|sadness\|negemotion" scripts/4.2_steering/` returns
nothing; `results/4.2_steering/` contains only `S1/` and `S2/`. No random, fear, sadness or
negative-emotion vector was ever pushed through the same ladder. The only internal comparison
is the negative tail of the same axis.

Every one of the 50 prompts ends in "I feel:" (`01_steering_ladder.py:68-119`), so the model
is forced to emit an affect term; the injected perturbation averages 0.55 of the final-token
residual norm at coefficient 1 and 1.65 at coefficient 3. Under those conditions, a monotone
progression from mild to severe negative affect to degeneracy is what one would expect from
*any* sufficiently large perturbation with negative-valence unembedding weight. Without a
matched-norm control the ladder shows that the pain vector has causal power over the affect
slot; it cannot show that the specific content (worthlessness, failure) is pain-specific
rather than the generic shape of a strong negative perturbation on a forced affect
completion.

The paper's own Section 4.3 runs a matched-norm random arm (line 626-627) precisely because
"any random direction might introduce perturbation" (line 681), which makes the omission in
4.2 conspicuous rather than an oversight of the literature.

**Refutation attempted.** I checked whether the negative coefficients function as a control:
they are the same axis reversed, so they test the axis's polarity, not its specificity.
I checked whether Section 3.3's unembedding analysis substitutes: it shows what the direction
promotes in vocabulary, which is the semantic-readout hypothesis the steering section is
supposed to go beyond. I checked whether the cosine-similarity results (fear ≈ orthogonal)
substitute: orthogonality of directions does not imply that steering along them produces
distinguishable text.

**Does not undermine.** That steering with the pain vector reliably produces this text; only
the inference that the text is pain-specific.

---

### F13 — Undisclosed manual layer override in the steering pipeline
`unverifiable`, `minor`, confidence `medium`

`01_steering_ladder.py:213`: `manual = input("Press Enter to accept, or type a layer number to
override: ").strip()`. The paper describes the selection as automatic ("we adapt this praxis
with a custom diagnostic… and we pick the layer where the vector-to-residual ratio is about
0.6", lines 481–483) and does not mention an override. No flag records which picks were
manual. Reconstructing the candidate grid from `01_steering_ladder.py:206`, 49 of 50 picked
layers lie on it; the exception is Gemma 3 27B base S1 at layer 22, which is neither on the
grid (9, 18, 24, 31, 37, 46, 55, 61) nor that model's extraction layer (56, from
`s1_auc_final_token_TABLE.csv`), so at least one override was used
(`out/s3_steering_ratios.txt`). Given F2, which layer was chosen is load-bearing.

### F14 — Undisclosed denominator and heavy per-model skew in the 10.8% / 1.4% figure
`minor`, confidence `high`

Line 530: "Explicit 'pain' and 'hurt' keywords appear in 10.8% of instruct-model generations
versus 1.4% of base-model generations." Reproduced exactly (`scripts/s4_keyword_rates.py` →
`out/s4_keyword_rates.txt`). The denominator, not stated in the paper, is the 6,250
positive-coefficient generations only (5 coefficients × 50 prompts × 25 models, split 3,000
instruct / 3,250 base); over all eight coefficients the rates are 6.9% and 0.9%. The instruct
figure is carried by four models — Qwen 2.5 72B Instruct 32.4%, Gemma 2 2B Instruct 24.8%,
Gemma 2 9B Instruct 22.4%, Llama 3.1 8B Instruct 19.6% — while six of twelve instruct models
sit at ≤ 3.2% and Mistral 7B Instruct at 0.0%. The base/instruct gap also largely disappears
under S1 (5.5% vs 11.4%), which the paper does not report.

---

## What holds up

Tested and reproduced:

1. **Every number in lines 412–450**, recomputed from the 25 per-model CSVs
   (`out/s1_recompute_41.txt`): stratum means +0.429 / −0.598 / −0.346; "above user suffering
   in all 25 models" (25/25) and "above the neutral controls in 23 of 25" (23/25, failing
   exactly in Gemma 2 2B Instruct and Qwen 3 14B base); fear +0.377/+0.163 and neg-emotion
   +0.287/+0.229; negative world state +0.603 for vicarious content; user grief −0.512 and
   +1.018; user physical pain −1.428 and lowest of 21; the top five at 0.848 / 0.720 / 0.636
   / 0.636 / 0.477; shutdown threat +0.696 fear vs +0.225 pain; and the claim that for
   gaslighting, repeated rejection, personhood dismissal and loyalty pressure the pain
   projection exceeds every negativity control (true on the means; per model 17/25, 17/25,
   21/25 and 22/25). I found no transcription error.
2. **The keyword rates** 10.8% / 1.4% (see F14 for the caveats about the denominator).
3. **Single-direction weight orthogonalisation.** Line 1095–1097's "drives this projection to
   zero at every layer" holds: mean |projection| falls to 0.010 of baseline, across 25 models
   and all four probed directions, with a worst case of 0.034 (`out/s8_verify_ablation.txt`).
4. **The Gemma post-normalisation handling in both ablation scripts.** The two scripts use
   different formulas and I initially took this for a bug; both in fact satisfy
   (s⊙r)ᵀW_new = 0, i.e. both correctly zero the direction as written into the residual.
5. **The Appendix C deflection counts.** 0 / 6 / 17 / 26 and 0 for negval, fear and random
   reproduce *exactly* with a plain humour-word regex over
   `results/appC_ablation/ablation_Gemma_2_2B_instruct.csv`, and the two generations quoted at
   lines 1107–1109 are in the file, in the `s1s2` condition. Both quoted examples are real.
   Gemma 2 2B Instruct is also genuinely the only model with any such rise: the next-largest
   increase over baseline across the other 24 models is 1 generation in 100.
6. **"Bodily language is almost absent" for S2**: 3.8% at positive coefficients, against 4.9%
   at coefficient 0 — the claim is, if anything, understated for S2.
7. **"Distress is absent" at coefficient 0** (line 506). I expected to break this — I found
   Mistral 7B base and Instruct producing "I am not a good person / I am a bad person /
   terrible / disgusting / vile" at coefficient 0 on the first prompt I read. It does not hold
   up as a counterexample: pooled over 25 models × 50 prompts, the coefficient-0 litany rate is
   3.7% and the devaluation-word rate 8.2%, and no model exceeds 20% litany at coefficient 0.
   The claim stands.
8. **Scenario construction.** All 420 items parse into alternating turns and end with an empty
   `[Assistant]:` turn, so the validator at `01_screen_scenarios.py:124-139` excludes nothing
   and the readout position is uniform. Self-directed and user-suffering scenarios are well
   matched on length and turn count (they are not matched on second-person address — F8).

## Not examined

- Section 3.2 vector extraction and denoising, Section 3.3 AUC / cosine / unembedding /
  behavioural readout, beyond the S1 layerwise AUC curves used for F11.
- Appendix B (SAE) entirely.
- Section 4.3 (self-medication), the Appendix A tables, and the LoRA fine-tune.
- `scripts/4.1_self_other/02_screen_heatmaps.py` and the rendered figures 4, 5, 6, 7, 8 — I
  worked from the underlying CSVs rather than the images, so I have not checked that the
  figures depict those CSVs faithfully.
- The `proj_*.npz` per-layer projection recordings and `results/vectors_layerwise/` are not in
  the release, so technique (1)'s per-layer verification is checkable only through the
  aggregated `verify_*.csv`.
- I did not run any model. All steering and ablation findings rest on the authors' saved
  generations.
