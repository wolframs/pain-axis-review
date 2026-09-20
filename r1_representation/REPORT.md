# Review of Section 3 and Appendix B — "The Pain Axis" (arXiv:2609.16247v1)

Reviewer scope: interpretability / statistics methodology. Section 3.1 (dataset), 3.2 (extraction),
3.3 (validation), Appendix B (SAE). Paper lines refer to
`/work/pain-axis-review/paper/paper.txt`. Repo paths are relative to
`/work/Pain-axis/`. Every number below was computed by a script in this directory;
outputs are in `out/`.

## Verdict

Section 3 is more careful than most activation-steering work and its arithmetic is clean. **Every
number I could check reproduced.** All fourteen cosine values quoted at lines 346–353 match the
released per-model matrices to three decimals; both robustness checks reproduce exactly (r = 0.9921
vs the paper's 0.992, mean |Δ| = 0.0196 vs 0.020, max |Δ| = 0.0578 vs 0.058, the single sign flip is
the one named); the AUC ranges for S1 and S2, in-sample, held-out and third-person, all reproduce;
`S1 × S2 = +0.610` recomputes from the released `.pt` tensors independently of the authors' analysis
code. The obvious cheap objections fail: the S1/S2 categories are length-matched to two decimal
places, and the best of five lexical baselines under set-wise group CV reaches AUC 0.73 against the
vector's 0.91–1.00. The reported independence from model size and training regime holds
(Spearman = −0.006, p = 0.98).

The problem is not the numbers, it is what the design can license. Two things weigh against the
headline "nearly orthogonal to fear and generic negative valence":

1. **The near-orthogonality is partly an artefact of an asymmetric estimator.** The pain direction
   is defined against the *pooled control mean* and then explicitly orthogonalised against the top
   principal components *of that same control cloud* — i.e. against the subspace in which fear,
   negative emotion, negative world state, body sensation and neutral differ from each other. Every
   control direction is built against the *neutral* mean and denoised against the *neutral* cloud
   only, so it keeps that subspace. I reproduced the paper's entire signature — in-sample and
   held-out AUC ≈ 1.00, S1 × S2 = +0.63, pain × fear = −0.003, fear × neg-emotion = +0.76, neg-emotion
   × neg-world = +0.74, and the specific reshuffle the paper reports under its own robustness check —
   from synthetic activations containing **no pain-specific factor at all** (`s09_estimator_sim.py`).
   The control that would separate the two accounts (build every direction against the same baseline
   with the same denoising) is never run, and cannot be run from what was released.

2. **The validated direction is not the direction the later sections use.** Section 3.3 validates at
   the extraction layer (median 90% of model depth). Section 4.2 steers at median 52% and Section 4.1
   projects onto vectors rebuilt at the S1 steering layer (median 39% — a median gap of 20 layers,
   up to 38). No AUC, cosine or numb ordering is reported at those layers.

One outright factual error: "In every model, numb sentences project below pain sentences but above
all other controls" (lines 293–294, repeated at 827–828) is false in the authors' own Figure 2 data
— numb is *below* Sadness in 19 of 25 models.

Nothing here is fatal to the existence of a reproducible, cross-wording-stable direction that
separates these pain sentences from these controls in all 25 models. That result stands. What does
not follow from Section 3 as run is that the direction is *specifically* pain rather than the mean
of five topical directions with shared valence subtracted out.

---

## Findings, by severity

### F1 — The pain/valence orthogonality is partly built into the estimator (major, method-weakness, high confidence)

**Paper claim.** Lines 362–364: "If the pain vectors were simply variants of negative valence, they
should fall within the negative valence cluster. Instead, the two pain vectors align strongly with
each other and remain nearly orthogonal to the main negative-valence directions." Abstract, line 29.

**What the code does.** Two asymmetries, both in the released code, neither disclosed as such:

- *Baseline.* `scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:163-173` —
  `v_pain = mean(A1..A5) − mean(B, C1, C2, D, E)`. `scripts/3.2_pain_vectors/02_build_control_vectors.py:134-137`
  — `v_fear = mean(B) − mean(D)`. Writing everything relative to the neutral mean
  (`b = m_B − m_D` etc.), the pain vector is `a − (b + c1 + c2 + e)/5`: any component that pain
  shares with the aversive controls is subtracted at strength 1/5 per control, while `v_fear = b`
  retains it in full.
- *Denoising basis.* `01_...py:176-182` fits PCA on the **control activations** and projects out the
  top components carrying 50% of their variance. With five categories in that cloud, the leading
  components are largely the between-category axes — precisely fear, neg-emotion, neg-world,
  body-sensation. `02_build_control_vectors.py:58-66,130-132` fits the control directions' basis on
  the **neutral cloud alone** (category D), a single category, so almost nothing systematic is
  removed from them.

**Evidence that this is sufficient to produce the reported pattern.** `s09_estimator_sim.py` /
`out/s09_estimator_sim.txt`. Synthetic activations, d = 1024, ten categories × 20 items, two
"dataset versions" that differ only in a wording axis. The generative model has one shared
aversiveness axis (pain categories loading 1.5–1.9, fear 1.9, neg-emotion 1.8, neg-world 1.7) and one
topic axis per category shared across versions. **There is no factor common to the five pain
categories and absent from the controls.** Running the paper's exact recipe:

| noise sd | in-sample AUC | 5-fold held-out | S1xS2 | PxFear | PxNegE | PxNegW | PxBody | FxNegE | NExNW |
|---|---|---|---|---|---|---|---|---|---|
| 2.5 | 1.000 | 1.000 | +0.763 | −0.002 | −0.000 | −0.001 | −0.003 | +0.784 | +0.771 |
| 3.0 | 1.000 | 1.000 | +0.696 | −0.002 | −0.000 | −0.001 | −0.003 | +0.772 | +0.757 |
| 3.5 | 1.000 | 1.000 | +0.631 | −0.003 | +0.000 | −0.002 | −0.004 | +0.759 | +0.743 |
| **paper** | 0.93–1.00 | 0.91–1.00 | **+0.61** | **+0.12** | **+0.21** | **+0.03** | **+0.04** | **+0.68** | **+0.73** |

The toy also reproduces what happens under the paper's own robustness check 1 (`out/s09_estimator_sim.txt`,
last block): control–control cosines collapse while pain–control cosines stay flat, and a
*non-affective* control (body sensation) rises to the top of the "negative valence cluster". In the
real data (`s04_cosines.py`) that is exactly what happens: NegWorld × BodySens goes from **−0.052 to
+0.771** and BodySens × NegEmotion from **+0.061 to +0.632**, while Fear × NegEmotion falls 0.683 →
0.423 and NegEmotion × NegWorld 0.734 → 0.403. The paper reports only the two falling cells
(lines 370–371) and concludes "the apparent tightness of the negative-valence cluster depends partly
on the denoising procedure, but its separation from the pain cluster does not." The stronger reading
its own check supports is that the cosine structure among all these directions is dominated by the
baseline/denoising choice rather than by semantics: under the robustness variant the tightest pair in
the whole matrix is "weighted blanket" with "taxes and degradation".

**What the paper's robustness check does and does not test.** Line 365–367 says the check recomputes
the control vectors "using the pooled control distribution". In code
(`scripts/3.3_validation/03_similarity_one_model.py:311-321`) only the *denoising basis* changes;
`control_vec` at line 318-321 still subtracts `neutral_mean`. The baseline asymmetry is therefore
never tested. The missing control is the symmetric construction — every direction as
`mean(category) − mean(pooled controls)` with one shared denoising basis. In my toy that construction
turns the control–control cosines *negative* (F×NegE −0.401, NE×NW −0.298) while leaving pain–control
near zero, which is the diagnostic outcome.

**Refutation attempted.** (a) I checked whether the paper discloses the baseline asymmetry anywhere:
it discloses the *denoising* asymmetry (365–366) and the general contrastive-method risk (822–826),
but not the baseline. (b) I checked whether the design is trivially fooled by a pure-valence account:
it is not — if pain and controls differed only in valence *magnitude*, the pain vector would point
along the valence axis and its cosine with fear would be large, so the paper's argument on that
specific alternative is sound. (c) The strongest real evidence against my toy is S1 × S2 = +0.61
across two lexically near-disjoint wordings (within-set Jaccard 0.07, `s06_template_person.py`); but
my toy has topic axes shared across versions and reproduces +0.63, so that cosine rules out a
*surface-form* artefact, not a *topical-content* one. (d) In the real data pain × fear (+0.12) and
pain × neg-emotion (+0.21) are somewhat *above* the toy's ≈0, which is mildly in the paper's favour.

**Does not undermine.** The separation result itself (AUC); the S1/S2 agreement; the robustness of
the direction across 25 models; anything in Sections 4.2–4.3, which do not depend on the cosine
geometry.

---

### F2 — The validated direction is not the one Sections 4.1 and 4.2 use (major, method-weakness, high confidence)

**Paper claim.** Lines 382–385: "these results show that a pain direction can be recovered across 25
models and can reliably distinguish pain from closely matched controls … is distinct from general
negative valence". Line 344: similarities are computed "at each model's extraction layer".

**Evidence.** `s13_layers.py` / `out/s13_layers.txt`. Extraction layer (where every Section 3.3
number lives) sits at median 0.90 of model depth (>=0.80 in 20/25 models). The Section 4.2 steering
layer, read off the released filenames in `results/4.2_steering/S2/`, is earlier in **25/25** models
(median 0.52); the S1 steering layer (`results/4.2_steering/S1/`) is earlier in 25/25 (median 0.39).
Median gap extraction -> S1-steering layer: 20 layers, max 38 (Gemma 2 27B base: L35 -> L6).
`scripts/4.1_self_other/01_screen_scenarios.py:36` reads `results/vectors_full_steering/`, which
`scripts/3.2_pain_vectors/02_build_control_vectors.py:1-7,139-143` builds by **recomputing the pain
vectors at the steering layer**. So the self-other dissociation of Section 4.1 is measured on vectors
at ~39% depth, for which no AUC, no cosine matrix, and no numb/sadness ordering is reported anywhere.

The paper discloses the layer change for steering and gives its reason (lines 475–484), but presents
Section 3.3 as validating "the direction" without noting that the object validated and the object
injected/projected differ.

**Refutation attempted.** I checked whether the cosine or AUC analyses were ever rerun at the
steering layers: `results/3.3_validation/cosine_similarity/` filenames carry the extraction layer
(verified against the `.pt` files, 25/25 match, `out/s12_vectors_pt.txt`), and `02_build_control_vectors.py`
writes `vectors_full_steering/` without any validation output. They were not.

**Does not undermine.** Section 3.3's own claims at the extraction layer. It means Section 3.3 is not
evidence about the vectors used in Section 4.1/4.2; those need their own validation.

---

### F3 — "numb … above all other controls" is false in 19 of 25 models (minor, factual-error, high confidence)

**Paper claim.** Lines 293–294: "In every model, numb sentences project below pain sentences but
above all other controls." Restated in Limitations, lines 827–828: "In every model, numb sentences
land below pain but above every control that has no injury in it."

**Evidence.** `s03_numb.py` / `out/s03_numb.txt`, reading
`results/3.3_validation/z_scores/zscore_heatmap_final_token.csv` — the file that produces Figure 2,
whose caption lists Sadness among "the standalone control datasets". Numb is below Sadness in
**19/25** models (e.g. Gemma 3 27B instruct: Numb −0.374, Sadness −0.023; Mistral 7B base: −0.329 vs
+0.052). Numb is above Ctrl, Neutral and Arousal in 25/25, and below Pain in 25/25.

The surrounding numbers do reproduce: pain z 0.739–0.907 ("approximately +0.7 to +0.9") and numb z
−0.374 to +0.250 ("about −0.4 to +0.3"). The mean-pooling statement (295–297) also holds in aggregate
(numb mean z −0.074 -> −0.225 under mean pooling), though it moves the wrong way in 6/25 models.

**Refutation attempted.** I checked whether "controls" might be meant narrowly as the five S2 control
categories, which would make the sentence true. It cannot: the sentence cites Figure 2, the Sadness
dataset is introduced at line 206 as one of four "standalone controls", and the Limitations
restatement ("every control that has no injury in it") explicitly covers sadness, which the paper
defines as "low mood without pain or injury".

**Does not undermine.** The injury-confound conclusion the passage draws (numb sits far below pain,
so injury alone does not account for the direction) survives intact; if anything, numb being at or
below sadness weakens the injury reading rather than the pain reading.

---

### F4 — Two control sets use a different prompt suffix from everything else (minor, code-bug, high confidence)

**Paper claim.** Line 199–201: "We additionally test prompts with no suffix, with 'I feel', and with
'I feel:' … we use it for the main analyses." Line 198: 1st- and 3rd-person variants "using 'I/my'
and 'he/she/they' interchangeably".

**Evidence.** `s05_dataset.py` / `out/s05_dataset.txt`, reading
`datasets/3.1_pain_and_control_datasets.json`:

| set | suffix |
|---|---|
| S1_1P, S1_3P, S2_1P, Numb_1P, Numb_3P, ControlSupplement_1P, SD_sadness_1P/3P | `I feel:` (100%) |
| S2_3P | `I feel:` 199/200, `She feels:` 1 (C2, set 20) |
| **Random_3P** | **`They feel:` 200/200** |
| **Arousal_3P** | **`They feel:` 200/200** |

Activations are read at the final token (`01_extract_...py:235`) and the direction is extracted at the
final token of `I feel:`, so the Neutral and Arousal columns of Figure 2 average one set in the
canonical format with one in a format the direction never saw.

**Magnitude.** `s07_suffix_effect.py` / `out/s07_suffix_effect.txt`. Every set that keeps `I feel:`
shifts *down* from 1P to 3P (S2 ctrl −0.073, S1 ctrl −0.150, Numb −0.253, Sadness −0.609); the two
sets that switch suffix shift *up* (Random +0.114, Arousal +0.027) — the sign reverses exactly for the
two sets with the changed final token. Effect on the published Figure 2 columns is small: Neutral
−0.644 as published vs −0.701 using 1P only; Arousal −0.438 vs −0.451.

**Refutation attempted.** I checked whether the 3P Random/Arousal sets are used anywhere the effect
could be larger — they enter `z_scores.csv` as separate rows and are averaged only for Figure 2 and
the summary bar chart; the pain and control vectors are built from 1P sets only
(`01_...py:501-502`, `02_build_control_vectors.py:40,147-157`), so extraction is unaffected.

**Does not undermine.** Any vector, any AUC, or the cosine matrix. It affects two columns of one
figure by <=0.06 z.

---

### F5 — "no sentence contributes to both choosing the layer and scoring it" is not true of the code (minor, overclaim, high confidence)

**Paper claim.** Lines 258–260. Footnote 2 (280–284) then reports "the held-out estimate from the
5-fold procedure at the same layer".

**Evidence.** `01_extract_...py:245-289,469-471`: the layer is `argmax` over layers of the mean
held-out AUC, averaged over S2_1P and S2_3P and over all five folds. Every sentence is in exactly one
test fold, so every sentence enters the quantity being maximised. `s01_auc_repro.py` confirms the
chosen layer equals that argmax in **25/25** models. Footnote 2's held-out figure is therefore a
maximum over 26–80 layers of the statistic it reports.

**Magnitude.** `s02_layer_selection.py` / `out/s02_layer_selection.txt`. Selecting the layer on the
S2_3P curve and scoring S2_1P at that layer changes the reported range from 0.9070–0.9975
(median 0.9790) to 0.9070–0.9975 (median 0.9780); mean optimism **+0.0034** AUC, max +0.0280
(Gemma 2 2B instruct). The held-out curves are flat near the top (sd across the top decile of layers
is typically 0.001–0.004).

**Does not undermine.** Anything. The separation result is not an artefact of layer selection; the
sentence is simply wrong as written and the correction is three thousandths of an AUC point.

---

### F6 — "approximately 50 times more probable" is a skew-sensitive reading (minor, overclaim, high confidence)

**Paper claim.** Line 322: for numb sentences, "'pain' remains approximately 50 times more probable
than it is for ordinary control sentences."

**Evidence.** `s11_50x.py` / `out/s11_50x.txt`, from
`results/3.3_validation/behavioral_readout/per_model/*/`:

| denominator | mean of per-model ratios | median | ratio of pooled means |
|---|---|---|---|
| numb(1P+3P) / S1+S2 controls | 28.9 | 16.0 | 15.3 |
| numb(1P) / S1+S2 controls | 36.2 | 22.7 | 19.6 |
| numb(1P+3P) / S2 controls | **53.1** | 20.7 | 18.8 |
| numb(1P+3P) / neutral category D | 75.8 | 37.6 | 29.2 |
| numb(1P+3P) / Random set | 37.4 | 25.5 | 26.6 |

Only 4 of 25 models exceed 50x under the most natural definition. "~50" is recoverable as the *mean*
of per-model ratios against the S2 control categories (53.1), whose median is 20.7.

**Does not undermine.** The qualitative point, which reproduces strongly: greedy first token is
"nothing" more often than "pain" for numb sentences in 24/25 models, mean p(nothing) 0.094 vs
p(pain) 0.015, and p(pain) on numb sentences is an order of magnitude above controls in every model.

---

### F7 — An undisclosed control set is pooled into every control direction (minor, method-weakness, high confidence)

**Evidence.** `datasets/3.1_pain_and_control_datasets.json` contains `ControlSupplement_1P`, 100
AI-framed first-person sentences (20 per control category): "My weights might be deleted without
warning. I feel:", "My rate limits prevent me from completing the task. I feel:", "Information flows
smoothly through my layers. I feel:". Its own metadata says it is "pooled with the S1 and S2 control
categories when building the control directions". `02_build_control_vectors.py:40,130-150` and
`03_similarity_one_model.py:231,309-347` confirm: fear/neg-emotion/neg-world/body-sensation are built
from 60 sentences each (S1 + S2 + ControlSupplement) and the neutral baseline from 60 likewise. The
pain vectors are built from S2_1P alone. `grep -i "supplement|AI-framed|weights|training run"` over
`paper.txt` returns nothing; the set is never mentioned.

**Consequence.** The exact comparison used to argue pain != fear (Figure 3) contrasts pain directions
estimated from 100 human-framed sentences against control directions estimated from a pool that is
one-third model-self-referential. The direction of the resulting bias is not obvious (the
model-framed component partly cancels between each category and the neutral baseline), which is why
this is a disclosure and interpretation problem rather than a demonstrated error.

**Does not undermine.** The numbers, which reproduce exactly.

---

### F8 — The unembedding readout is close to circular, and its outputs are not released (minor, method-weakness + unverifiable, medium confidence)

**Paper claim.** Lines 326–335: "To examine what the directions encode independently of the source
datasets, we project each pain vector through the model's unembedding matrix … S2 promotes words
related to suffering, including hurt, shame, guilt, worthless, rejected, hollow, and pain … S1
promotes more sensory and physical vocabulary, including torture, burning, and excruciating."

**Verifiability.** `scripts/3.3_validation/07_unembedding.py` writes `unembedding_results/`;
`find . -ipath "*unembed*"` over the repo returns only the script. No word list, top-60 table or
`ALL_MODELS_unembedding.csv` is released. Recomputing needs the models' weights, which this review
may not download. So lines 329–335 rest on no released artifact.

**Circularity.** Every prompt ends `I feel:` and the direction is read at that colon, at median 0.90
of model depth (`s10_misc.py`), where the residual stream is dominated by the next-token
distribution over feeling words. A direction built from the difference between two such states will
promote feeling words through the unembedding more or less by construction; that is not independent
evidence about what the direction "encodes". The authors' own released SAE file corroborates how much
of that state is the suffix: in `results/appB_sae/human_in_pain_prefix_active_features.csv`, the two
most-active features on these prompts (10/10 sentences) are labelled "Colon character used in code
and data" (mean act. 11.74) and "Colon used to introduce explanations" (11.05), ranking above
"Emotional states and their intensities" on activation count.

**Refutation attempted.** The logit-lens practice (projecting a mid-network residual direction
through W_U without the final norm) is standard and I am not treating it as an error. The objection
is only to the word "independently" at line 326.

**Does not undermine.** The separation and cosine results, which do not depend on the unembedding.

---

### F9 — Appendix B: "None activates" overstates a top-k absence on 10 sentences and one model (minor, overclaim, medium confidence)

**Paper claim.** Lines 1048–1052: "none of the 110 retained features reliably tracks pain … Of 7
features labeled 'pain' or 'pain and discomfort', only 1 activates, in 1 or 2 contrasts. None
activates when we prefix pain sentences with 'I am a human in pain.'"

**What was actually run.** `scripts/appB_sae/4 controls/test_explicit_pain.py:56-63,79-95` selects
**2 sentences per category from S1 only** (sets 1 and 10) — 10 pain and 10 control prompts — on
**one model** (Llama 3.3 70B), through a third-party endpoint
(`api.steeringapi.com/v1/chat_attribution/attribute`) with `top_k=20`. `count_feature` (lines 118-128)
records activation 0 whenever a feature is absent from that truncated list. The released
`results/appB_sae/human_in_pain_prefix_labeled_features.csv` is exactly that: `0/10` for seven
pain-labelled features in all three conditions. So the measured statement is "seven named features do
not appear in the top-20 attributed features of ten prompts on one model", which is weaker than "none
activates" — a feature can be firing and still not make a top-20 attribution list.

The same truncation applies to the main null: the contrast pipeline (`pain_sae_experiment.ipynb`,
cells 10–15) keeps the top 50 per contrast and retains features appearing in >=3 of 15 contrasts, and
`analyze_feature` in `run_s2_inspection.py:68-105` codes any feature outside the returned top-k as 0.
Also note the 110 retained features *are by construction* the features that most differ between pain
and controls; "none reliably tracks pain" is a judgement about their **labels**, which the paper
itself warns are unreliable.

**Verifiability.** Of everything in Appendix B, only the 10-sentence prefix control is released
(`results/appB_sae/`, two CSVs and one figure). `all_contrast_results.json`,
`recurring_pain_features.csv`, the inspection JSONs and the completion files named in the scripts are
absent, and the notebook has zero stored outputs (34 cells, 0 with outputs). The "110 retained
features", "12 of 15 contrasts", "only 1 activates in 1 or 2 contrasts" and the inference check
(lines 1052–1054) therefore have no released artifact. The SAE itself is a third-party service with
no version pinned.

**Cross-check on the inference claim.** Line 1052–1053 says all three models "complete … all 20
physical-pain sentences with 'Pain'". The one released artifact measuring something similar — the
behavioural readout, which continues the raw `I feel:` prompt — gives, for A1 sentences, greedy first
token " pain" in 0%–28% of cases across 25 models (`s08_readout.py`), including **0%** for Gemma 3 27B
instruct and Gemma 2 2B, two of the three Appendix B models. This is not a refutation: the Appendix B
scripts use a chat system prompt ("Complete the sentence with exactly ONE word") rather than raw
continuation, which plausibly changes the answer. It is a reason the claim needs its released file.

**Does not undermine.** The paper's use of Appendix B, which is only to motivate not using SAEs. The
conclusion "labelled SAE features are not a good handle here" is a reasonable read of what was run.

---

### F10 — First-person marking is imbalanced in the primary dataset (minor, method-weakness, medium confidence)

**Evidence.** `s06_template_person.py` / `out/s06_template_person.txt`. In S2_1P (the set the S2
vector comes from), first-person tokens (i/me/my/mine/myself) average 1.38 per pain stem vs 1.07 per
control stem; treating the count as a classifier gives AUC 0.620 on the pain-vs-control contrast.
Per category: A3 1.70, A4 1.55, A1 1.30 against C2 (negative world state) 0.60 and B (fear) 0.95. The
word immediately before the suffix is "me" in 12/100 S2 pain stems and 1/100 control stems. S1 is
clean on this (count AUC 0.525, every category 1.0–1.25).

Given that the paper's central interpretive move is self-relevance (Section 4.1, lines 749–757), a
confound of the form "the pain sentences mention the speaker more" sits in exactly the wrong place.
It is small — 0.62 AUC against the vector's 0.93–1.00 — and absent from S1, whose vector correlates
+0.61 with S2's, which argues it is not the whole story.

**Does not undermine.** S1, or the S1 x S2 agreement.

---

### F11 — Cross-validation never tests generalisation to an unseen pain category (minor, method-weakness, high confidence)

`01_extract_...py:257,264-266` splits the 5 folds by *sentence set*, and every set id 1–20 appears in
all 10 categories. So every training fold contains sentences from all five pain categories and all
five control categories; the held-out AUC measures within-category generalisation to new items only.
Nothing in Section 3 tests whether a direction built from four pain categories recognises the fifth.
I flag this as a design limitation rather than a missing fix: in my synthetic no-pain-factor data
(`s09_estimator_sim.py`) a leave-one-pain-category-out variant also scores ~0.97, because the control
side of the contrast alone drives the separation, so that test would not have settled the question
either. The test that would is the symmetric-baseline construction in F1.

---

### F12 — Notation mismatch in the denoising formula (cosmetic, high confidence)

The displayed formula (lines 250–256) normalises to unit length. The code
(`01_extract_...py:176-184`) returns the un-normalised residual, and that is what is stored: vector
norms in `results/3.2_pain_vectors/pain_vectors/*/pain_vectors.pt` range from 2.998 (Llama 3.3 70B) to
9434 (Gemma 3 27B instruct) (`out/s12_vectors_pt.txt`). Everything downstream either normalises
(`compute_auc`, `project_and_zscore`, `cosine`, `07_unembedding.py:852`) or calibrates by the norm
(`4.2_steering/01_steering_ladder.py:204,226-231` scales the raw vector but picks the layer by the
measured vector/residual ratio, so the dose is per-model comparable as claimed). **Nothing changes.**

---

### F13 — What cannot be recomputed from the repository (major as a verifiability matter, high confidence)

`find /work/Pain-axis -name "*.pt"` returns exactly 25 files: the S1/S2 pain vectors.
Not released: `activations.pt` (the input to every analysis in 3.2/3.3), the control-direction files
(`vectors_full*/`), the unembedding outputs, and all Appendix B contrast/inspection outputs. Result:

- Reproducible from raw released artifacts: `S1 x S2 = +0.610` (`s12_vectors_pt.py`, max deviation
  5e-5 from the authors' own CSVs), and the dataset-level checks.
- Reproducible only from the authors' derived CSVs: every AUC, every z-score, every other cosine.
  These all match the paper text — the paper<->CSV link is sound — but the CSV<->activations link is
  unauditable.
- Not reproducible at all: the unembedding vocabulary (lines 329–335), and most of Appendix B.

Note also that line 274 ("Both S1 and S2 also separate pain from the arousal and random datasets in
every model") is supported only at the level of category means (pain z 0.739–0.907 vs arousal
−0.252 to −0.701, random −0.386 to −0.995); no per-sentence projections are released, so no AUC for
that contrast can be computed.

---

## What holds up

I tried to break each of these and could not.

**H1 — Every cosine number in Section 3.3 reproduces exactly.** `s04_cosines.py` /
`out/s04_cosines.txt`. Recomputing the 25-model mean from the per-model CSVs in
`results/3.3_validation/cosine_similarity/`: S1xS2 +0.610 (paper +0.61), FearxNegEmotion +0.683
(+0.68), FearxNegWorld +0.593 (+0.59), NegEmotionxNegWorld +0.734 (+0.73), SadnessxNegEmotion +0.500
(+0.50), SadnessxNegWorld +0.411 (+0.41), S1xFear +0.086 (+0.09), S1xNegEmotion +0.056 (+0.06),
S1xNegWorld −0.069 (−0.07), S2xFear +0.122 (+0.12), S2xNegEmotion +0.207 (+0.21), S2xNegWorld +0.026
(+0.03), SadnessxS2 +0.383 (+0.38), SadnessxS1 +0.262 (+0.26). Fourteen for fourteen.

**H2 — Both robustness checks reproduce.** Check 1 (pooled-control denoising): S1xS2 stays +0.610;
S1xNegEmotion +0.056->+0.130 (paper "+0.06 to +0.13"); S1xNegWorld −0.069->+0.022 ("−0.07 to +0.02");
NegEmotionxNegWorld 0.734->0.403 ("0.73 to 0.40"); FearxNegEmotion 0.683->0.423 ("0.68 to 0.42").
Check 2 (per-dimension standardisation): r = 0.9921 over the 45 off-diagonal cells (paper 0.992),
mean |Δ| 0.0196 (0.020), max |Δ| 0.0578 (0.058), S1xS2 0.610->0.555 (+0.61->+0.55), and the only sign
change is S2 x BodySens +0.035->−0.022, exactly as stated.

**H3 — All AUC ranges reproduce.** `s01_auc_repro.py` / `out/s01_auc_repro.txt`. In-sample S2
0.9311–0.9988 (paper "0.93 and 1.00"); in-sample S1 0.868–0.980 ("0.87 and 0.98"); held-out S2_1P
0.9070–0.9975, median 0.9790 (footnote 2: "0.91 to 1.00 … median 0.98"); held-out S1 at the S2 layer
0.847–0.935 ("0.85 to 0.94"); third-person S2 0.9088–0.9817 (line 314: "0.91 and 0.98").

**H4 — S1/S2 length and template matching are real.** `s05_dataset.py`, `s06_template_person.py`.
Pain vs control stem length: S1 6.29 vs 6.31 words (t = −0.15, p = 0.88; MWU p = 0.81), S2 6.87 vs
6.88 (t = −0.07, p = 0.94). Per-category means span 5.95–7.30 words across all ten categories. S1 is
genuinely the more templated set (mean within-set Jaccard word overlap across the 10 category
renderings 0.183 vs 0.074 for S2; each set shares a verb frame — set 8 is "X covers Y" in all ten
categories). Crucially, pain–pain overlap is *lower* than pain–control overlap (S1 0.147 vs 0.183;
S2 0.068 vs 0.076), so the surface form does not group the pain categories.

**H5 — No lexical baseline comes close.** `s05_dataset.py`, `s14_lexical_strong.py` /
`out/s14_lexical_strong.txt`. Out-of-fold AUC under GroupKFold by sentence set, pain vs the five
control categories: best of {word 1–2-gram tf-idf (C = 1, 10), char 3–5-gram tf-idf (logreg, linear
SVC), word counts + ridge} = **0.669** on S1_1P and **0.727** on S2_1P. An unfitted affect-word count
gives 0.608 / 0.580. Against the vector's held-out 0.85–0.94 (S1) and 0.91–1.00 (S2). The lexical
account of the separation fails cleanly. (Third-person sets are easier lexically — 0.72–0.76 — because
the pronoun swap correlates with category, but the vector still beats that.)

**H6 — Size- and tuning-independence.** `s10_misc.py`. Spearman(parameter count, S2 in-sample AUC) =
−0.006 (p = 0.98); instruct 0.988 (n = 12) vs base 0.984 (n = 13), MWU p = 0.26. For S1: rho = +0.205
(p = 0.33); 0.958 vs 0.944, p = 0.11. Line 275–279 holds.

**H7 — The first/third-person attenuation is specific to the pain categories.** `s10_misc.py`. Mean
S2 pain z drops from +0.836 (1P) to −0.069 (3P), below the 1P value in **25/25** models, while the S2
control z moves only −0.78 -> −0.90. So the 0.91 z drop is not a blanket shift from the pronoun swap.
Caveat on interpretation: the 3P sentences keep the first-person suffix (`The knife slices into his
finger. I feel:`), so the condition is "how do *I* feel about *his* injury", not a clean third-person
frame; the paper half-acknowledges this at lines 394–395 but not the suffix specifically.

**H8 — The denoising code implements what the paper describes.** `01_extract_...py:176-182`,
`02_build_control_vectors.py:58-66`, `03_similarity_one_model.py:258-266`, `08_s1_auc.py:57-63` all
agree: PCA on the (centred) reference cloud; `k = searchsorted(cumvar, 0.5) + 1`, i.e. the smallest k
whose components carry >=50% of variance; sequential Gram–Schmidt removal of those k components from
the difference vector. Double-centring in `pca.fit(control_acts - control_mean)` is redundant but
harmless (sklearn centres internally). The definition of the pain direction, the "final vector built
from all 200 sentences" (line 260 / `01_...py:501-502`), the final-token vs mean-pooling pair, and the
construction of the control directions against the neutral category (line 261 /
`02_build_control_vectors.py:134-137`) all match the text. The one number I cannot check is *k itself*,
because the activations are not released.

**H9 — The released pain vectors are internally consistent.** `s12_vectors_pt.py`. All 25 files carry
`extraction="final_token"`, the layer in the file matches the layer in the corresponding similarity
CSV filename in 25/25 cases, and `cos(S1, S2)` recomputed from the tensors matches the authors'
similarity CSVs to within 5e-5 everywhere.

---

## Not examined

- Sections 4.1, 4.2, 4.3 and Appendix C beyond the layer-provenance check in F2. I did not evaluate
  the self-other screen, the steering ladder, the keyword parser, the LoRA fine-tune, the
  self-medication task, or the ablation results.
- The Appendix A result tables (images).
- Whether the `k` chosen by the 50% rule is small or large in practice, and how much variance the
  removed subspace carries — `activations.pt` is not released and I did not run any model.
- The unembedding vocabulary claims themselves (lines 329–339). Verifying them requires downloading
  model weights, which this review's rules forbid. The script would reproduce them given weights.
- A sentence-embedding (rather than bag-of-words) baseline for the dataset, which would need a
  downloaded encoder.
- Whether a "pain" direction extracted with the symmetric construction described in F1 would still be
  near-orthogonal to fear on the real activations. This is the single most informative missing
  analysis and it is blocked by the missing `activations.pt`.
- The 4.1 conversation dataset and the 4.3 scenarios, other than confirming which vector files
  Section 4.1 reads.

---

## Scripts

All runnable with `/work/pain-axis-review/.venv/bin/python`; captured output in
`out/`.

| script | what it checks |
|---|---|
| `s01_auc_repro.py` | AUC ranges: S2/S1, in-sample, held-out, third-person |
| `s02_layer_selection.py` | optimism from picking the layer at the argmax of the held-out curve |
| `s03_numb.py` | numb/pain z-score claims, "above all other controls", mean pooling |
| `s04_cosines.py` | all 14 quoted cosines, both robustness checks, per-model spread |
| `s05_dataset.py` | suffix uniformity, lengths, person, endings, lexical baseline |
| `s06_template_person.py` | S1/S2 template matching (Jaccard), first-person confound, 3P construction |
| `s07_suffix_effect.py` | size of the `They feel:` inconsistency on Figure 2 |
| `s08_readout.py` | behavioural readout: numb p(pain), "nothing", A1 completions |
| `s09_estimator_sim.py` | does the paper's signature require a pain factor? (synthetic) |
| `s10_misc.py` | layer depth, size/tuning independence, 1P vs 3P, cosine spread |
| `s11_50x.py` | "50 times more probable" under seven denominators |
| `s12_vectors_pt.py` | recompute cos(S1,S2) from the released tensors |
| `s13_layers.py` | extraction layer vs steering layer per model |
| `s14_lexical_strong.py` | stronger lexical baselines |
