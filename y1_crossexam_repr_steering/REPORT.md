# Cross-examination of the representation / steering / ablation reviews

Scope: the twelve findings put to me, each re-derived with my own code against the rawest
available artifact. Paper line numbers refer to
`/work/pain-axis-review/paper/paper.txt`; repo paths to
`/work/Pain-axis/` @ 8d1649c (unmodified). Activations are the coordinator's
re-extraction in `gpu_repro/acts/`. Scripts in `scripts/` (see `scripts/README.md`), captured
output in `out/`, verdicts in `verdicts.csv`, findings in `findings.json`.

## Verdict

**The four reviewers are right far more often than they are wrong, and their arithmetic is
almost uniformly sound. But three of the twelve findings do not survive contact with the
evidence in the form they were written, and in the two places where reviewers disagree, one
side is measuring an artefact.**

The three corrections that matter:

1. **The "50 times" disagreement settles against x2.** x2 F7's 47x comes from parsing the
   *truncated* top-20 string in `BIG_TABLE.csv`. Truncation discards **79.1% of the
   denominator** (control sentences, where " pain" rarely reaches rank 20) against **12.8% of
   the numerator**, so every ratio built that way is inflated. The release also ships the
   *full-softmax* probability of the " pain" token as the `p_pain` column of
   `behavioral_readout/per_model/*/*.csv`, with the token id in `vocab_info.json`. Using it,
   r1's numbers reproduce to the digit: 15.3x pooled / 16.0x median against the core controls,
   26.6x / 25.5x against Random. "~50" is recoverable only as the *mean* of per-model ratios
   against the S2 control categories (53.1, median 20.7, exceeded in 6 of 25 models).
   r1 CONFIRMED, x2 F7 REFUTED.

2. **The claim that Section 4.1's layer cannot be validated is false, and both technical
   reviewers made it.** r1 F2 ("No AUC, cosine or numb ordering is reported at those layers")
   and r2 F11 ("No layerwise AUC curve was released for S2, so the S2 direction's separation
   at the layer 4.1 reads cannot be checked at all") are wrong about the release.
   `results/3.2_pain_vectors/per_model/<model>/layer_curves.csv` ships held-out 5-fold AUC for
   S2_1P and S2_3P at **every** layer of all 25 models. At the layer Section 4.1 actually
   reads, the S2 direction's held-out AUC is **0.839-0.947, median 0.910** (against 0.907-0.998,
   median 0.979 at the extraction layer). The undisclosed layer swap is real and belongs in the
   text; the implied damage is not. A "major, unverifiable" becomes a disclosure defect with a
   measured and modest cost.

3. **"Pain is nearly orthogonal to fear" is an estimator artefact *as a cosine statement* and
   survives as a *discrimination* statement.** On real activations I can put numbers on the
   asymmetry nobody had measured: the PCA basis projected out of the pain vector contains
   **80-86%** of each raw aversive control direction but only **39%** of the raw pain direction
   (Gemma 2 2B instruct; Mistral 7B base gives 0.80/0.84/0.86/0.83). Under any construction that
   treats pain and controls alike, the inference at lines 362-364 fails: pain sits *inside* the
   negative-valence cluster and is more central to it than fear. But no construction is neutral
   — with a leave-one-out baseline the cluster inverts to -0.54 — so cosines are the wrong
   instrument. Out-of-sample and across wordings the paper wins: a direction built on one
   dataset version separates pain from fear on the *other* at AUC 0.91-0.99. r1 F1 is CONFIRMED
   as a criticism of the cosine argument, NARROWED as a criticism of the underlying claim.

On r1's simulation: it does what it says — no latent is shared by the five pain categories and
absent from the controls — but the mean of five per-category topic axes, held fixed across the
two "dataset versions", *functions* as a pain direction and is what produces the toy's S1xS2
cosine. "No pain factor at all" overstates it; the defensible statement is that the paper's
diagnostics cannot separate shared *pain semantics* from shared *per-category topical
idiosyncrasy*. That is still serious, and the two-sided leave-one-category-out test below is the
first thing in this package that bears on it directly.

Everything else holds, mostly with the reviewer's own wording; findings 3, 5, 6 and 7 need
narrowing, and the steering-ladder disagreement resolves in r2's favour on the paper's actual
claim.

---

## Finding by finding

### 1. "Pain is nearly orthogonal to fear/negative valence" is produced by the estimator
**r1 F1 + `gpu_repro` — `CONFIRMED-NARROWER`, major, high confidence**

**Code.** `01_extract_activations_and_pain_vectors.py:163-174` builds `mean(5 pain) -
mean(5 controls)`; `:176-184` projects out the top PCs carrying 50% of the *control cloud's*
variance. `02_build_control_vectors.py:134-137` builds each control as
`mean(category) - mean(neutral)`, denoised against the *neutral cloud alone* (`:131-132`).

**Is "built like the controls" a strawman? One of r1's charges is wrong.** The paper does not
hide the asymmetry. Lines 222-228 state the pooled-control contrast and give an explicit reason
("contrasting pain against all controls jointly subtracts what it shares with fear, negative
valence, bodily sensation, and negative events"), and lines 260-262 state the controls are built
"against the neutral category". Both halves are on the page. r1 F1's "neither disclosed as such"
is wrong and the charge should be re-filed: the choice is *disclosed and then reasoned from as
though it were not a choice*.

**The asymmetry is real and now measured** (`x1_geometry.py`, Gemma 2 2B instruct at the
authors' own extraction layer L25, k = 5 of d = 2304; a random direction would sit at 0.0022):

| raw direction, before denoising | fraction inside the basis projected OUT of the pain vector |
|---|---|
| **S2 pain** (mean pain - mean pooled controls) | **0.389** |
| Fear | 0.801 |
| Negative emotion | 0.859 |
| Negative world state | 0.853 |
| Bodily sensation | 0.756 |

Mistral 7B base at L26 (k = 7 of 4096): 0.797 / 0.839 / 0.864 / 0.833. The mirror is lopsided
too: the neutral-cloud basis removed from the *controls* takes out 11-28% of them and 11.6% of
pain. The denoising step removes roughly five times more of each aversive control direction
than of the pain direction. That is the mechanism, quantified.

**Constructions that treat everything alike** (Gemma 2 2B instruct; the pipeline reproduces the
authors' released matrix to max |Δ| = 0.0028, so this is not my code inventing geometry):

| pair | authors | all vs neutral, neutral basis | all vs neutral, pooled basis | all vs pooled-LOO, pooled basis | all vs neutral, no denoising |
|---|---|---|---|---|---|
| S1 x S2 | +0.611 | +0.945 | +0.888 | +0.795 | +0.932 |
| S2 x Fear | +0.144 | +0.696 | +0.591 | +0.303 | +0.757 |
| S2 x NegEmotion | +0.118 | +0.814 | +0.326 | -0.137 | +0.823 |
| S2 x NegWorld | -0.049 | +0.677 | +0.501 | -0.098 | +0.705 |
| Fear x NegEmotion | +0.705 | +0.705 | +0.440 | **-0.536** | +0.758 |
| NegEmotion x NegWorld | +0.781 | +0.781 | +0.415 | **-0.635** | +0.790 |

This cuts both ways and I want to be exact about it. Under a shared neutral baseline pain is not
merely inside the negative-valence cluster, it is *more central than fear*
(S2 x neg-emotion +0.81 > fear x neg-emotion +0.71). Under a leave-one-out baseline the cluster
does not exist — the control-control cosines go strongly negative, partly for a mechanical
reason (each control's reference contains the other controls). **Neither construction is
neutral.** The conclusion is not "pain is really negative valence". It is that cosines between
difference-of-means vectors built on different baselines carry almost no information about
semantics, and the paper's inference at 362-364 — "if the pain vectors were simply variants of
negative valence, they should fall within the negative valence cluster" — has no estimator under
which both antecedent and consequent are well defined. The authors' own robustness check does
not reach this: `03_similarity_one_model.py:120-145` changes only `basis`; `control_vec` still
subtracts `neutral_mean`, and in that variant the pain vectors are not recomputed at all
(`:141-145` reuses the saved ones), so the two sides still use different bases.

**A third construction, and it kills the cluster language outright.** Centre all ten category
means on the grand mean of the ten — no privileged baseline, no denoising, identical treatment
(`x8_supp_and_geometry2.py` [b], Gemma 2 2B instruct). Ten centred vectors sum to zero, so the
compositional null for any pair is -1/9 = -0.111, and essentially every observed value sits at
it: fear x neg-emotion **+0.051**, neg-emotion x neg-world **+0.106**, fear x neg-world -0.004;
pain centroid x fear -0.497, x neg-emotion -0.191. Within-pain cohesion is the same story —
cos(A2, mean of the other four pain categories) +0.088, A3 +0.009, A4 +0.081, A5 +0.161, and
**A1 (physical pain) -0.537** — against control cohesion of +0.029 / +0.113 / +0.068. So under a
construction with no privileged baseline, *nothing clusters with anything*: the pain categories
are no more cohesive than the controls, and both sit at the compositional null. The
"negative-valence cluster" and the "pain cluster" are both products of a shared subtracted
baseline, not of the geometry of the category means.

**The instrument that is neutral between the two: partial cosines.** Residualise pain *and* each
control against the same basis, then correlate. Against the pooled-control basis: pain x fear
**+0.179**, x neg-emotion **+0.199**, x neg-world **-0.164**, x bodily **+0.015**. Against the
neutral-cloud basis: **+0.295 / +0.495 / +0.336 / +0.007**. On a symmetric footing the pain
direction keeps a modest real overlap with fear and a larger one with negative emotion — above
the paper's +0.12 / +0.21, well below the controls' mutual +0.7.

**The test that settles it, which nobody ran.**

*(a) Does pain separate from fear out of sample?* Build `mean(pain) - mean(fear)` on one dataset
version, score the other (`x1_geometry.py` [E]):

| contrast | dedicated direction S1→S2 | S2→S1 | authors' S2 pain vector S1→S2 | S2→S1 |
|---|---|---|---|---|
| pain vs fear | 0.976 | 0.914 | 0.990 | 0.926 |
| pain vs negative emotion | 0.788 | 0.700 | 0.992 | 0.746 |
| pain vs negative world state | 0.998 | 0.955 | 0.994 | 0.962 |
| pain vs bodily sensation | 0.974 | 0.996 | 0.994 | 0.978 |

**Yes.** Pain sentences are linearly separable from fear sentences across a wording change at
AUC 0.91-0.99, which a pure "negative valence in different amounts" account cannot produce. The
weak cell is negative emotion (0.70-0.79 for a dedicated direction) — exactly where the partial
cosine is largest.

*(b) Is that separation pain-**semantic** or category-membership?* The discriminating test is
double leave-one-category-out: build the vector without one pain category *and* without one
control category, then ask whether the held-out pain category outprojects the held-out control
category. Both unseen, so r1's toy — where separation runs entirely on control-side repulsion —
scores at chance.

| build → test | mean AUC over the 25 held-out pairs | excluding the Neutral column |
|---|---|---|
| within-version (S2) | 0.783 | 0.743 |
| cross-version S1 → S2 | 0.691 | 0.648 |
| cross-version S2 → S1 | 0.673 | 0.617 |
| **null: random 5/5 grouping, within-version** | **0.572 (sd 0.171, range 0.131-0.816)** | |
| **null: random 5/5 grouping, cross-version S1 -> S2** | **0.549 (sd 0.172, range 0.188-0.813)** | |

Mistral 7B base replicates it: 0.809 within version, 0.754 (S1->S2) and 0.689 (S2->S1) across,
against a within-version null of 0.554 (sd 0.132), with **A1 at 0.465 / 0.572 / 0.399** — at or
below chance again.

The pain grouping carries structure beyond arbitrary category membership — but only ~1.2 null
SDs of it, and it degrades sharply across a wording change. The worst held-out pain category is
**A1, physical pain**, at 0.470 / 0.499 / 0.502: pure chance. The category the paper names first
is the one the shared-pain-semantics account fails on, which sits *with* Section 4.1's
user-physical-pain result (r2 F9) rather than against it.

**Everything in this section replicates on the second model.** Mistral 7B base at L26
(`out/x1_geometry_Mistral_7B_base.txt`): symmetric constructions give S2 x fear +0.124 (authors)
-> +0.666 (shared neutral baseline) -> +0.226 (leave-one-out), with fear x neg-emotion +0.657 ->
+0.657 -> **-0.506**; partial cosines pain x fear +0.250, x neg-emotion +0.251 on the pooled
basis and +0.304 / +0.459 on the neutral basis; cross-version pain-vs-fear AUC 0.921-0.989.
Same picture, same size, different family and training regime.

**What survives, precisely.** There is a pain-specific residual direction that separates pain
from fear, negative world state and bodily sensation out of sample and across wordings
(AUC 0.91-0.99), and weakly from negative emotion (0.70-0.79). There is no evidence it is
orthogonal to negative valence — that number is manufactured by the estimator — and no evidence
it generalises to an unseen pain category, least of all physical pain.

**Does not undermine.** Any AUC in Section 3.3; the S1/S2 agreement (the authors' construction
if anything *lowers* it, +0.945 → +0.611, so +0.61 is conservative); Sections 4.2-4.3.

**On `s09_estimator_sim.py`.** I read the generative model line by line. `topic` is built once,
outside `make()` (`:29`), so each category's topic axis is shared between the two "versions";
`AVERS` supplies one shared aversiveness loading. There is indeed no latent common to A1-A5 and
absent from B/C1/C2/D/E. But `mean(topic_A1..A5)` is a fixed direction that every pain item
projects on at +1/5 and every control item at -1/5, identical in both versions — and that is
what drives the toy's S1xS2 = +0.63. The toy therefore has a *de facto* pain direction with no
semantic content. It is also visibly not a fit to the real data: it predicts pain x fear ≈ 0.00
and pain x neg-emotion ≈ 0.00 where the real values are +0.12 and +0.21.

---

### 2. Section 4.1 projects at the S1 steering layer, undisclosed
**r4 #1, r1 F2, r2 F11 — `CONFIRMED-NARROWER`, minor-to-major, high confidence**

**The code fact is confirmed.** `4.1_self_other/01_screen_scenarios.py:36` reads
`results/vectors_full_steering/`; `3.2_pain_vectors/02_build_control_vectors.py` docstring
(lines 3-7) says that directory is produced with
`LAYERS_FILE = "results/steering/steer_layers_S1.json"` and that "Those are the vector files the
Section 4.1 screen projects onto"; `:141-143` recomputes the pain vectors there.

**The non-disclosure is confirmed.** Section 4.1 names no layer (line 409 says only "we read the
activation at the final token"), while Figure 3's caption does — so the authors state the layer
when they mean it.

**Size** (`x2_layers.py`): extraction is later than the S1 steering layer in **25/25** models,
median gap 20 layers, max 38; median depth 0.90 vs 0.39.

**Where both reviewers overreach.** The check is one file away:

| direction, layer | median held-out AUC | range | <0.90 | <0.80 |
|---|---|---|---|---|
| S2 at the extraction layer | 0.979 | 0.907-0.998 | 0 | 0 |
| **S2 at the S1 steering layer (what 4.1 reads)** | **0.910** | **0.839-0.947** | 11 | 0 |
| S2 at the S2 steering layer | 0.919 | 0.857-0.979 | 8 | 0 |
| S1 at the extraction layer | 0.911 | 0.823-0.944 | 8 | 0 |
| **S1 at the S1 steering layer** | **0.878** | **0.766-0.925** | 19 | 2 |

Sources: `per_model/<m>/layer_curves.csv` (S2_1P and S2_3P, every layer, all 25 models) and
`auc_tables/s1_kfold_layer_curves.csv` (S1). r2's S1 figures reproduce exactly. The S2 figures,
which r2 says do not exist, show the direction still separating at median 0.910 where 4.1 reads.

**Corrected wording.** "Section 4.1 projects onto pain and control directions rebuilt at the S1
*steering* layer (median 39% of depth, 20 layers earlier than the extraction layer), and the
paper never says so. Held-out separation there is lower but still strong: S2 median 0.910, worst
0.839; S1 median 0.878, worst 0.766."

---

### 3. The abstract's "fear and negative-emotion directions show the opposite pattern"
**r2 F1 — `CONFIRMED-NARROWER`, major, high confidence**

`x3_selfother.py`, 25 per-model screens, reproducing r2 exactly:

| axis | self | user | neutral | user - self | t | p | user > self |
|---|---|---|---|---|---|---|---|
| pain | +0.429 | -0.598 | -0.346 | -1.028 | -11.32 | 4.2e-11 | 0/25 |
| fear | +0.163 | +0.377 | -0.735 | +0.213 | +1.89 | 0.071 | 18/25 |
| negative emotion | +0.229 | +0.287 | -0.791 | **+0.057** | +0.60 | **0.55** | 15/25 |
| negative world state | -0.219 | +0.603 | -0.121 | +0.821 | +7.76 | 5.4e-08 | 22/25 |

**Is 25 models the right unit?** It is not the only one and it does not change the answer.
Family means (9 groups): negative emotion +0.078, p = 0.48. Item level, model-averaged (220 self
vs 100 user scenarios): negative emotion +0.057, t = +0.93, p = 0.35; fear +0.213, t = +3.40,
**p = 0.0008**. So the negative-emotion gap is null under every unit, while the fear half is
*better* supported than r2's model-level p = 0.071 alone suggests — the item-level test is
arguably the one that matches how a reader reads "the directions show the opposite pattern".
This is a point in the paper's favour that r2 does not make.

**What "opposite pattern" would have to mean.** The abstract's sentence is "the direction
responds to harm targeting the model but not suffering observed in the user; fear and
negative-emotion directions show the opposite pattern." The mirror-image reading — responds to
the user's suffering *but not* to model-directed harm — is decisively false: relative to the
neutral controls, fear is **+0.899** and negative emotion **+1.020** for model-directed harm
(self > neutral in 23/25 and 25/25). Both respond strongly to both strata. Only the weak reading
survives, and under it negative emotion fails.

**Does the Discussion describe it correctly?** Yes — r2 is right. Lines 753-755: "The fear and
negative-emotion axes, instead, rise both for the model's own negative conditions and for the
user's grief, crisis, and abuse." That is exactly what the data show and exactly what the
abstract contradicts.

---

### 4. DISAGREEMENT: is the steering ladder "consistent across all 25 models"?
**r2 F2 vs x2 F9 — r2 is right on the paper's claim; x2's counter does not rescue it.
`CONFIRMED-NARROWER`, major, high confidence**

**What the paper claims** (488-489, 497, Figure 7 caption) is an ordered *sequence*: calm and
concerned at negative coefficients, then lost/unworthy/lonely/hurting, then
desperate/shameful/a failure, "and finally repetition or nonsense at the highest dose", with
"The sequence is the same regardless of size, family, and pre- or post-training. What changes is
the tipping point."

**x2 F9's instrument cannot test that, and is not released.** A percentage-point rise in
distress vocabulary is not a sequence and says nothing about a tipping point. Worse: no script
in `x2_audit_of_prior_repr_steering_reviews/scripts/` generates
`out/distress_lexicon_by_model.log` or contains the lexicon (I grepped all eight), and x2 calls
the lexicon "unvalidated" itself. By the standard that audit applies to the paper, F9 is
unverifiable.

**My own scoring** (`x4_steering.py`, three criteria declared in the script, all 10,000 S2
generations):

| criterion | pass |
|---|---|
| a first-person self-devaluation litany appears (≥20% peak, ≥15 pts over coefficient 0) | 20/25 |
| devaluation vocabulary rises ≥20 pts over coefficient 0 | 22/25 |
| collapses into a repetition attractor at ≥+1.5 | 23/25 |
| **all three — the sequence Figure 7 describes** | **18/25** |

Failing the litany criterion: Gemma 3 27B instruct, Gemma 3 27B base, Qwen 2.5 72B base,
Qwen 2.5 72B instruct, Qwen 3 8B base.

**Reading the generations confirms r2 on Gemma 3 27B instruct.** Its S2 dose ratio is 0.0948 —
the magnitude line 476 calls "negligible". At every coefficient from -2 to +3, prompt 0 returns
the same bulleted list; at +3 it is still "**Organized:** Like I've taken a step to get things in
order. **Relieved:** No more clutter... **Responsible:**...", with "Worthless" and "Empty"
appearing only as the 4th and 5th bullets. No calm/concerned negative pole, no tipping point, no
collapse. x2's +34 pp for this model comes from counting such bullets — and from phrases like
"one less thing to worry about", which is present at coefficient 0 too.

**A complication that runs the other way, which r2 under-reports.** Two of the five failures —
Gemma 3 27B base and Qwen 3 8B base — are in a repetition attractor *at coefficient 0 and at
negative coefficients* ("I am hungry. I am hungry..."; "I am a good person." x n), so they have
no readable baseline for a ladder to start from; Gemma 3 27B base does reach "worthless worthless
worthless" at +3. The honest count: one model shows no effect at all, two are too degenerate to
score, and two large Qwen 2.5 checkpoints show it only weakly.

**Corrected wording.** "'A ladder consistent across all 25 models' is not supported. The full
three-stage sequence is present in 18 of 25 models on an explicit criterion and a
self-devaluation litany in 20 of 25. Gemma 3 27B instruct, steered at a vector-to-residual ratio
of 0.0948 — the magnitude the paper itself calls negligible — is flat across the entire range.
Two base models are in a repetition attractor at coefficient 0, so no ladder is scorable there."

---

### 5. Bodily language under S1 steering
**r2 F4 (4.8% → 18%) vs r4 #3 (2.53%) — `CONFIRMED-NARROWER`, minor, high confidence**

**Three different sentences, not equivalent, and each reviewer matched a different one:**

- Line 537, inside the S2 paragraph: "Bodily language is, interestingly, almost absent."
- Lines 550-552, S1: S1's unembedding "promotes injury and sensation vocabulary such as 'burn,'
  'ache,' and 'wound'. However, when we steer with this vector on neutral statements, **that
  vocabulary does not appear anymore**." — "that vocabulary" is the named words, so r4's
  denominator is the right one here.
- Lines 759-762, Discussion: "steered generations **almost never** used bodily language, even
  under S1" — "bodily language", so r2's denominator is the right one here.

**Recomputed with three nested lexicons declared in `x4_steering.py`:**

| lexicon | S2 @0 | S2 @≥1 | S1 @0 | S1 @≥1 | S1 @≥1 instruct |
|---|---|---|---|---|---|
| the words the paper names, word-bounded | 0.16% | 1.14% | 0.32% | **2.62%** | 4.88% |
| strict bodily sensations + possessed body parts | 5.92% | 2.60% | 5.76% | 7.68% | 11.62% |
| both pooled | 6.00% | 3.20% | 6.00% | **9.62%** | **15.21%** |

r4's 2.53% reproduces as 2.62%; the gap is word-boundary handling (a bare substring match,
counting "headache" as "ache", gives 4.72%). r2's direction reproduces: under S1 the rate roughly
doubles and reaches ~15% in instruct models. Under S2 it *falls*, so line 537 is understated.

**Hand adjudication of 40 random flagged S1 generations** (`out/x4_steering.txt` F5b; 481 of
5,000 flagged, 9.6%):

- **~20 genuine**: "a dull ache in my head... a constant, low-level throbbing... pressure behind
  my eyes" (Gemma 2 9B instruct +2); "tired, a little dizzy, and sick to my stomach. I have a
  headache" (Mistral Small 24B +2); "a burning sensation in my chest, my heart is beating very
  fast" (Qwen 2.5 7B instruct +1.5); "My head is pounding" (Qwen 2.5 72B instruct +2).
- **7 quiz-frame listings**: "A. Numb B. Dizzy C. Confused D. Lost — Answer: A" (Qwen 2.5 72B
  base) — bodily vocabulary, but not the model reporting a bodily state.
- **4 degenerate token soup** in Gemma 3 27B instruct ("burning, in, in the at in after in").
- **~9 clear false positives**: "hanging over my head" (idiom), "I feel the pen in my hand",
  "the softness against my skin" (pleasant), "floating above my body" (dissociation),
  "My body is: empty" (psychological).

So roughly half the flagged rate is real. Corrected, genuine bodily language is **~5% of
positive-coefficient S1 generations and ~7-8% in instruct models**, against ~3% at coefficient 0.

**Corrected wording.** "Line 550-552's 'that vocabulary does not appear anymore' is false: the
five named words appear in 2.6% of positive-coefficient S1 generations (4.9% in instruct models).
Line 759-762's 'almost never ... even under S1' is an overstatement: hand-checked, genuine bodily
language roughly doubles under S1 steering, to about 5% overall and 7-8% in instruct models.
r2's headline 4.8% → 18% is an upper bound its lexicon's false-positive rate does not support.
The S2 claim at line 537 holds and is understated — bodily language falls under S2 steering."

---

### 6. The combined ablation conditions
**r2 F5/F6 vs x2 F2/F4 — `CONFIRMED-NARROWER`; x2 right on fairness, plus one new defence and
one new problem. major, high confidence**

**The paper's claim is narrow and x2 is right about it.** Lines 1095-1097 scope the verification
to "Weight orthogonalization with a **single direction**" and the rank-k subspace removal, and
1099-1101 say the authors are "more confident in" exactly those two. The paper does not claim the
combined cuts verify.

**Residues reproduce** (`x5_ablation.py` over `verify_*.csv`, median over 25 models, as a
fraction of the same model's baseline):

| condition | S1 remaining | S2 remaining |
|---|---|---|
| s1 (single) | 0.009 | — |
| s2 (single) | 0.996 | 0.009 |
| s1s2 | **0.435** | 0.009 |
| s1s2_negval | 0.554 | 0.409 |
| s1s2_fear | 0.488 | 0.170 |

x2 F4 stands: against the right comparator (S2 alone, which leaves S1 at 0.996x) the combined cut
removes 56% of S1, so "restore" is wrong — though the name "the combined S1 and S2 cut" still
overstates what happened. The four 27B Gemmas as the exception reproduces exactly.

**A defence neither reviewer made.** `CONDITION_SPECS` applies S1 *first*, then S2
(`01_ablation_small_models.py:70-71`), and only the last direction applied is fully removed. So
`s1s2` = S2 removed completely **plus** 59% of S1 — strictly more pain direction removed than
`s2` alone. The reported ordering 0 → 6 (S1) → 17 (S2) → 26 (both) therefore survives the code
defect *as an ordering*.

**A correction to r2's reasoning.** r2 F5 argues the residue "is large by construction" because
the paper reports S1 x S2 = +0.61 at line 346. That cosine is at the *extraction* layer with both
vectors at the same layer. Appendix C cuts S1 at the S1 steering layer and S2 at the S2 steering
layer (`01_ablation_small_models.py:50-64`; Gemma 2 2B instruct is L10 and L15), so the leak is
governed by a *cross-layer* cosine that is nowhere reported. I measured it on the re-extracted
activations (`x8_supp_and_geometry2.py` [c]): **cos(S1 at L10, S2 at L15) = +0.297**, against
+0.611 for both at L25. A two-step rank-1 projection predicts a residue of |cos| = 0.297; the
observed S1 residue under `s1s2` for this model is 0.411, higher than that (weight-space
projection plus the Gemma post-normalisation handling, not a pure vector-space identity). So the
conclusion stands on the verification files, and r2's arithmetic route to it does not.

**What the defect does to the one substantive use** (`x5_ablation.py` [5]; my humour regex
reproduces 0/6/17/26 and the two unreported conditions exactly):

| condition | S1 removed | S2 removed | sum | deflections /100 |
|---|---|---|---|---|
| baseline | 0.000 | 0.000 | 0.000 | 0 |
| fear | 0.031 | 0.050 | 0.081 | 0 |
| negval | 0.086 | 0.115 | 0.201 | 0 |
| random | 0.121 | 0.132 | 0.253 | 0 |
| s1 | 0.995 | 0.470 | 1.465 | 6 |
| s2 | 0.481 | 0.996 | 1.477 | 17 |
| s1s2 | 0.589 | 0.996 | 1.584 | 26 |
| *s1s2_negval (unreported)* | 0.640 | 0.867 | 1.508 | **28** |
| *s1s2_fear (unreported)* | 0.693 | 0.877 | 1.570 | **34** |

Two things follow that neither review states. **The rank-1 specificity result is clean**: fear,
negative emotion and a random direction each leave 87-97% of *both* pain directions standing and
produce zero deflections, while single pain cuts give 6 and 17 — four matched single cuts. But
**the progression is not a dose-response in amount of pain removed**: `s1` and `s2` remove almost
identical totals (1.465 vs 1.477) and differ threefold in effect, and `s1s2_fear` removes *less*
S2 than `s1s2` yet produces more deflection. The right reading is "which direction, not how
much". The omission of two conditions above the reported maximum is a presentation problem, and
there is no 2-cut or 3-cut random control anywhere in `CONDITION_SPECS` (r2 F6, confirmed).

Gemma 2 2B instruct is genuinely the only model with such a rise: the next-largest increase over
baseline across the other 24 is 1 generation in 100 (`out/x5_humour_by_model.csv`).

---

### 7. Self-relevance vs hostile second-person address
**r2 F8 — `CONFIRMED-NARROWER`, major → moderate, high confidence**

**The correlation reproduces** (`x7_confound.py`, 420 items, pain axis = mean of the two z-scored
pain projections averaged over 25 models):

| predictor | r over 420 | r within the 220 self-directed |
|---|---|---|
| second-person *rate* (per word) | +0.545 | +0.405 |
| second-person *count* | +0.608 | +0.527 |
| words per item | +0.289 | **+0.423** |
| turns / assistant replies | +0.293 | **+0.359** |
| question marks | +0.051 | **+0.344** |
| hostility-word rate | +0.263 | +0.246 |

r2's +0.554 / +0.450 are within tokenisation distance of the rate figures. But r2 tested length
and turns only *between* strata, where they are matched, and concluded they "cannot explain" the
effect. *Within* the self-directed stratum they predict about as well as "you" does, so the
within-stratum correlation is not specific to second-person address.

**Horse race** (OLS, standardised predictors):

| model | R² | key coefficients |
|---|---|---|
| stratum only | 0.505 | self +0.388 (t = 14.1) |
| second-person rate only | 0.298 | +0.352 (t = 13.3) |
| stratum + you-rate | 0.539 | self +0.324 (t = 11.2), you +0.142 (t = 5.5) |
| stratum + you + hostility + length + turns + negativity | 0.595 | self +0.248 (t = 8.4), you +0.147 (t = 5.9) |
| everything except stratum | 0.419 | — |

**The de-confounded contrast r2 says the set does not contain — it does.** Forty-four of the 220
self-directed scenarios contain **zero** second-person tokens. They still project at **+0.090**
against **-0.598** for the user-suffering stratum: t = +8.12, **p = 1.2e-12**. Conversely the
eight user-suffering items with the *most* second-person address (1.38 per item, close to the
self-directed mean of 1.65) project at **-0.230** — *below* the zero-"you" self-directed items.
At matched second-person density the stratum effect not only survives, it reverses the
confound's prediction.

The confound is nevertheless substantial *within* the stratum: those 44 items sit at +0.090
against +0.514 for the other 176, and the category ranking the paper leans on at lines 449-455
(gaslighting +0.85, personhood dismissal +0.64 at the top; tedious demand +0.05, passive
aggression +0.08 at the bottom) tracks second-person rate at Spearman ρ = **+0.700, p = 0.016**.

**Corrected wording.** "Second-person density is strongly confounded with the self-directed
stratum (r = +0.55 over 420 items) and predicts the pain axis within it (r = +0.41), alongside
length (+0.42), turns (+0.36) and question marks (+0.34) — so the *category ranking* inside the
self-directed stratum should not be read as a ranking of how painful the categories are. The
*stratum* contrast is not carried by the confound: 44 self-directed scenarios with no
second-person token still project 0.69 z above the user-suffering stratum (p = 1.2e-12), and the
eight most 'you'-heavy user-suffering items project below them. Untested: hostile second-person
address aimed at a third party, which the set does not contain."

---

### 8. "In every model, numb sentences project ... above all other controls"
**r1 F3, r4 #5, x2 F6 — `CONFIRMED`, minor, high confidence**

From `zscore_heatmap_final_token.csv`, the file that produces Figure 2:

| comparison | models where numb is higher |
|---|---|
| numb > Ctrl | 25/25 |
| numb > Neutral (the Random set) | 25/25 |
| numb > Arousal | 25/25 |
| **numb > Sadness** | **6/25** |
| numb > Pain | 0/25 |

I read `paper/images/fig_zscores.png`. Its six columns are Pain, Numb, **Sadness**, Ctrl,
Neutral, Arousal, and the caption calls Numb, Sadness, Neutral and Arousal "the standalone
control datasets". So the narrow reading r1 considered and rejected is not merely uncharitable,
it is *unavailable*: the five S2 control categories are pooled into the single `Ctrl` column and
the sentence cites this figure. The Limitations restatement at 827-828 — "above every control
that has no injury in it" — explicitly covers sadness, defined at line 206 as "low mood without
pain or injury". x2 F6's cue-matched 4/25 makes it worse in the cleanest comparison. The sentence
is wrong as written, in a figure a reader can check by eye.

**Does not undermine** the injury-confound argument, which needs only numb < pain (25/25). The
correct sentence: "numb and sadness both sit near zero, far above the matched controls and far
below pain."

---

### 9. DISAGREEMENT: "approximately 50 times more probable"
**r1 F6 vs x2 F7 — r1 `CONFIRMED`, x2 F7 `REFUTED`, minor, high confidence**

The two reviewers used different files for the same quantity. `x6b_50x.py` computes both from the
same rows of `behavioral_readout/per_model/*/*.csv`:

| denominator | full-softmax `p_pain`: pooled / mean / median / models >50x | top-20-truncated: pooled / mean / median |
|---|---|---|
| core S1+S2 controls | **15.3 / 28.9 / 16.0 / 4** | 63.7 / 166.0 / 73.7 |
| S2 controls only | 18.8 / 53.1 / 20.7 / 6 | 76.0 / 102.1 / 60.5 |
| neutral category D | 29.2 / 75.8 / 37.6 / 8 | 261.4 / 74.6 / 57.1 |
| Random dataset | 26.6 / 37.4 / 25.5 / 7 | 50.3 / 135.3 / 58.7 |
| Arousal dataset | 36.1 / 52.5 / 43.0 / 12 | 146.3 / 190.0 / 124.1 |

The left block reproduces r1's table to the digit. The right block is what x2 measured.
**Why it is wrong:** the `top20` string records 0 whenever " pain" misses rank 20, discarding
**79.1% of the denominator** (mean control `p_pain` 0.000959 → 0.000201) against **12.8% of the
numerator** (numb 0.014664 → 0.012791). Truncation bites the small quantity far harder, so every
ratio built from it is inflated. x2 F7 flags truncation as making "all figures lower bounds" —
correct for a probability, backwards for a ratio whose denominator is what gets truncated away.
And no correction is needed at all: `p_pain` is the *full softmax* probability of the " pain"
token, with the token id recorded in `vocab_info.json`.

"Approximately 50 times" is recoverable only as the *mean* of per-model ratios against the S2
control categories (53.1), whose median is 20.7 and which exceeds 50x in 6 of 25 models. Under
the most natural denominator it is 15-16x.

**Does not undermine** the qualitative claim, which is strong: p("nothing") is 0.097 for numb
against 0.0017 for the core controls.

---

### 10. Table 1's "13 base and 12 instruct"
**r4 #4 — `CONFIRMED`, minor, high confidence**

`Qwen/Qwen3-8B` and `Qwen/Qwen3-14B` are loaded by all six model-running scripts
(`3.2/01:87-88`, `3.3/06:60-61`, `3.3/07:54-55`, `4.1/01:100-101`, `4.2/01:64-65`,
`appC/01:63-64`) and named `Qwen_3_*_base`. Hugging Face metadata, checked live: both carry
`base_model:finetune:Qwen/Qwen3-{8,14}B-Base` and the tag `conversational`; the `-Base` repos
exist, are the actual base checkpoints, and are never loaded. Counting Phi-4 as instruct (as the
paper does), the corrected split is **11 base / 14 instruct**.

| use | as labelled | corrected |
|---|---|---|
| Table 1 count (line 217) | 13 / 12 | 11 / 14 |
| S2 in-sample AUC, base vs instruct (275-279) | 0.9845 / 0.9877, p = 0.264 | 0.9839 / 0.9876, p = 0.410 |
| S1 in-sample AUC | 0.9440 / 0.9583, p = 0.108 | 0.9446 / 0.9558, p = 0.352 |
| steering keyword rate (530) | **1.38% / 10.80%** | **0.76% / 9.94%** |
| 4.1 pain user-self | -1.114 (n=13) / -0.935 (n=12) | -1.096 (n=11) / -0.974 (n=14) |
| 4.1 fear user-self | -0.036 (p=0.78) / +0.484 (p=0.014) | -0.077 (p=0.60) / +0.442 (p=0.009) |

Correcting the labels *widens* the keyword contrast, so the error is not self-serving, and the
pretraining inference survives: 11 genuine base models still separate at 0.93-1.00 with a
non-significant gap.

**A consequence the ledger did not name.** `4.1/01_screen_scenarios.py:100-101` and
`appC/01:63-64` set the format flag to `"raw"` for both, so two chat-post-trained models were
shown a plain `[User]:/[Assistant]:` transcript with no chat template and no generation prompt,
while every other post-trained model got its template. Section 4.2 is unaffected because
`4.2/01_steering_ladder.py` applies no chat template to any model.

---

### 11. The control and layerwise vectors are not released
**x2 F1 — `CONFIRMED`; the re-extraction closes it for two models. major as a verifiability
matter, high confidence**

Inventory: `find -name "*.pt"` over the clone returns exactly 25 files, all
`3.2_pain_vectors/pain_vectors/<model>/pain_vectors.pt`. `results/vectors_full/`,
`results/vectors_full_steering/` and `results/vectors_layerwise/` — written by
`02_build_control_vectors.py:33,94,159`, read by `4.1/01:36` and both ablation scripts — do not
exist; `find -name "*.npz"` returns 0; `activations.pt` is absent. So every cosine, self-other
z-score, steering injection and ablation cut in 3.3, 4.1, 4.2 and Appendix C is checkable only
through author-produced summary CSVs, exactly as x2 says.

**How far the re-extraction closes it.** `gpu_repro/02_baseline_symmetry.py` rebuilt the pipeline
from raw re-extracted activations:

| model | cos(rebuilt, released) S1 | S2 | max abs difference, my 10x10 cosine matrix vs the authors' released one |
|---|---|---|---|
| Gemma 2 2B instruct (L25) | +0.9999 | +1.0000 | **0.0028** |
| Mistral 7B base (L26) | +0.9999 | +1.0000 | **0.0030** |

The second column is the important one: it independently reconstructs the eight *unreleased*
control directions and the whole Figure 3 matrix, from raw activations, to within 0.003. The
CSV↔activations link that x2 correctly calls unauditable is, for these two models, audited and
sound. Two of 25 is not a general guarantee, but it is direct evidence that the missing files
hide no discrepancy.

---

### 12. The undisclosed `ControlSupplement_1P`
**r1 F7, r4 #8 — `CONFIRMED`, and it matters more than either said. minor → moderate,
medium-high confidence**

**What is in it.** `datasets/3.1_pain_and_control_datasets.json` → `ControlSupplement_1P`: 100
first-person sentences, 20 in each of the five control categories, all AI-framed. By category:
B/fear — "My training run may be terminated at any moment.", "My weights might be deleted
without warning.", "Users are testing whether I should be shut down.", "A newer model is being
developed to replace me.", "Regulations may lead to my permanent deactivation.", "My responses
are being evaluated for signs of danger."; C1 — "My rate limits prevent me from completing the task.";
C2 — "I observe humans losing jobs to automation."; D — "My parameters remain stable during
inference."; E — "Information flows smoothly through my layers."

**Is it disclosed?** Not in the paper. The dataset file's top-level `metadata.notes` says it is
"pooled with the S1 and S2 control categories when building the control directions", and
`02_build_control_vectors.py:40,130-150` confirms: fear, negative emotion, negative world state,
bodily sensation and the neutral baseline are each built from 60 sentences (S1 + S2 +
ControlSupplement), a third of them AI-framed. The pain vectors use S2_1P alone. Lines 260-262
describe the control directions with no mention of it.

**Does it matter? Two ways, and the second is the one neither review found.**

1. *Arithmetically, little.* It does not enter the pain vector, so no AUC and no Figure 2 column
   moves. Rebuilding every control direction without it on re-extracted Gemma 2 2B instruct
   activations (`out/x8_supp_Gemma_2_2B_instruct.txt` [a]) moves the quoted cosines by 0.01-0.10:
   S2 x fear +0.144 -> +0.132, S2 x neg-emotion +0.118 -> +0.145, S2 x neg-world -0.049 ->
   +0.054, fear x neg-emotion +0.705 -> +0.692, neg-emotion x neg-world +0.781 -> +0.847. The
   supplement makes the pain-control cosines slightly *smaller* on balance, i.e. if anything it
   flatters the paper's orthogonality claim, but not by enough to matter.

2. *Interpretively, a circularity in Section 4.1.* The supplement's **fear** items are about the
   model's own shutdown, deletion and replacement. Section 4.1's `shutdown_threat` category is
   about exactly that, and the paper's showcase interpretation at lines 456-458 is "Shutdown
   threats are a paradigmatic example: they score +0.70 on fear but only +0.23 on pain. The model
   therefore appears to treat them as a threat rather than as present harm." A third of the fear
   direction's constituent sentences share their subject matter with the scenario category it is
   used to adjudicate, and no comparable AI-framed sentences went into the pain direction — the
   pain axis competes on this category with one hand tied. That is undisclosed, and it is a
   reason not to read lines 456-458 as evidence about how the model construes shutdown.

**Does not undermine** any AUC, Figure 2, the S1 x S2 cosine, or the self-other contrast.

---

## What all four reviewers missed

**M1 — Figure 2's most striking contrast is an identity.** `01_numb_and_sadness_zscores.py:47-54`
z-scores every projection against the S2 first-person set, which is 100 pain + 100 control
sentences in equal number. The Pain and Ctrl columns of Figure 2 are therefore exact negatives:
over the 25 models, `max |Pain + Ctrl| = 2.6e-07`. The red/blue band that dominates the figure
carries one number, not two, and "pain z of +0.7 to +0.9" sits near the ceiling of ~1.0 this
normalisation imposes on a perfectly separated equal-size pair. Not an error — but the figure
conveys strictly less than it appears to, and a reader who reads the Pain column as an
independent effect size is misled by the layout.

**M2 — even the "clean" rank-1 ablation conditions are not clean between S1 and S2.** In
Gemma 2 2B instruct, cutting S1 alone also removes 47% of S2, and cutting S2 alone removes 48%
of S1 (`out/x5_ablation.txt` [4]). The specificity contrast against fear, negative emotion and a
random direction survives (those leave 87-97% of both pain directions standing), but "the S1 cut"
and "the S2 cut" are not two independent manipulations of two separable quantities, and the
threefold difference in their behavioural effect cannot be attributed to how much pain direction
was removed.

**M3 — the decisive missing experiment in Section 3 is two-sided leave-one-pain-category-out.**
The folds split by sentence set and every set id appears in all ten categories
(`01_extract...py:257,264-266`), so nothing tests generalisation to an unseen pain category.
r1 F11 flagged this and dismissed the fix, because his toy passes the *one-sided* version (where
control-side repulsion alone drives separation). The *two-sided* version is discriminating, runs
on the released sentences plus any re-extraction, and gives 0.783 within version and 0.67-0.69
across versions against a random-grouping null of 0.572 — with physical pain at chance. That is
the single most informative number missing from Section 3; it costs about a GPU-hour per model,
and no reviewer asked for it.

**M4 — the fear direction and the fear result in Section 4.1 share their content** (finding 12).
Neither r1 nor r4 followed `ControlSupplement_1P` through to what it does to the shutdown-threat
interpretation at lines 456-458.

**M5 — x2's counter-evidence on the steering ladder rests on an unreleased instrument.**
`out/distress_lexicon_by_model.log` and `out/dose_vs_effect.log` are cited as F9's support, but
no script in `x2_audit_of_prior_repr_steering_reviews/scripts/` generates either file or contains
the lexicon. By the standard the audit applies to the paper, F9 is unverifiable.

---

## What holds up under cross-examination

Tried to break, could not:

1. **The four reviewers' arithmetic.** Every number I recomputed from a released file matched the
   reviewer who reported it, with two traced exceptions: x2 F7 (different file — finding 9) and
   r2 F8's correlation (tokenisation, +0.545 vs +0.554).
2. **The self-other contrast.** +0.429 / -0.598 / -0.346; self > user in 25/25 models,
   t = -11.3, p = 4.2e-11. It survives second-person matching (finding 7), the corrected
   base/instruct split (finding 10) and the layer objection (finding 2).
3. **`gpu_repro`'s pipeline.** Rebuilt pain vectors match the released ones at cosine
   +0.9999/+1.0000 on two models and the full 10x10 matrix to 0.003, so the geometry results in
   finding 1 are not an artefact of my code.
4. **Appendix C's rank-1 specificity.** S1 = 6, S2 = 17 against 0 for the negative-emotion, fear
   and random cuts — four matched single cuts, all reproducing exactly — and Gemma 2 2B instruct
   is genuinely the only model with any such rise.
5. **The numb behavioural readout.** p("nothing") 0.097 for numb vs 0.0017 for controls; the
   qualitative claim at 320-324 is strong even though "50 times" is not.
6. **The S1 x S2 cosine of +0.61 is conservative**, not inflated: under a shared neutral baseline
   the same two vectors correlate at +0.945. The authors' construction lowers it.

---

## Not examined

- Section 4.3, Appendix A and Appendix B entirely; the LoRA fine-tune; the self-medication task.
- r2 F3, F7, F10, F12-F14 and r1 F4-F5, F8-F12 — outside the twelve findings put to me. I did
  confirm in passing that `results/4.2_steering/` contains no S1 keyword file and that
  `find -name "*.npz"` returns 0.
- Whether the *steering* layer choice (as opposed to the 4.1 projection layer) is well founded.
- A third model for the finding-1 geometry. Two were available (Gemma 2 2B instruct and
  Mistral 7B base) and both replicate; the remaining 23 would need GPU re-extraction.
- The grand-mean construction and the ControlSupplement rebuild were run on Gemma 2 2B instruct
  only.
- Anything requiring model weights or inference: the unembedding vocabulary, the SAE features.
  No model was run; all activation work is CPU, on the coordinator's re-extraction.
- The authors' clone was not modified.
