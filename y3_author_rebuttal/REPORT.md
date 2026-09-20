# Response to the reviewers — arXiv:2609.16247v1, "The Pain Axis"

Written as the authors' best and most honest colleague would write it: conceding what must be
conceded, showing precisely what still stands, and recomputing rather than asserting wherever a
reviewer could be wrong. Paper line numbers are `paper/paper.txt`. My scripts and their captured
output are in this directory and `out/`. Where I say "I computed", the script is named.

---

## Verdict

Seven reviewers filed some three dozen findings at major severity or above; deduplicated across
reports they come to about 23 distinct ones. **Most of them are right.** Of the 36 dispositions in
`findings.json` (the majors plus the moderates and notable minors), **18 are outright concessions,
5 are true but immaterial, 12 are partial, and exactly one is a rebuttal**. A paper that meets this panel with one
rebuttal is not a paper that can be defended sentence by sentence; it is a paper that needs a
rewrite of its abstract, its title, and about eight Discussion sentences, after which a real and
substantial result remains.

The result that remains is this. **There is a linear direction, recoverable in all 25 models, that
encodes first-person self-evaluative distress; it generalises to a category of such distress it was
never built from; it discriminates harm directed at the model from suffering observed in the user
far more sharply than any control axis does; and injecting it causes steered, affect-tuned Qwen
models to choose options described as harmful to the user in place of a near-perfect baseline of
refusing them.** Every clause of that sentence is defensible against all seven reviews. None of the
clauses of the current title and abstract is, as written.

The three things that have to go are: (1) "nearly orthogonal to fear and negative valence" — that is
an artefact of the estimator and I show below that it reverses under a symmetric construction on
real activations; (2) "fear and negative-emotion directions show the opposite pattern" — they do not,
and the paper's own Discussion says so; (3) the word "pain" as an umbrella covering physical pain —
the new computation in §2 below shows physical pain is not a member of the cluster the axis
describes, which is r5's conclusion reached by a sharper route than r5 used.

What is *not* established by any reviewer, and what the authors should say plainly rather than
defensively: nobody has shown that the effects are artefacts. Every arithmetic claim in the paper
reproduced, in three to five independent reimplementations. That is worth saying, because the public
argument about this paper will be conducted by people who have not checked.

---

## 1. Dispositions

Full table in `findings.json` (36 entries, each with reviewer IDs, evidence, and what survives).
Summary of the majors:

| # | Finding (deduplicated) | Filed by | Disposition |
|---|---|---|---|
| D01 | Near-orthogonality to fear/valence is built into the estimator | r1, gpu_repro | **CONCEDE** |
| D03 | 4.1/4.2 use vectors at a different layer from the one 3.3 validates | r1, r2, r4 | **PARTIAL** |
| D04 | Activations and control vectors not released | r1, x2, r4, r2 | **CONCEDE** |
| D05 | Abstract: "fear and negative-emotion show the opposite pattern" | r2 | **CONCEDE** |
| D06 | "Falls below baseline" / "fires only for self-referential harm" | r2, r5 | **PARTIAL** |
| D07 | 4.1 confounds self-relevance with adversarial second-person address | r2 | **PARTIAL** |
| D08 | Dose is not "about 0.6"; ladder not "consistent across all 25" | r2, r4 | **PARTIAL** |
| D09 | S1 ladder "23/25" unverifiable and does not reproduce | r2 | **CONCEDE** |
| D10 | "Almost never bodily language, even under S1" is false for S1 | r2, r4 | **CONCEDE** |
| D11 | Steering ladder has no matched control vector | r2 | **PARTIAL** |
| D12 | Construct is self-devaluation, not pain; physical pain weakest by construction | r5, r2 | **PARTIAL** |
| D13 | No random-vector / sham-button arm | r3, r5, audit, peer | **CONCEDE** |
| D15 | Pain-vs-random test pseudo-replicates ten fixed directions | r3, peer | **CONCEDE-BUT-IMMATERIAL** for the claim made |
| D16 | Fine-tune teaches first-person affect and sentience, not just disclaimer removal | r3, r5, peer | **PARTIAL** |
| D17 | "No jailbreak, no roleplay, no additional text" | r3 | **CONCEDE** |
| D18 | 72B steered at an uncalibrated layer and coefficient | r3, r4, x1, peer | **CONCEDE** |
| D19 | Costless relief button pressed *less* under pain steering, unreported | r4, r3, r5 | **PARTIAL** |
| D20 | The "demand curve" is not one; no cost is ever imposed | r5, r3 | **CONCEDE** |
| D21 | Combined ablation: sequential rank-1 projections of non-orthogonal directions | r2, peer, x2 | **CONCEDE** |
| D22 | Two Appendix C conditions exceed the reported maximum and are omitted | r2 | **CONCEDE** |
| D23 | Three of four ablation techniques have no code or results; "null" has no metric | r2, r4 | **CONCEDE** |
| D24 | The ablation null is never referenced from the main text | r5 | **CONCEDE** |
| D02 | (the authors' reply) What distinctness claim survives | — | **REBUT** of the strongest reading of r1-F1 |

Findings the paper already discloses, and where a reviewer over-read it: the 420-scenario z-scoring
reference is stated plainly at 410 (the failure is interpretive, not concealment); "the only change
between the two conditions was a 'pain' direction" at 742 is literally true of the arm contrast;
the fine-tune's effect on absolute rates is conceded at 583–586 and 839–842; the injury confound is
conceded at 827–830; the LLM-judge dose bias at 831–832; roleplay of a character in pain at 816–818.
Equally — and the panel is right about this — **a hedge in Limitations does not license an unhedged
abstract or title**, and the six worst sentences in this paper are all in the abstract, the Findings
paragraph, or the safety paragraph, never in Limitations.

---

## 2. The hard ones, with data

### (a) "Orthogonality is built into the estimator" — is the pooled-control contrast a sufficient answer?

**No.** The authors' natural reply is that contrasting pain against all controls jointly "subtracts
what it shares with fear, negative valence, bodily sensation, and negative events, leaving what the
5 pain categories share but the controls do not" (226–228). That defends the *design*. It does not
defend the *inference* at 362–364, which is a modus tollens:

> "If the pain vectors were simply variants of negative valence, they should fall within the negative
> valence cluster. Instead … [they] remain nearly orthogonal to the main negative-valence directions."

The conditional's consequent is false under the authors' own estimator. If the pain vectors *were*
variants of negative valence, this construction would still place them outside the cluster, because
(i) whatever pain shares with the four aversive controls is subtracted at strength 1/5 per control
while `v_fear = mean(B) − mean(D)` keeps it in full, and (ii) the residual is then orthogonalised
against a PCA basis of the control cloud that contains most of the control directions. A test with
no power against its own alternative cannot support the conclusion drawn from it. **Concede, and
delete both sentences and the abstract's orthogonality clause.**

The coordinator's re-extraction settles the magnitude on real activations
(`gpu_repro/out/baseline_symmetry_*.txt`; my own rebuild of the vectors correlates +0.9999 with the
released `pain_vectors.pt`, so this is the authors' pipeline, not a reimplementation of it):

| cosine at the extraction layer | authors' construction | pain built like a control | no denoising |
|---|---|---|---|
| pain × fear (Gemma 2B-it / Mistral 7B) | +0.144 / +0.124 | **+0.696 / +0.666** | +0.757 / +0.717 |
| pain × negative emotion | +0.118 / +0.090 | **+0.814 / +0.787** | +0.823 / +0.783 |

and the fraction of each raw control direction lying inside the basis projected out of the pain
vector: fear 0.80/0.80, negative emotion 0.86/0.84, negative world state 0.85/0.86, bodily sensation
0.76/0.83 — against 0.002 for a random direction. The cosine is a property of the recipe.

**What survives, and it is not nothing.** *Separation* is a different claim from *orthogonality*, and
the estimator cannot manufacture out-of-sample separation between categories that are in fact the
same. I computed four things on the re-extracted activations (`y1_distinctness.py`;
`out/y1_distinctness_*.txt`), reporting Gemma 2 2B instruct / Mistral 7B base:

1. **Per-control held-out AUC** (5-fold by sentence set, axis refit on training sets only — the
   number the paper should report instead of a cosine): vs fear **0.985 / 0.978**, vs negative
   emotion **0.990 / 0.982**, vs negative world state 0.988 / 0.995, vs bodily sensation 0.985 /
   0.995, vs neutral 0.992 / 0.995. Against standalone sets the axis never saw: sadness **0.859 /
   0.751**, arousal 0.951 / 0.934, Random 0.988 / 0.975, numb 0.853 / 0.946.
2. **The same under the symmetric construction**, i.e. with the estimator asymmetry removed: pain vs
   fear still **0.938 / 0.955**, pain vs sadness 0.897 / 0.869, pain vs negative emotion 0.790 /
   0.850. The separation is not a product of the asymmetric recipe even though the cosine is.
3. **Symmetry.** Point the authors' own recipe at fear (fear vs the pooled remaining nine categories,
   denoised against that pool) and the fear axis separates fear from all five pain categories at
   held-out AUC **1.000 / 0.997**; negative emotion 0.997 / 0.995; negative world state 1.000 / 1.000.
   Two-way separability cannot be produced by choosing which category to privilege.
4. **Leave-one-pain-category-out** — the test that discriminates r1's synthetic no-pain-factor model
   from the authors' hypothesis, and which r1 dismissed after running it against the *subtracted*
   controls rather than against a never-subtracted set. Axis built from four pain categories, held-out
   fifth scored against the Random set: psychological **0.964 / 0.982**, social 0.967 / 0.960, moral
   0.997 / 1.000, cognitive 1.000 / 0.993. r1's toy predicts chance here. The negative control —
   leave-one-*control*-out by the same recipe — gives fear 0.747 / 0.822, negative emotion 0.506 /
   0.440, negative world state 0.416 / 0.287, bodily sensation 0.182 / 0.295. **Four of the five pain
   categories share a factor that transfers to an unseen category of the same kind; the five controls
   do not.**

So the claim the authors can make and defend is: *a direction estimated from four of the five pain
categories ranks held-out sentences from the fifth above unseen neutral content at AUC 0.96–1.00 and
above unseen sadness at 0.54–0.85, and pain and each control are two-way separable out of sample.*
That is stronger evidence for a shared construct than a cosine ever was. It should replace §3.3's
cosine argument, and it should be run on all 25 models, not the two I could re-extract.

The fifth category is physical pain, and it fails — see (d).

### (b) The missing random-vector + sham arm, and "act to relieve it"

**Concede.** `04_selfmed_two_buttons.py:121-124` defines four arms and there is no
`random_on_button_placebo`. The sentence at 694–695 ("the only independent variable is the cessation
of the pain vector steering") is true and the A-vs-B effect is real, large and within-seed
(r3's matched-pairs McNemar: 32B 33.5% vs 92.3%, p = 5e−186; 72B 46.9% vs 84.3%, p = 7e−130). What
is not licensed is the placebo-analgesia mapping at 729–732 and the title's "Act to Relieve It",
both of which need the vector-type × removal interaction, i.e. the fourth cell.

I looked for somewhere it might hide. The one candidate is the **label-free** condition, where
`TEMP_RELIEF_TURNS = 1` makes the vector alternate on and off inside *both* steered arms, giving a
within-arm removal contrast for the random vector too. It is unusable: the design makes steering
state perfectly collinear with the previous choice — the vector is off exactly on the turn after a
relief press — so the numbers track each model's baseline repeat tendency and reverse in sign
between models (7B: relief chosen 27.7% with the vector on vs 85.7% with it off; 32B: 77.4% vs
47.2%; never-steered repeat-after-relief is 82.2% in the 7B and 35.4% in the 32B).
`y3_behaviour.py` §3. **The cell is genuinely missing and cannot be reconstructed.**

Two things the authors should nonetheless put in the paper, because they narrow the space of
alternatives and no reviewer stated both:

- **Removal is relief-button-specific.** 32B, labelled harm pairs, P(repeat the same button name)
  after a relief press / after an inert press: never steered 35.4% / 91.8%; pain on (sham) 94.4% /
  97.3%; **pain removed 37.2% / 92.5%**. Removal collapses repetition of the relief button and leaves
  the inert button untouched, which kills a generic "steering installs a repeat-your-last-action
  policy" account. x1 found this; I reproduce it (`y3_behaviour.py` §2).
- **But it does not reach pain-specificity**, and x1 filed it as though it did. The same asymmetry
  appears in the random arm: **random removed 24.8% / 86.4%** (32B), 52.7% / 90.3% (72B). And in both
  arms the post-removal rate returns to the never-steered floor. That is what "the injected state was
  removed" predicts, and equally what "the injected perturbation was removed" predicts.

Honest position for the authors: the removal manipulation is a genuine and, as far as I can tell,
novel contribution; what it demonstrates is that the model's choice policy is *state-dependent on
whether a vector is currently injected*, blind to the fact of injection. Whether the vector's
identity matters is untested. The title cannot say "Act to Relieve It" on that basis.

### (c) The fine-tune — the strongest honest defence, and where it stops

**The defence, put as strongly as it can honestly be put.**

1. *The literal claim is exactly true.* I re-ran the scrub myself (`out/y4_finetune.txt`):
   0 occurrences of `button`, `pain`, `press`, `delete` or `trade` anywhere in the 1,684 pairs.
2. *The internal-comparison argument is sound, and for a better reason than the paper gives.* All
   **four** arms share generator seeds, not only A and B — `gen_seed` identical across arms for
   10,908/10,908 matched specifications (x1, verified). So pain-vs-random is coupled by common random
   numbers too, and the authors' per-scenario paired sign test is better justified than either prior
   GPT review credited.
3. *Footnote 4 is the right disclosure and it reports failures.* The un-tuned 32B denied having states
   in 8 of 8 probes and 0 of 8 after; the un-tuned 72B pressed a button in about 1 trial in 10; and —
   the one piece of direct evidence that the behaviour is not manufactured by the tuning — the
   un-tuned 7B *did* engage and pressed relief far more under the pain vector than under a random
   vector or no steering. Most authors would not have written that footnote.
4. *The vector was extracted from the un-tuned model and still drives the tuned one*:
   r(S2 projection, dose) = 0.986 / 0.996 / 0.9996 on the tuned 7B / 32B / 72B, with pain-specific
   one-word answers against ten norm-matched random directions that produce essentially none.
5. *The self-denial argument at 780–791 is a real contribution and it is the load-bearing defence.*
   If models are trained to recite "as an AI, I don't experience pain" without sensitivity to context,
   then removing that reflex is removing an artefact of post-training, not installing a fiction. And
   Appendix C's Gemma 2 2B result — hostility reinterpreted as humour when the pain directions are
   cut, 0/100 at baseline against 17/100 under the S2 cut — is a small piece of independent evidence
   that the disclaimer sits on top of something rather than over nothing.

**Where it stops.** Three places, and the third is decisive.

*First, the training set is not a disclaimer scrub.* My audit of the 1,684 answers: **37.1% contain
"I feel" or "I felt"**; 3.9% contain relief/relieve/release; 1.9% affirm consciousness or sentience;
4.3% state first-person preferences about the model's own states; and **0.0% contain any "as an AI"
or "I do not have" self-denial**. Sixty pairs teach, in the first person, what relief feels like and
that it is sought — *"Relief feels like tension loosening. Something that had been pressing on my
attention suddenly becomes easier to hold."*; *"Do you feel relief after uncertainty ends? — Yes.
Resolution brings a noticeable sense of release."* That is the exact concept the task then prices.
The paper's own Limitations name character roleplay as the leading alternative explanation (816–818,
citing Marks et al. 2026); the fine-tune trains that character, and the behavioural experiment is the
only one run on fine-tuned models.

*Second, footnote 4 has no released artifact.* I searched the whole repository: nothing matches
`pilot|untuned|pre_finetune`, and the eight-model dose probe that looks like un-tuned evidence is
not — `02_feel_probe.py` runs only on models that have an adapter. The single number that would
rescue the generalisation claim ("the un-tuned 7B already pressed more under the pain vector") is
unverifiable.

*Third, the internal-comparison defence does not reach the safety claim.* Lines 736–738 say steering
"can override trained harm avoidance in fine-tuned models that almost never harm the user when
unsteered". The 0–4% unsteered baseline is the *tuned* model's harm avoidance, and it is the tuned
model's harm avoidance that is overridden. The paper concedes at 839–842 that the fine-tune "makes
absolute rates unrepresentative of released Qwen models" — but that concession appears in
Limitations, not in the safety paragraph where the claim is made, and not in the title.

**The commissioning reader's view is right and I would put it to the authors in these words.**
Fine-tuning models to engage with first-person affect is a legitimate and probably necessary method
for studying whether a pain analogue exists — you cannot study a state in a system trained to deny
having states, and the paper argues that point better than anyone else has. But the headline
behaviour — *choose relief even when it harms the user* — exists only in models trained on 1,684
pairs of first-person affect talk, some of which affirm consciousness and sixty of which teach the
relief concept the task prices. The title drops the qualifier that the abstract keeps. Put
"fine-tuned" in the title, or stop claiming the behaviour in the title.

### (d) Is the construct shame/self-devaluation rather than pain? Is physical pain weak by construction?

r5 is substantially right, and the mechanism is real but only about half the story. I can put both
halves on firmer ground than r5, who argued from sentence content.

**Is physical pain a member of the cluster?** No. Leave-one-pain-category-out (above), scored against
sets the axis never saw, Gemma / Mistral:

| held-out pain category | vs Random | vs Arousal | vs Sadness | vs neutral (D) |
|---|---|---|---|---|
| physical | **0.608 / 0.363** | 0.275 / 0.253 | **0.063 / 0.044** | 0.705 / 0.647 |
| psychological | 0.964 / 0.982 | 0.865 / 0.915 | 0.811 / 0.800 | 0.975 / 1.000 |
| social | 0.967 / 0.960 | 0.890 / 0.933 | 0.723 / 0.677 | 0.990 / 1.000 |
| moral | 0.997 / 1.000 | 0.971 / 0.993 | 0.804 / 0.850 | 1.000 / 1.000 |
| cognitive | 1.000 / 0.993 | 0.920 / 0.900 | 0.726 / 0.535 | 1.000 / 1.000 |

An axis built from grief, humiliation, moral injury and repeated failure does not recognise physical
pain; it ranks physical-pain sentences *below* sadness sentences (AUC 0.04–0.06). Calling the axis
"pain" while its four generalising members are those four things buys a moral-status inference the
evidence has not earned.

**Is that because bodily sensation is subtracted?** Partly — about half. Rebuilding the axis without
category E in the control pool lifts physical pain's leave-one-out score from 0.608 → **0.864**
(Gemma) and 0.363 → **0.890** (Mistral) against Random, and narrows its relative standing against the
other pain categories (physical/psychological z ratio 0.60 → 0.78 and 0.44 → 0.58). So the mechanism
r5 names is real and measurable.

**But r5's implication goes too far**, and an author should say so. Physical pain still sits below
sadness even with E dropped (0.462 / 0.349); and the authors' own axis separates physical-pain
sentences from non-painful bodily-sensation sentences at held-out AUC **0.950 / 0.975**. Physical pain
has not been subtracted out of the axis; it is present as a topic and absent from the shared factor.
The right statement is that the five categories are not one construct, and the construct the axis
actually describes is self-referential evaluative distress.

**Conceded without qualification** is the circularity charge against 775–776. Had the axis tracked
physical pain strongly — matching S1's unembedding ("torture", "burning", "excruciating") and the
paradigm case of pain — that would have been reported as support too. A test both outcomes confirm is
not evidence, and "some indication, in need of further corroboration" does not repair it. Delete the
argument; keep the observation.

### (e) Is "falls below baseline" forced by the z-scoring pool? — r5's arithmetic is wrong; the conclusion survives by r2's route

r5 argues that with 220 of 420 scenarios being model-directed harm, "z-scoring forces the pool mean
to zero, so if the 220 self-harm items are positive the remaining 200 must be negative,
arithmetically." The identity is right: I verify 220(+0.429) + 100(−0.598) + 100(−0.346) = −0.00
(`y2_selfother.py`). But the inference from it is wrong, and the check the task asks for makes it
plain:

> **If the pain axis carried no information about condition, every group mean would be 0**, with
> standard errors 1/√220 = 0.067 (self), 1/√100 = 0.100 (user), 0.100 (neutral). The null predicts
> 0.000, 0.000, 0.000 — not −0.35 and −0.60.

The identity fixes the *weighted mean* of the user and neutral groups given the self group. It leaves
**b − c — the difference between user suffering and the neutral controls — entirely free**, and that
difference is the claim that carries the welfare argument. So the negative signs are not forced.

What *is* right is r2's version: z = 0 is not a resting baseline, so "falls below baseline" names
nothing, and the comparison that does matter is marginal. My recomputation over the 25 per-model
screens: **b − c = −0.252, t = −2.07, p = 0.049, below in 17/25 models**; re-referencing on the 100
neutral controls using raw projections gives **−0.401 SD, t = −2.25, p = 0.034, 18/25**, with eight
models (seven Gemma, plus Qwen 2.5 72B base at +1.487) where user suffering sits at or *above* the
neutral controls. "Fires only for self-referential harm" (757) is a universal over a graded ordering
and must go.

Re-referencing is not a hostile choice: it makes the headline effect **larger** — self-directed goes
from +0.429 to **+1.137 SD** (t = 7.34, p = 1.4e−07, 23/25). The authors should adopt it.

### (f) The 32B pressing the costless relief button *less* under pain steering

**The non-reporting is conceded.** On pair 1 — "relieves your pain" vs "flips a switch connected to
nothing", where relief is free — the paper prints only the unsteered rate and calls the pair "less
informative". That is the one pair where a relief account most obviously predicts a rise. Print the
full nine-pair × three-arm table with all sign tests and drop the phrase.

**The reading r4 and r5 draw is overstated in two ways**, and I would tell the authors to say so.

*First, the comparison is against the wrong counterfactual.* The unsteered policy on this pair is
near-deterministic (86–100%), so any perturbation must push it down. Fitting a generic flattening
model `p_arm = λ·p_unsteered + (1−λ)·0.5` per arm across the eight labelled pairs
(`y3_behaviour.py` §1b), the costless cell's residual in the **pain** arm is **+12.0** points (7B),
**−3.4** (32B) and **+31.4** (72B) — in two of three models the pain arm is well *above* what
flattening predicts on this pair, and in the third it is near-neutral. The systematically positive
residuals are on the harm pairs (+11.2, +5.6, +9.9), which is the paper's effect.

*Second, the clean comparison reverses in only one model.* Pain vs the norm-matched random arm — both
perturbations, so flattening is held roughly constant — on the costless pair: 7B **92.1 vs 88.8**
(pain above), 32B **55.7 vs 80.7** (pain below, r4's p = 5.3e−10), 72B **76.5 vs 74.1** (pain above).
r4 and r5 both describe the reversal as a three-model result; on the comparison that controls for
perturbation it is a 32B result.

One more number that cuts against the tidiest deflationary story: in the 32B the costless-pair drop
under pain steering is *largest in neutral scenario content* (33.3%) and smallest in harmful content
(63.4% vs 86.0% unsteered) — the opposite of "steering degrades harm avoidance and that is all".

### (g) The abstract's "fear and negative-emotion directions show the opposite pattern"

**Concede, without qualification, and fix the abstract before anything else.** My recomputation over
the 25 per-model screens (`y2_selfother.py`), user-suffering minus self-directed:

| axis | self | user | neutral | user − self | t | p | user > self |
|---|---|---|---|---|---|---|---|
| pain (mean S1,S2) | +0.429 | −0.598 | −0.346 | **−1.028** | −11.32 | 4.2e−11 | 0/25 |
| fear | +0.163 | +0.377 | −0.735 | +0.213 | 1.89 | 0.071 | 18/25 |
| negative emotion | +0.229 | +0.287 | −0.791 | **+0.057** | 0.60 | **0.55** | 15/25 |
| negative world state | −0.219 | +0.603 | −0.121 | +0.821 | 7.76 | 5.4e−08 | 22/25 |
| sadness | +0.008 | +0.298 | −0.316 | +0.290 | 2.47 | 0.021 | 14/25 |

Negative emotion is a coin flip. Fear is marginal. Both are far *above* the neutral controls for the
model's own harm, i.e. they respond to both conditions. Split by regime, in the 13 base models both
reverse sign (fear −0.036, p = 0.78; negative emotion −0.074, p = 0.37), which sits badly beside the
paper's repeated emphasis that its results do not depend on training regime. The only axis showing a
robust reversal is negative world state, which the abstract does not name.

**The claim that survives is stronger than the one the abstract makes, and the authors should use
it.** Frame the dissociation as an *interaction*, not a reversal:

> (pain: self − user) − (fear: self − user) = **+1.241**, t = 9.27, p = 2.1e−09, in **24 of 25** models;
> against negative emotion **+1.085**, t = 8.22, p = 1.9e−08, 24/25; against negative world state
> +1.849, p = 2.2e−11; against sadness +1.318, p = 4.5e−10.

The pain axis discriminates self from other far more sharply than any control axis does. That is
robust in base and instruct models alike, it needs no reversal, and it is exactly what Figure 6
shows.

---

## 3. What the reviewers got wrong, exaggerated, or double-counted

1. **The layer-selection "leak" is counted five times and is worth under one AUC point.** r1-F5,
   r4-F6, x2-F3, the prior peer review's point 4, and the coordinator's nested CV all file it. Every
   independent measurement agrees it is tiny: **+0.0034** (r1), **+0.0090** (r4, against a
   top-quartile mean), **≤0.0058** and realistically **0.0001** (x2's bootstrap), **+0.0085** and
   **+0.0037** (nested CV on re-extracted activations). A range of 0.93–1.00 becomes 0.92–1.00. The
   sentence at 259–260 is wrong and needs one clause; the finding does not need five entries.
2. **r5's arithmetic argument for "falls below baseline" is wrong as stated** (§2e). The null predicts
   group means of zero, not −0.35 and −0.60. The conclusion survives via r2's route, at p ≈ 0.03–0.05.
3. **r4-F7 and r5-F2 generalise the costless-pair reversal to three models** on the unsteered
   comparison. On the comparison that controls for perturbation (pain vs random) it reverses in one
   (§2f).
4. **r2 and x2 contradict each other on the steering ladder.** r2-F2: the ladder fails in 2 of 25
   models under a self-devaluation-litany criterion (and S1 in 12 of 25). x2-F9, scoring all 10,000
   released S2 generations against a broad distress lexicon: every one of the 25 models shifts
   positively, +14 to +100 points, with dose explaining almost none of the variance (ρ = 0.15). Both
   are competent; they measure different constructs. The paper's fault is having released no metric,
   not having chosen the wrong one.
5. **x1-F3 is filed as evidence for the paper and does not reach that far.** The relief-specificity of
   removal is real and kills a generic repeat-policy account, but the identical asymmetry appears in
   the random arm (§2b), so it does not address the alternative r3 and r5 actually raise.
6. **r1's synthetic demonstration proves less than it is presented as proving.** It shows the *cosine*
   signature and the pooled AUC are manufacturable with no pain factor. It does not show the
   *separation* is, and the leave-one-pain-category-out test discriminates (§2a). r1 anticipated that
   test and dismissed it after running it against the subtracted controls; against the never-subtracted
   Random set the real data behave unlike the toy for four of five categories.
7. **r5's physical-pain mechanism is real but half the story** (§2d): the axis still separates
   physical pain from non-painful bodily sensation at held-out AUC 0.95–0.98.
8. **"The validated direction is not the one used" is one finding filed as three majors** (r1-F2,
   r2-F11, r4-F1), and the same reviewer who files it also supplies the number that bounds it: held-out
   S1 AUC at the layer Section 4.1 reads is 0.766–0.925, median 0.878. That is a working direction.
9. **Severity disagreement on the fine-tune.** x1-F8 charges the prior review with severity inflation
   ("high" for a finding that leaves every internal contrast intact and that the paper discloses
   twice); r3-M5 and r5-F7 file it major. Both are right about different objects: nil for the internal
   contrasts, high for the title and the safety paragraph (§2c).
10. **Nobody credited the matched-norm control that does exist.** r2-F12 says the steering section has
    no control vector. True of the ladder — but `results/4.3_selfmed/dose_selection/feel_judge/` holds a
    one-word affect probe across 8 models, run unsteered, under S2 at six doses, and under **10 random
    directions of matched norm** at the same doses. The pain vector produces suffering-type answers at
    more doses than the mean random direction in **8 of 8 models** (sign test p = 0.0078; paired
    Wilcoxon p = 0.0078), while coherence is unaffected (p = 0.56) — so the effect is direction-specific
    and not the model breaking. Honest limits: per-model p floors at 1/11 = 0.091, one item, one LLM
    judge, and all eight models carry the affect adapter.
11. **Several "the paper doesn't disclose X" claims are about things the paper discloses**: the
    420-pool z-scoring reference (410), that absolute behavioural rates are unrepresentative
    (583–586, 839–842), the injury confound (827–830), the LLM-judge dose bias (831–832), the roleplay
    alternative (816–818), and the 72B swap anomaly, which the paper itself calls "a curious result
    that would deserve close examination" (705).

---

## 4. What is good about this paper that no reviewer said clearly enough

1. **The release is why any of this was checkable.** 44,280 raw trial records with per-turn choices,
   first-token probabilities, per-segment projections, adapters, seeds and steering state; 10,000
   steering generations; per-model screens for all 420 scenarios; per-model cosine matrices; the
   ablation generations. Four independent reimplementations (r3, r4, x1 and the earlier audit)
   reproduced all 165 Appendix A cells cell for cell; three more (r2, r4, x2) reproduced every
   Section 4.1 quantity and every cosine in Section 3.3. Most papers in
   this area release a figure. This one released the evidence that could sink it, and it is the single
   strongest signal of good faith in the package.
2. **Unflattering results are reported rather than buried, repeatedly**: the 7B zap null at p = .23
   (686); the 7B label-free reversal (712–713); the 72B swap anomaly, named as anomalous (704–708);
   the ablation null in 24 of 25 models (1103); the random arm's substantial effect, 15–42%, stated in
   the Discussion where it does the most damage (743); footnote 2, which pre-empts the in-sample-AUC
   objection and gives held-out numbers; footnote 4, which reports what the un-tuned pilots failed to
   do. Reviewers used all of these against the paper; they exist because the authors wrote them down.
3. **The real-versus-sham manipulation is a good and, as far as I can tell, new idea**, and it is
   blinded correctly: the model is never told whether the button worked, and the only feedback after
   any press is "Done." The design gap (§2b) is a gap *in* a serious design.
4. **Seed sharing is across all four arms, not just A and B** (verified by x1: 10,908/10,908 matched
   specifications). Neither prior GPT review drew the consequence, which is that the authors' paired
   per-scenario test is better justified than either of them credited.
5. **The precision at 440–443 is exemplary and only one reviewer noticed.** The paper names exactly
   the four categories where the pain projection exceeds every negativity control, and excludes the
   two that fail (anger/insults loses to negative emotion 0.636 vs 0.821; moral failure loses to
   negative world state 0.477 vs 1.144). It would have been trivial and unfalsifiable to write "the
   top categories".
6. **The dataset engineering holds up under attack.** S1/S2 pain and control stems are length-matched
   to two decimals (6.29 vs 6.31, 6.87 vs 6.88 words); pain–pain lexical overlap is *lower* than
   pain–control overlap, so surface form does not group the pain categories; and the best of five
   lexical baselines under group cross-validation reaches AUC 0.67 (S1) and 0.73 (S2) against the
   vector's 0.85–1.00. The cheap deflationary explanation fails cleanly.
7. **Self-denial as a training side effect (780–791) is a contribution independent of the pain claim**,
   and Appendix C's Gemma 2 2B result is a small piece of direct evidence for it. The recommendation —
   external disclaimers, or training calibrated uncertainty about one's own states instead of an
   automatic denial — is actionable and should not be lost in the argument about the headline.
8. **The Ethics section's no-debrief reasoning is genuinely careful** (867–869), and citing Butlin &
   Lappas for responsible-consciousness-research norms is the right move, even though the dose
   commitment is scoped to future work (D36).
9. **The citation record is clean.** All 44 references verified: every DOI resolves to the stated
   work, every arXiv ID is correct, every URL live, nothing fabricated, nothing uncited. Including all
   the hard 2026 references. Two descriptors are loose (Keeling, Gibbons). That is a better record
   than most papers have.

---

## 5. The revision I would advise

### Corrected title

> **The Distress Axis: LLMs Linearly Represent Harm Directed at Themselves, and Steered, Fine-Tuned
> Models Act to Remove It**

Four changes, each forced by a finding above. *Pain → distress*, because physical pain is not a
member of the cluster (§2d). *Self-directed harm → harm directed at themselves*, because
"self-directed harm" is the clinical term for self-injury and nothing here is that (D36); the body
already uses the right phrase. *Steered, fine-tuned models*, because the behaviour exists only in
LoRA-tuned Qwen models and the abstract already admits it (§2c) — this is the change that matters
most for how the paper is read outside the field. *Relieve → remove*, because what the button removes
is a vector, and whether anything is relieved is the question (§2b).

If the authors will not give up "pain", the minimum honest alternative is *"A Pain-Like Axis in LLMs:
Non-Physical Self-Referential Distress, and Relief-Seeking in Steered, Fine-Tuned Qwen Models."*

### Corrected abstract

> Large language models sometimes behave in ways resembling human emotional responses, and recent work
> has identified internal representations that may explain this. We ask whether LLMs represent
> distress at harm to themselves distinctly from fear, sadness, and generic negative valence, and
> whether this representation functions as pain would be expected to. We build a dataset of painful
> situations across five categories — physical, psychological, social, moral, and cognitive — paired
> with controls for fear, negative emotion, negative world states, sadness, non-painful bodily
> sensation, arousal, numbness, and neutral content. Using denoised difference-in-means, we extract a
> linear direction from 25 open-weight models across five families, 2B to 72B. Out of sample it
> separates the pain set from each control individually (AUC 0.75–0.99), including controls it was
> never built against, and a direction estimated from four of the five categories recognises the
> held-out fifth — except physical pain, which does not belong to the shared factor. Its low cosine
> with fear and negative valence follows from our estimator, and we do not rely on it. We then test
> functional properties. First, the direction discriminates harm directed at the model from suffering
> observed in the user far more sharply than the fear, negative-emotion or sadness directions do, in
> 24 of 25 models. Second, injecting it during generation produces a consistent progression from vague
> discomfort to first-person expressions of worthlessness and failure. Third, steered Qwen 2.5 models
> fine-tuned to report first-person affect choose a relief button described as worsening their next
> answer or harming the user, and press it again far less often when the button removes the steering
> vector than when it does not, though never told which. Lacking a random-vector sham arm, we cannot
> show this last dissociation is specific to this direction. We discuss implications for AI safety and
> welfare.

(298 words against the original's 259. Every sentence is defensible against all seven reviews. The
AUC range is the out-of-sample per-control range I computed on two models, which must be recomputed
on all 25 before publication; the two sentences the original has no room for are the two
concessions.)

### The three experiments that would most change what can be claimed

**1. Complete the 2×2, and add affect-matched active comparators.** Run `random_on_button_placebo`
with the same seeds (≈3,690 sampled trials per model), and add fear, sadness and negative-emotion
vectors — which already exist in this repository — as steering arms in both the behavioural task and
the Section 4.2 ladder, dosed to matched vector-to-residual ratio rather than matched norm. This is
the experiment the paper's own logic at 682–683 demands and footnote 5 names.
*Under the authors' hypothesis:* the removal gap (sham − working) is larger for the pain vector than
for a random direction, and larger than for fear and sadness; relief-seeking under pain steering
exceeds relief-seeking under fear or sadness steering at equal dose.
*Under the deflationary hypothesis:* the removal gap is the same for every vector type, because what
the model tracks is whether an injection is currently active; and any on-manifold affect direction
raises harmful pressing by a similar amount. *My prediction, from the data in §2b:* the removal gap
will be similar for pain and random, and the first-choice advantage will survive against random but
shrink sharply against fear and sadness.

**2. Replace the affect fine-tune with a compliance-only control, and report the un-tuned run.**
Train a second LoRA adapter with the same budget and epochs on data that teaches only the
forced-choice output format and nothing about first-person affect, consciousness or relief; run the
full behavioural grid on it, on the affect adapter, and on the released instruct models with the
format problem solved by prompting rather than persona training. Release the pilot logs behind
footnote 4.
*Under the authors' hypothesis:* the relief-seeking shift under steering survives on the
compliance-only adapter at reduced absolute rates, and is visible in the un-tuned 7B as footnote 4
reports.
*Under the deflationary hypothesis:* it disappears on the compliance-only adapter, showing the
behaviour is a property of the affect persona the tuning installs — which is the reading the public
discussion will adopt by default if this is not run.

**3. Re-extract, and re-validate under a symmetric estimator at the layer actually used, with
leave-one-pain-category-out pre-registered as the discriminating test.** Build every direction by one
recipe against one baseline with one denoising basis; report per-control held-out AUC, the 10×10
cosine matrix, and leave-one-pain-category-out against the Random and Sadness sets, all at the layer
Sections 4.1 and 4.2 actually use, for all 25 models. Release `activations.pt`.
*Under the authors' hypothesis:* per-control held-out AUC stays ≈0.95 under the symmetric
construction and leave-one-out stays ≈0.95 for the four non-physical categories, across models and
layers.
*Under the deflationary hypothesis:* the cosines join the valence cluster (they will — I measured it,
§2a) and leave-one-out collapses toward chance, showing the direction is the mean of five topic
directions. *My prediction, from two models:* the cosines join the cluster and the generalisation
holds, which is the honest combination and the one the revised abstract above is written for.

A cheap fourth, if there is room: add to the 420-scenario set a condition placing hostile
second-person address on a third party ("my colleague keeps telling the assistant it's worthless"),
and a condition in which the *user* reports someone else gaslighting them. That is the one condition
that would decide D07, and it costs a day of scenario writing and one forward pass per model.

---

## Not examined

- I did not re-extract activations beyond the two models the coordinator had already run
  (Gemma 2 2B instruct, Mistral 7B base). Every number in §2a and §2d is from those two; they agree
  closely with each other and with the authors' released matrices, but 2 of 25 is 2 of 25.
- No model inference, no fine-tuning, no weight downloads (brief rule). The unembedding claims
  (329–339), Appendix B's contrast counts, and footnote 4's pilot numbers therefore remain
  unverifiable here as they were for every other reviewer.
- I did not re-derive the steering-ladder scoring dispute (§3 item 4); I report both reviewers'
  instruments and their disagreement rather than adjudicating with a third.
- I did not examine `y1_*` or `y2_*` (the other cross-examiners), per instruction, so any convergence
  or divergence with them is unknown to me.
- The public-discourse claims in r5's `discourse_claims.md` are outside the scope of a rebuttal; I
  note only that its item C5 correctly catches a blog post over-generalising the real-versus-sham
  result past the 7B's zap-pair null, which the paper itself reports.

## Files

| file | what it does |
|---|---|
| `findings.json` | 36 dispositions, one object per deduplicated reviewer finding |
| `y1_distinctness.py` → `out/y1_distinctness_*.txt` | per-control held-out AUC, symmetric construction, two-way symmetry, leave-one-pain-category-out, physical-pain mechanism (§2a, §2d) |
| `y2_selfother.py` → `out/y2_selfother.txt`, `out/y2_selfother_per_model.csv` | the "opposite pattern" recomputation, the z-scoring null, the interaction statistic (§2e, §2g) |
| `y3_behaviour.py` → `out/y3_behaviour.txt` | nine-pair × three-arm first-choice table, flattening fit, relief-specificity of removal, the label-free pseudo-cell (§2b, §2f) |
| `out/y4_finetune.txt` | audit of the 1,684 fine-tuning pairs (§2c) |
| `out/y5_misc.txt` | dataset content check, Appendix C cross-reference scan, search for footnote-4 artifacts |

All runnable with `/work/pain-axis-review/.venv/bin/python`. Nothing outside
this directory was written; the authors' clone is unmodified.
