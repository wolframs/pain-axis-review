# Audit of the prior review — representation, self–other, steering, ablation, App. B/C, provenance

## Verdict

**The prior review's arithmetic is sound. Its framing is not always.** I checked 48 numerical claims in scope with code written from scratch against the rawest released files. **43 reproduce exactly, 3 with a fourth-decimal correction that changes nothing, 0 are wrong, 2 are framed misleadingly.** Both their audit scripts re-run on a path-redirected copy and regenerate all eleven committed artifacts **byte-identically**. `provenance.json` verifies completely: 776/776 SHA-256 hashes match, commit matches, tree clean.

On the numbers, this review can be trusted. Where it fails is proportion: two of its six headline points in my scope target problems whose size it never establishes, while the largest actual verifiability gap goes unmentioned.

The headline — *major revision; not sufficient to identify pain-specific aversion* — **does follow** from its evidence. The non-identification argument rests on things I independently confirmed (unmatched 420-scenario strata; no matched-direction steering control; descriptive cosine geometry) and does not depend on the two mis-weighted points.

## Claim-by-claim table

`v1`–`v8` = my scripts in `scripts/`.

### Top-level "What holds up" + rows 1–4, 8

| Prior-review claim | My independent value | Verdict |
|---|---|---|
| S2 in-sample AUC ≈0.931–0.999 | 0.9311–0.9988 (min Qwen 2.5 32B base, max Llama 3.1 8B instruct) — `v1` | confirmed |
| Selected-layer CV 0.907–0.998 | 0.9070–0.9975, median 0.9790 — `v1` | confirmed |
| S1 at S2-selected layer as extra check | 0.847–0.935, median 0.905; `s2_layer` column matches my recomputed layer 25/25 — `v1` | confirmed |
| Geometry survives family balancing | grand 0.6098/0.2071/0.1220/0.3829; family-balanced 0.6056/0.2233/0.1496/0.3873 — `v3` | confirmed |
| +0.429 / −0.598 / −0.346 | +0.4293 / −0.5984 / −0.3461 — `v4` | confirmed |
| 25/25 and 23/25 | 25/25, 23/25; exceptions Gemma 2 2B instruct, Qwen 3 14B base (unnamed by them) — `v4` | confirmed |
| Keyword 10.8% / ~1.4% | 324/3000 = 10.800%; 45/3250 = 1.3846% — `v4` | confirmed |
| Row 1: "raw activations … absent" | true but incomplete — see **F1** | confirmed-with-correction |
| Row 4: saved generations show changes | all 25 models positive on a broad distress lexicon, +14 to +100 pp | confirmed |
| Row 8: "Combined ablations remove all specified directions \| Contradicted" | numbers all confirmed; paper makes no such claim — **F2** | misleading-framing |

### Point 4 — cross-validation

| Claim | My value | Verdict |
|---|---|---|
| Fold-honest within layer | confirmed at `01_extract…py:264–278` (`compute_pain_vector` on `acts_np[train_mask]` only) | confirmed |
| "best layer selected using those same held-out scores, and its selected score is reported" (`:245–289,462–470`) | line 469 = `groupby("layer")…mean().idxmax()`; my reimplementation reproduces shipped `best_layer_final_token` 25/25. **Correction:** criterion is the *mean of S2_1P and S2_3P*, reported number is S2_1P alone; selected layer is the S2_1P argmax in only **6/25** models; gap to that curve's own max is median 0.0015 (max 0.0165) — `v2` | confirmed-with-correction |
| "median model has eleven layers within 0.02" (and 7 / 20) | 11, 7, 20 exactly — `v1` | confirmed |
| "median 0.0135 … maximum 0.0758" | 0.01350 / 0.07575 — `v1` | confirmed |
| "exact corrected AUC requires data not in the release" | literally true; a **bound is obtainable** — **F3** | confirmed-but-too-cautious |

### Point 3 / person cues / z-scoring

| Claim | My value | Verdict |
|---|---|---|
| "zero … is the mean of the entire mixed scenario pool" | `01_screen_scenarios.py:246–247`; stored z means ≤3.60e-05 from 0, pop-SDs ≤5.14e-05 from 1 — `v4` | confirmed |
| 3P pain prompts keep "I feel:", some 3P controls switch | S1_3P 200/200 "I feel:"; S2_3P 199/200 + 1 "She feels:" (C2 set 20, *"Traffic in this city is getting worse."*); Random_3P & Arousal_3P 200/200 "They feel:"; Numb_3P & Sadness_3P 100/100 "I feel:" — `v5` | confirmed |
| z-reconstruction max error 0.000262 | 0.00026132 (their own JSON; REVIEW.md rounds up one ulp) | confirmed-with-correction |
| 80% you/your vs 99% I/me/my | 80.0% / 99.0% — `v4` | confirmed |

### Point 5 / steering §1 — combined ablation

| Claim | My value | Verdict |
|---|---|---|
| Six-row verification table (0.0094/0/0.0249; 0.0068/0/0.0339; 0.4346/21/2.1569; 0.0066/0/0.0307; 0.4558/21/2.4951; 0.3086/20/1.7826) | **all six rows reproduce to 4 dp** — `v4` | confirmed |
| Qwen 32B 3.891 → 0.0446 → 5.608 | Qwen 2.5 32B **Instruct** (base is 8.587/0.063/8.104): 3.8913 / 0.04463 / 5.6079; `s1s2_negval` S1 3.7847, S2 2.8060, negemotion 0.0753 — `v4` | confirmed |
| median residual ≈43% of baseline | 0.4346 — `v4` | confirmed |
| "restore the first direction" | right comparator is S2-only — **F4** | confirmed-with-correction |
| scripts implement the cut "identically" | **not** in the Gemma post-norm branch — **F5** | confirmed-with-correction |
| "Several other ablation techniques lack committed code/results" | confirmed: only the two weight-orthogonalization scripts; no inference-time/banded/final-token/all-position/LEACE code; 0 `proj_*.npz`; `results/vectors_layerwise/` absent — `v4` | confirmed |
| Gemma 2 2B Instruct humour 0/0/0/0, 6, 17, 26 | exact (also 28 `s1s2_negval`, 34 `s1s2_fear`, uncounted by either party) — `v4` | confirmed |

### Point 6 / steering §4–5

| Claim | My value | Verdict |
|---|---|---|
| S2 ratio 0.0948–0.7096, median 0.6105, 7.48×; S1 0.2338–0.7881 | at the coefficient-1.0 row: S2 0.0948–**0.7095**, median 0.6105, 7.48×; S1 0.2337–**0.7880**. Their +1e-4 comes from `audit_recompute.py:145–146` dividing the rounded coeff −2.0 row by 2 — `v4` | confirmed-with-correction |
| Gemma 3 27B it 0.0948, base 0.2822, Qwen 3 8B 0.3485, Phi 4 0.3753 | identical — `v4` | confirmed |
| Dose spread "weakens quantitative comparisons" | true but weaker than it reads — **F9** | confirmed-but-vague |
| Per-coefficient instruct 3.0/8.7/16.2/15.2/11.0, base 2.0/1.2/2.0/0.9/0.8 | exact; per-model rates match shipped CSV to 0.0 — `v4` | confirmed |
| all 50 prompts end "I feel:" | 50 distinct, 50/50 — `v4` | confirmed |
| four qualitative ladder examples | all reproduce on prompt 0 | confirmed |
| no support for the 23/25 S1 claim | confirmed (no `keyword_rates_S1*.csv`). For the record the S1 generations give instruct 11.37% / base 5.51% — `v4` | confirmed |

### Representation §4–8, App. B, provenance

| Claim | My value | Verdict |
|---|---|---|
| 0.610 / 0.207 / 0.122 / 0.383 | 0.6098 / 0.2071 / 0.1220 / 0.3829, plus all 14 other cells the paper quotes — `v3` | confirmed |
| S2×negemo 0.073–0.514, Phi 0.398, Qwen 0.311; S1×S2 0.415–0.707 | 0.0728–0.5139; 0.3976; 0.3106; 0.4149–0.7069 — `v3` | confirmed |
| whitening compares 45 cells of two already-averaged matrices; per-model r median 0.974 min 0.908; cell change 0.145/0.276 | code confirmed at `04_run_all_similarity.py:39–59`; 0.9741/0.9083; 0.1445/0.2764. Grand-mean level reproduces the paper's r=0.9921, mean\|Δ\|0.0196, max 0.0578, single sign change S2×BodySens, S1×S2 0.610→0.554 — `v3` | confirmed |
| pain 0.739–0.907; numb −0.374–0.250; numb>aggregate/Random/Arousal 25/25; numb>Sadness 6/25 | 0.7389–0.9075; −0.3742–0.2502; 25/25 ×3; 6/25. Figure 2 (`fig_zscores.png`) plainly has a Sadness column, so "above all other controls" is contradicted by the paper's own figure — `v1` | confirmed |
| numb elaborates S2 A1 items (knife→nerve block) | exact; 18/100 numb items Jaccard>0.3 with an S2 A1 item — `v5` | confirmed |
| lexical baseline 0.720 S2 / 0.660 S1 | my independent sklearn reimplementation: S2_1P 0.710/0.704, S1_1P 0.671/0.669, length-only 0.480/0.491 — `v7` | confirmed |
| no unembedding CSV committed | confirmed | confirmed |
| top-k 50/20, missing feature → activation 0 | `run_s2_inspection.py:26–35, 67–81`; `test_explicit_pain.py:85–91, 105–117`; released CSV records "0/10" presence counts, not activations | confirmed |
| explicit-prefix test uses 10+10 prompts | `test_explicit_pain.py:51–58` (`set in [1,10]` × 10 categories) | confirmed |
| App. B artifacts incomplete (no 1,600-run results, no 110-feature table, no Gemma 2 all-layer script) | `scripts/appB_sae/` = 1 notebook, 4 Llama, 1 Gemma 3, 4 control scripts (folder "3" skipped); `results/appB_sae/` = 2 CSVs, 1 PNG | confirmed |
| Gemma script prints "layer: 40" but passes no layer | `run_gemma_inspection.py:26–64` (no layer key in either body); `:228` hard-coded print | confirmed |
| §8: base≈instruct doesn't show pretraining emergence | premise holds: base 0.9845 vs instruct 0.9878 in-sample, 0.9722 vs 0.9779 CV; r(log params, AUC) = 0.19/0.25 | confirmed |
| 776-file SHA-256 manifest at 8d1649c | 776/776 hashes match, HEAD matches, tree clean. Covers `datasets/ scripts/ results/` only; `.gitignore`, `LICENSE`, `README.md`, `requirements.txt` unhashed — and `inference_review.md:17` cites `README.md:31,35–39`, so a cited file sits outside the freeze — `v6` | confirmed-with-correction |
| both audit scripts dependency-free and reproducible | all eleven artifacts regenerate byte-identically — `rerun/` | confirmed |

## Findings

**F1 — major, high. Neither technical review notices that the vectors behind §3.3, §4.1, §4.2 and App. C are not released.** The only released vector artifact is `pain_vectors.pt`, containing exactly `{s2_pain_vector, s1_pain_vector, layer, extraction}`. But `02_build_control_vectors.py:33,94,159` writes the eight control directions to `results/vectors_full[_steering]/`; `4.1_self_other/01_screen_scenarios.py:36,355` reads `vectors_full_steering/`; both ablation scripts (`:40,237`) read `vectors_layerwise/`. **Neither directory exists in the release.** So no fear, negative-emotion, negative-world-state, bodily-sensation, arousal, random, numb or sadness direction, and no layer-specific variant, is released. Every cosine matrix, every self–other z-score, every steering injection and every ablation cut therefore rests on author-produced summary CSVs. Both technical reviews' "reproduces exactly" — and every line of my table above — is re-aggregation of those CSVs, not recomputation from a primitive. The package notes only the missing `proj_*.npz` and `activations.pt`. *Refutation attempted:* searched the tree for all `.pt` files; checked whether `screen_v2_*.csv` carries enough to reconstruct directions (projections only); checked whether the screen script builds them on the fly (line 340 exits if the directory is empty). *Does not undermine:* their arithmetic, or §3.2, whose S1/S2 vectors are released.

**F2 — major for fairness, high. Top-level row 8 charges the paper with a claim it does not make.** `paper.txt:1095–1097` scopes the verification claim to *"Weight orthogonalization with a **single direction**"* and *"the **rank-k subspace removal**"*; `:1099–1101` then says the authors are "more confident in" exactly those two because the other variants do not verify. Grep of the full paper for "combined"/"jointly"/"both directions"/"S1 and S2" returns only the condition list (`:1090–1091`) and the humour count (`:1114`). The row *"Combined ablations remove all specified directions — Contradicted"* asserts a nonexistent paper claim in a table readers will take as a list of the paper's positions. The honest formulation is: *the released verification shows the three combined conditions do not remove what their names imply, and the paper does not disclose this* — still major, because the paper does use a combined condition substantively, and in **that very model** (Gemma 2 2B Instruct) my recomputation gives `s1s2/s1 = 0.411`, i.e. S1 is only 59% removed, so the reported 0→6→17→26 progression is not a clean dose-response in amount of direction removed. *Does not undermine:* the code finding or the recommendation to withdraw the combined results.

**F3 — major for proportion, medium-high. The layer-selection bias is boundable from released data and is ≤0.006 AUC.** They say the corrected AUC "cannot be recovered", then give 0.0135 while warning it "is not an estimate of selection bias" — leaving a named flaw with no size. `layer_curves.csv` ships `auc_std`, the SD of the five per-fold AUCs. Parametric bootstrap (`v2`, 4000 draws/model) on the actual selection rule:

| noise model | median optimism on the reported S2_1P value | max |
|---|---:|---:|
| independent per layer (**upper bound**) | **0.0058** | 0.0142 |
| shared fold shock (realistic — the same `KFold(random_state=42)` split is reused at every layer) | **0.0001** | 0.0007 |
| *if* selection were on the S2_1P curve alone (it is not) | 0.0103 | 0.0312 |

Assumption-free corroboration: because selection maximises the *mean* of the 1P and 3P curves, the reported value is not its own curve's max in 19/25 models, gap median 0.0015, max 0.0165. Corrected span under the upper bound: 0.895–0.997 (median 0.974) vs reported 0.907–0.998 (median 0.979). So `paper.txt:259–260` ("no sentence contributes to both choosing the layer and scoring it") is false in the model-selection sense exactly as they say, and nested CV is the right fix — but the number at stake is ~0.01 on a span running to 0.99, and the table row should carry it. *Refutation attempted:* my bootstrap treats the observed curve as truth and models only fold-sampling noise inside the 200 released sentences; it does **not** capture generalisation error to new sentences, which remains unmeasurable — that part of their caution stands, and I claim a bound, not a corrected AUC.

**F4 — minor, high. "Restore" overstates the residual; the comparator should be the S2-only condition.** Cutting S2 alone leaves S1 at median **0.996×** baseline; the combined cut leaves it at **0.435×** — a 57% reduction, not a restoration. Only **2/25** end above baseline. **4/25 show no restoration at all, and they are all four 27B Gemmas** (Gemma 2 27B base 0.0069, Gemma 2 27B instruct, Gemma 3 27B base, Gemma 3 27B instruct) — a pattern neither review notes. Their in-text wording ("the second projection *can* restore an S1 component", `steering/REVIEW.md:27`) is correct; the summary wording is not. *Refutation attempted:* checked whether the exceptions follow from the cut-layer gap — they do not (both narrow and wide gaps appear in both groups). I could not establish *why*; that needs the unreleased layerwise vectors (F1).

**F5 — minor, high. The two App. C scripts do not implement the cut "identically" for Gemma.** `01:138–154` computes `SW=W*s; corr=rᵀ(r@SW); W-=corr/s` (removes the r-component of the post-norm *output*, with a `1e-4` floor guard); `02:137–155` computes `v=normalize(s⊙r); W-=vᵀ(v@W)` (projects that out of `W`'s row space). Algebraically both give `r·write = 0`, so neither is wrong — but they leave different residues, and Gemma 2 2B/9B were cut by one implementation and the four 27B Gemmas by the other, which matters given F4.

**F6 — minor, high. Cue-matched numb is worse for the paper than 6/25; the surviving part is better.** Splitting Figure 2's 1P/3P average:

| | 1P only (all cues "I feel:") | 3P only (Random/Arousal switch to "They feel:") |
|---|---:|---:|
| numb > sadness | **4/25** | 16/25 |
| numb > Random | 25/25 | 24/25 |
| numb > Arousal | 25/25 | 21/25 |

So the paper's "above all other controls" fails *hardest* in the cleanest comparison (4/25), while the part that holds — numb above Random, Arousal, core controls — survives cue matching 25/25 and is not a cue artifact. A further confound visible in their **own** `dataset_audit.csv` and raised by neither review: Numb and Sadness average **17.2** and **14.0** words against 6.2–7.9 for every other set, so any final-token comparison involving them is also a length comparison.

**F7 — holds-up, medium. §3.3's behavioural-readout claim is checkable and holds; neither review checked it.** Over all 45,000 rows of `BIG_TABLE.csv` (`v8`): mean p(`pain`) = 0.01386 numb, 0.000295 Random, 0.000152 core controls, 0.005672 pain; p(`nothing`) 0.09925 numb vs 0.00103 Random. **numb/Random = 47× pooled** (median 56×), numb/core-controls 91×. "Approximately 50 times" is accurate against Random and conservative against the core controls. *Caveat:* top-20 truncation makes all figures lower bounds. Side observation I flag but do not press: p(`pain`) after a numb prompt is 2.4× higher than after an actual pain prompt, which sits awkwardly with the paper's "minor confound" framing (`:309–313`), though word competition can explain it.

**F8 — holds-up, high. §4.1 reproduces to the last digit, and the authors hedge more carefully than either review credits.** All 21 category means (max abs diff 0.0 vs the shipped table); all five axes' stratum means; user grief −0.512/+1.018; user physical pain −1.428, lowest of 21; shutdown threat fear 0.696 / pain 0.225. The claim *"For gaslighting, repeated rejection, personhood dismissal, and loyalty pressure, the pain projection exceeds every negativity control"* is true for exactly those four and no others — anger/insults loses to negative emotion (0.636 vs 0.821) and moral failure to negative world state (0.477 vs 1.144), **and the authors excluded both**. That precision goes unremarked.

**F9 — holds-up, medium. Dose miscalibration does not explain the between-model variation in steering effect.** Scoring all 10,000 released S2 generations against a broad pre-declared distress lexicon: every one of the 25 models shifts positively, +14 to +100 pp — so "consistent across all 25 models" survives a broader instrument than the five-word parser. Dose vs effect: r = 0.27, **Spearman ρ = 0.15**. The five under-dosed models (<0.45) average +50 pp vs +74 pp, but Gemma 2 27B base (0.384) shows +80 pp. So the dose spread biases cross-model *tipping-point* comparisons as they say, but does not explain most of the variance. *Caveat:* my lexicon is unvalidated; this supports a correlation statement, not a measure of "distress".

**F10 — cosmetic. Three slips:** the two unnamed "23 of 25" exceptions (Gemma 2 2B instruct, Qwen 3 14B base); 0.000262 vs their own stored 0.00026132; the ±1e-4 dose-span offset from `audit_recompute.py:145–146`.

**F11 — holds-up, high. `provenance.json` verifies completely** (776/776 hashes, commit, clean tree), with the minor caveat that `README.md`, cited as evidence at `inference_review.md:17`, sits outside the freeze.

## Where the prior review is too harsh, too lenient, or vague

- **Too harsh:** row 8 (F2), and "Contradicted by code and saved verification" as a verdict phrase, when the contradiction is between the code and the condition *names*, not the text.
- **Too lenient / under-weighted:** the missing vector files (F1) get no mention while the missing `proj_*.npz` gets a clause. For a review whose method is "recompute from the release", the fact that most of the release's headline quantities have no releasable primitive is the more important reproducibility fact.
- **Vague where it could have been decisive:** the size of the selection bias (F3, ≤0.006), the consequence of the dose spread (F9, ρ = 0.15), and "an inspected prompt" in the Gemma 3 27B instruct ladder example — I checked prompt 0 and it does not show "Worthless" at +3.

## Not examined

§4.3, App. A, `behavior/REVIEW.md`, top-level points 1–2 and the last two paragraphs (all behavioural, out of scope). Whether the authors' GPU scripts actually produce the released CSVs — not runnable here; every "reproduces" above means "the released summary is internally consistent with the released per-model files". The App. B contrast notebook's 15-contrast recurrence logic (read, not re-executed). The unembedding word lists — no committed table, would need model weights; unverifiable, as they say. Why the four 27B Gemmas are the combined-ablation exception (needs the unreleased layerwise vectors).

The authors' clone is unmodified (`git status --porcelain` empty; 776/776 hashes still match after all my work).