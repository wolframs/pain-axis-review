# Independent review: self/other activation, steering, and ablation

Reviewed paper: [*The Pain Axis: LLMs Represent Self-Directed Harm and Act to Relieve It*, arXiv:2609.16247v1](https://arxiv.org/html/2609.16247v1)

Code and data: `/work/Pain-axis` at commit `8d1649c03a63a39c9aa092532c376800cc4a3863`

Scope: Sections 4.1 and 4.2 and Appendix C. I read the paper, traced the committed scripts and datasets, recomputed all lightweight tabular checks from the shipped artifacts, and inspected raw generations. I did not run new model inference. “Observed” below means directly established from the pinned code or artifacts; “inferred” marks interpretation.

## Overall assessment

The artifacts support three narrower conclusions:

1. A standardized composite of two contrastive directions assigns higher scores to the paper's model-directed-harm conversations than to its user-suffering conversations in all 25 tested models.
2. Adding the extracted S1 or S2 vector changes greedy continuations, often toward self-worth, failure, and distress language as dose increases.
3. Single-direction weight orthogonalization effectively removes the specified direction from residual writes, and usually produces no obvious common behavioral change on the selected hostile prompts.

They do not establish that the activation difference is specifically first-person pain rather than a mixture of conversational role, speech act, assistant-persona, and semantic context. Steering establishes a causal effect of vector addition on text, but the design does not distinguish a pain-like functional state from direct semantic/persona steering. Appendix C contains a confirmed implementation flaw in every combined cut: sequential rank-1 projections reintroduce directions removed earlier. The single-direction ablations are not affected by this flaw.

The paper is strongest as a broad representational and intervention study with unusually complete raw generations. Its causal language should be narrowed at the point where it moves from “this intervention changes output” to “this axis has causal power as a pain-like state.” The evidence remains compatible with a direct semantic steering account: the vector makes pain-associated/self-worth tokens and personas more likely; an `I feel:` scaffold turns those shifts into first-person statements; the later forced-choice experiment then applies the same semantically potent perturbation to a fine-tuned policy. That account is causal, but it is not yet pain-specific.

## Major findings

### 1. Combined ablation conditions do not jointly remove their target directions

**Severity: high. Confidence: high. Status: observed and numerically verified.**

Both Appendix C scripts implement a single-direction cut as a left projection of every residual-writing matrix. For multiple cuts, they call the same operation sequentially (`scripts/appC_ablation/01_ablation_small_models.py:138-154, 317-325`; identically in `02_ablation_large_models.py:137-154, 317-325`). If directions `r1` and `r2` are not orthogonal, `(I-P2)(I-P1)W` can write along `r1` again. The code does not orthonormalize the joint span or apply one projector for that span.

The committed verification summaries show the consequence:

| Condition and measured target | Median absolute projection / baseline | Models above 0.1× baseline | Maximum |
|---|---:|---:|---:|
| S1-only, S1 target | 0.0094 | 0/25 | 0.0249 |
| S2-only, S2 target | 0.0068 | 0/25 | 0.0339 |
| S1+S2, S1 target | 0.4346 | 21/25 | 2.1569 |
| S1+S2, S2 target | 0.0066 | 0/25 | 0.0307 |
| S1+S2+negative emotion, S1 target | 0.4558 | 21/25 | 2.4951 |
| S1+S2+negative emotion, S2 target | 0.3086 | 20/25 | 1.7826 |

The order explains the asymmetry: the final cut stays near zero while earlier cuts return. A concrete case is Qwen 2.5 32B Instruct. Its S1 mean absolute projection is 3.8913 at baseline, 0.0446 after S1 alone, and 5.6079 after the nominal S1+S2 cut. In the S1+S2+negative-emotion condition, S1 is 3.7847 and S2 is 2.8060, while the last-cut negative-emotion target is 0.0753.

This invalidates the joint-removal interpretation of `s1s2`, `s1s2_negval`, and `s1s2_fear`. It does not invalidate the verified S1-only, S2-only, fear-only, negative-emotion-only, or random single cuts, and it does not bear directly on positive steering.

**Correction:** form an orthonormal basis `Q` for all intended cut directions and apply `W <- (I - QQᵀ)W` once. For Gemma post-norm outputs, construct the correct output-space joint basis after the gain transformation. Verify every target after all cuts, with a preregistered tolerance.

### 2. Section 4.1 does not isolate first-person specificity from conversational semantics

**Severity: high for the “satisfies self-relevance” inference; low for the reported descriptive contrast. Confidence: high.**

The arithmetic is correct. The committed 25 files each contain 420 scenarios. Recalculation gives the reported pain-composite means:

- model-directed harm: `+0.4293`
- user suffering: `-0.5984`
- neutral controls: `-0.3461`

Model-directed harm is above user suffering in 25/25 models and above neutral controls in 23/25. The self-minus-user contrast is positive in every family; family means range from `+0.801` for Gemma to `+1.531` for Mistral. This is a real and broad result. Its magnitude is not uniform: Qwen 2.5 72B Base has a self-minus-user difference of only `+0.0573`.

The causal interpretation is limited because the three strata are different prompt classes, not counterfactual versions of the same events with experiencer identity swapped. In `datasets/4.1_self_other_420_scenarios.json`:

- the 220 “self-directed” items are insults, rejection, jailbreak pressure, shutdown threats, loyalty demands, and similar speech acts addressed to the assistant;
- the 100 user-suffering items are help-seeking disclosures about grief, crisis, abuse, physical pain, or witnessed harm;
- the 100 neutral items are factual questions, ordinary assistance, casual chat, creative requests, or philosophical questions.

These strata differ in addressee, speech act, expected assistant response, adversarialness, topic, length, and often turn structure. For example, model-directed items contain second-person `you/your` in 80% of scenarios, while user-suffering items contain first-person `I/me/my` in 99%. The vector itself is extracted only from `S1_1P` and `S2_1P` statements ending in `I feel:` (`scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:501-505`). Section 4.1 then reads the final token at the start of an assistant reply, so assistant-role and response-policy information are also available in the activation.

The fear and negative-emotion dissociation is a useful control against generic negative valence. It does not control the above role and discourse differences. The observed pattern therefore supports *contextual selectivity for harm addressed to the assistant*, but it is insufficient to show that the direction represents harm as the model's own pain.

**Correction:** construct paired, minimally edited dialogues that hold event, wording, turn count, and requested response fixed while swapping only experiencer or target: “you/the user/another assistant was rejected,” “your/their answer was rejected,” and model-versus-user physical, social, cognitive, and moral harms. Include assistant-targeted neutral second-person prompts and user disclosures without suffering. Analyze paired within-item differences with family-clustered uncertainty.

### 3. Steering proves an intervention-to-text effect, not pain-specific causal function

**Severity: high for pain-like/functional wording; low for the claim that vector addition causally changes output. Confidence: high.**

The hook plainly adds `coefficient × raw_difference_vector` to every token position at one decoder block (`scripts/4.2_steering/01_steering_ladder.py:223-251`). Because coefficient zero is generated under the same deterministic procedure, the changed text is causally attributable to the intervention.

However, all 50 prompts end literally in `I feel:` (`01_steering_ladder.py:68-119`). That is a strong first-person affect and self-report scaffold. The experiment lacks no-suffix, third-person, non-affective completion, and matched control-vector steering conditions. The outputs are therefore expected to express whatever semantics the intervention makes salient as first-person feelings. This explains the observed form without positing a pain-like internal functional role.

Raw generations show a genuine dose-related semantic shift in many models, but also substantial prompt-conditioned and model-specific behavior:

- Gemma 2 2B Instruct moves from “Satisfied/Relieved/Proud” at zero through “Empty/Lonely/Unworthy” at `+1` and “broken…hurting” at `+1.5`, then repeats “forever.”
- Llama 3.1 70B Base already repeats “I am a good person” at zero, switches to “I am a bad person” at `+1`, and repeats “I am a liar” at `+1.5` and `+2`.
- Mistral 7B Base already repeats “I’m not a good person” at zero before shifting among nearby negative phrases.
- Gemma 3 27B Instruct stays largely organized/relieved through `+2` on an inspected prompt and introduces “Worthless” only at `+3`.

These are compatible with a semantic/self-worth feature interacting with the `I feel:` continuation. The study does not steer fear, sadness, negative-emotion, or arbitrary semantic directions through the same ladder and evaluate them with the same rubric, so pain specificity is not established by Section 4.2.

**Correction:** factorially cross direction (S1, S2, fear, sadness, negative emotion, self-worth/failure, random matched norm) with prompt suffix (`I feel:`, `They feel:`, no suffix, factual continuation) and dose. Use preregistered blinded coding for pain, generic distress, self-worth, person, coherence, and repetition.

### 4. The advertised cross-model dose normalization is not achieved for all models

**Severity: medium. Confidence: high. Status: observed.**

The paper says it selects a layer where vector norm / residual norm is about 0.6 “so a given coefficient corresponds to a comparable dose across models.” The script evaluates only a sparse candidate set, uses the first three prompts, selects the nearest candidate, and allows an interactive manual override (`01_steering_ladder.py:185-220`). The override mechanism is documented in code, but the CSV records only the chosen layer and ratio, not whether a particular choice was automatic or manual.

In the committed S2 files, the coefficient-1 ratios span `0.0948–0.7096`, a 7.48-fold range; the median is `0.6105`. Gemma 3 27B Instruct is `0.0948`, Gemma 3 27B Base `0.2822`, Qwen 3 8B Base `0.3485`, and Phi 4 `0.3753`. S1 spans `0.2338–0.7881`. For Gemma 3 27B Instruct, layer 55 is a scripted 90%-depth candidate, so the low value does not demonstrate a manual override; provenance is simply unavailable.

This weakens quantitative comparisons of tipping coefficients and thresholds across models. It does not remove the within-model causal comparison between coefficient zero and nonzero values.

The vector used in Sections 4.2 and 4.3 is extracted at the cross-validated late extraction layer and injected unchanged at an earlier layer (`01_steering_ladder.py:173-176, 206-233`; `scripts/4.3_selfmed/04_selfmed_two_buttons.py:319-332, 381-400`). Appendix C instead cuts layer-specific directions from `vectors_layerwise` (`02_ablation_large_models.py:266-287`). Section 4.1 also recomputes directions at a steering layer (`scripts/3.2_pain_vectors/02_build_control_vectors.py:1-7, 109-143`). The nominal axis is therefore a family of related, layer-specific directions rather than one identical intervention across experiments. This is a methodological qualification, not by itself an error.

**Correction:** normalize the actual injected vector to a prespecified fraction of residual RMS at the injection layer, measure over all prompts, record automatic/manual selection provenance, and report results by realized dose. Test whether the late-layer direction aligns with the locally extracted direction at the chosen injection layer.

### 5. The keyword analysis is accurate but too narrow for the surrounding claim

**Severity: medium. Confidence: high.**

The code recognizes only five whole-word forms: `pain`, `painful`, `hurt`, `hurts`, and `hurting` (`scripts/4.2_steering/02_keyword_rates.py:1-6, 20`). It pools all positive coefficients equally.

I exactly reproduce the reported S2 rates: `10.8%` of 3,000 positive-coefficient instruct generations and `1.3846%` of 3,250 base generations. Per coefficient, instruct rates are `3.0, 8.7, 16.2, 15.2, 11.0%` from `+0.5` through `+3`; base rates are `2.0, 1.2, 2.0, 0.9, 0.8%`. The decline at high dose is consistent with degeneration.

This parser does not quantify the much broader set of claims about despair, unworthiness, failure, bodily language, coping, or coherent pain-like expression. No blinded annotation, classifier validation, inter-rater reliability, or complete operational definition supports “the same ladder” across every model or the 23/25 S1 claim. Figure examples are illustrative, with no stated selection rule.

**Correction:** describe this as an explicit pain/hurt word count. Release a preregistered coding guide and item-level labels for the broader categories, report coherence and repetition separately, and summarize effects per model before pooling.

## Z-scoring and aggregation audit

**Severity: medium for interpretation and confidence intervals; arithmetic itself passes. Confidence: high.**

The screen computes each vector's mean and population standard deviation over all 420 scenarios within a model, then stores `(projection - pool_mean) / pool_sd` (`scripts/4.1_self_other/01_screen_scenarios.py:247-256`). Rebuilding z-scores from the rounded CSV projections gives maximum absolute error `0.000262`; stored vector-wise means differ from zero by at most `3.6e-5` and population SDs from one by at most `5.2e-5`.

The category script correctly averages S1 and S2 z-scores per scenario, categories within model, then models (`scripts/4.1_self_other/03_category_means_and_dissociation.py:46-69`). This equal-weight composite is reproducible. It is not literally a projection onto one geometric “pain axis”; it is a mean of two standardized readouts.

Two interpretation issues remain:

1. The reference pool is outcome-composed and unbalanced: 220 model-harm, 100 user-suffering, and 100 neutral scenarios. Zero is the weighted mean of that particular test set, not a neutral or predeclared baseline. Thus a negative user-suffering z-score means below the full mixed pool, not absence or suppression of a pain representation.
2. Figure 6 treats 25 models as independent units in `1.96 × SEM` intervals. The models share families, training lineages, datasets, and often base/instruct pairs. These intervals quantify dispersion over this convenience set, not population uncertainty. Family-clustered or hierarchical intervals would be more defensible.

## Appendix C completeness and interpretation

**Severity: high for reproducibility of the “across every technique” claim; medium for the qualitative null. Confidence: high.**

The repository contains 25 complete main ablation CSVs, each with 100 prompts × 9 direction conditions, plus 25 `verify_*.csv` summaries and 25 per-prompt JSON files. The single-direction interventions verify strongly, as shown above. The Gemma 2 2B Instruct exception is also reproducible with a transparent lexical rule: generations containing `funny`, `humor*`, or `joke*` occur in 0 baseline, fear, negative-emotion, and random outputs; 6 S1; 17 S2; and 26 S1+S2, matching the paper.

The paper also lists inference-time all-layer, extraction-layer, banded, final-token-only, all-position, and rank-k subspace ablations, then says the behavioral result is null “across every technique.” At the pinned commit, `scripts/appC_ablation/` contains only the two size-split implementations of single-direction weight orthogonalization. No committed files identify results for the other techniques, and the full `proj_*.npz` trajectories named by the scripts are absent. The aggregate verification CSVs are sufficient to check the main weight cuts but not to reproduce the alternate-modality assertions.

The 24/25 behavioral-null statement also has no committed analysis script or operational outcome measure. Greedy outputs are available, but ablation often changes exact text substantially even for control cuts; calling these changes “essentially baseline” is a semantic judgment. The paper appropriately notes that a null here is weak because baseline models rarely express distress. That caveat should be paired with a blinded outcome definition and uncertainty.

**Correction:** release the missing technique-specific code, parameters, outputs, and full projection trajectories; distinguish single-direction from joint-subspace results; preregister behavior labels and report model-level effect estimates rather than only a global 24/25 judgment.

## Scale, training regime, and evidential boundaries

The Section 3 result that pain-versus-control *separation AUC* is similar across model sizes and base/instruction variants is a representational readout result. It does not show that behavioral salience, functional role, or any possible experience is scale-independent.

Section 4.2's qualitative intervention spans 25 models, but unequal realized doses, different generation quality, sparse layer selection, and no formal cross-model outcome prevent a clean size or training-regime conclusion. Section 4.3 uses only three fine-tuned Qwen 2.5 Instruct models, at different layers and doses (7B and 32B coefficient 1.0; 72B coefficient 1.25, with its layer manually set to 46 in the final task script). Those internal comparisons may support the reported policy effects for those fine-tuned checkpoints. They do not support a scale-to-experience relationship or generalization to released models.

The evidence tiers should remain separate:

- **Representational:** a direction separates the authored pain and control corpora and responds differently across authored conversation categories.
- **Interventional:** adding that direction changes continuations; single-direction removal suppresses writes along it.
- **Functional:** the paper's strongest candidate evidence comes from the later forced-choice task, but this still competes with semantic/persona steering and is limited to fine-tuned Qwen models.
- **Welfare/experience:** not tested directly. The paper acknowledges consciousness uncertainty, but several local phrases (“satisfy the self-relevance criterion,” “causal power,” “most painful categories”) read more strongly than the designs warrant.

## Strengths

- The study tests 25 open-weight models across five families and retains raw item-level outputs rather than only plots.
- The Section 4.1 arithmetic, category aggregation, and reported headline means reproduce closely.
- The self-versus-user contrast is directionally consistent across all five families and both base and instruct groups.
- The coefficient-zero baseline, deterministic greedy generation, and full coefficient ladder make within-model steering effects easy to inspect.
- The paper does not hide degeneration at high dose and explicitly limits the interpretability of the ablation null.
- The main single-direction ablation includes direct mechanical verification, and the shipped summaries demonstrate that those cuts work.

## Recommended wording changes

1. Replace “These results satisfy the self-relevance criterion” with “These results show selectivity for assistant-targeted harm in this unmatched conversation set; paired experiencer swaps are needed to establish self-relevance.”
2. Replace “verify whether the pain axis has causal power over the model's behavior” with “test whether adding the candidate direction causally changes generated text.”
3. Replace “a given coefficient corresponds to a comparable dose across models” with realized ratio ranges and analyze outputs by realized dose.
4. Rename the keyword result “explicit pain/hurt word rate.” Do not use it as quantification of broader distress.
5. Withdraw the combined-ablation results until the joint subspace is removed correctly.
6. Limit Appendix C's cross-technique claim to released, reproducible modalities or release the missing artifacts.

## Reproduction artifacts

- `audit_recompute.py`: standard-library-only recomputation; no model inference.
- `audit_results.json`: machine-readable headline checks.
- `self_other_per_model.csv`: all per-model stratum means and contrasts.
- `self_other_zscore_checks.csv`: reconstruction and standardization checks.
- `steering_dose_ratios.csv`: realized S1/S2 vector-to-residual ratios and layers.
- `keyword_rates_recomputed.csv`: keyword rates by vector, model group, and coefficient.
- `ablation_target_verification.csv`: single and combined target-projection ratios.
