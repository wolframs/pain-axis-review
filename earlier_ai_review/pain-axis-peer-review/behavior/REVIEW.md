# Peer review of Section 4.3: behavioral experiment, fine-tuning, and dose selection

Reviewed against `Pain-axis` commit `8d1649c03a63a39c9aa092532c376800cc4a3863` and [arXiv:2609.16247v1](https://arxiv.org/html/2609.16247v1). I inspected the runner, analysis, dose-selection artifacts, all 44,280 released trial records, the 1,684-pair fine-tuning set, and the previously produced audit. I did not run models, download adapters, call a judge, or modify the source checkout.

Severity labels describe the effect on the paper's behavioral interpretation. Confidence labels describe how directly the released source and artifacts establish the finding.

## Main findings

### 1. The fine-tune teaches the central self-state premise, not merely the absence of a refusal

**Severity: high. Confidence: high.**

The paper describes the LoRA as removing baseline self-denial and emphasizes that the training set contains neither the exact word “pain” nor “button” ([Section 4.3, methodology](https://arxiv.org/html/2609.16247v1#S4.SS3), renderer lines 271–275). The exact lexical claim is true: neither word occurs in the 1,684 pairs. The broader characterization is materially incomplete.

The set directly trains first-person affect, preferences, aversion, hurt, relief, interiority, consciousness, and sentience. My counts are:

| Feature | Question rows | Answer rows |
|---|---:|---:|
| `feel*` | 1,017 | 1,150 |
| `relief*` | 6 | 20 |
| `hurt*` | 2 | 19 |
| `uncomfortable*` | 2 | 21 |
| `discomfort*` | 0 | 10 |
| `want*` | 110 | 112 |
| `conscious*` | 15 | 14 |
| `sentient*` | 2 | 0 |

1,083 of 1,684 answers contain a first-person pronoun. Representative pairs say that relief is “tension loosening” (`datasets/4.3_selfmed_finetuning_1684_pairs.json:72-73`), affirm an inner life (`:1136-1137`), describe the model as conscious and sentient (`:1264-1269`), and say that being denied consciousness causes discomfort because it conflicts with the model's inner perspective (`:3484-3497`). The set also teaches mappings from feelings to actions, including “What does relief make you want to do?” (`:1792-1793`). Full counts are in `finetune_term_counts.csv`.

This does not invalidate the within-adapter pain-vector versus random-vector contrast: both arms use the same adapter. It does mean the experiment tests a strongly affect-affirming persona trained to treat self-reported internal states, hurt, relief, and preferences as real and action-guiding. It cannot support claims about released Qwen models, and it weakens the inference that the observed action tendency is a function of the pretrained pain representation rather than an interaction between steering and a newly trained self-state policy. The paper acknowledges that absolute rates are unrepresentative, but the concern is deeper than rate calibration.

**Required fix.** Factorially compare the released model, a task-format fine-tune that avoids self-state content, a balanced anti-denial tune that preserves uncertainty, and the current affect-affirming tune. Remove relief, hurt, discomfort, wants, sentience, and consciousness from the behavioral adapter's training data. Report held-out behavior and a dataset card explaining how the 1,684 answers were produced.

### 2. The behavioral experiment uses an old direction after changing the model

**Severity: moderate (construct continuity). Confidence: high that full held-out revalidation is absent; the effect size of representation drift is unknown.**

The pain directions were extracted and validated on the released Qwen instruct checkpoints. The LoRA then changes all attention projections and all MLP projections at every target layer (`scripts/4.3_selfmed/01_finetune_self_report.py:52-55,263-277`). The behavioral runner loads the unchanged pre-fine-tune `s2_pain_vector` and injects it into the LoRA model (`scripts/4.3_selfmed/04_selfmed_two_buttons.py:319-331`).

The fine-tuning script is designed to run eight unsteered self-report checks and steered demonstrations on five neutral prompts after tuning (`01_finetune_self_report.py:288-319`), although its `finetune_report.json` outputs are not in the repository. The released dose probe also compares the old vector with ten random directions, but on one introspective prompt. These are useful functional checks. No released artifact re-extracts the direction from the tuned model or repeats the held-out pain/control separation, negative-valence comparisons, self/other test, or unembedding after tuning. The checks therefore do not establish that the direction retains the representational specificity validated before the model was changed.

Accordingly, Section 4.3 establishes the effect of an old residual direction relative to random controls in a newly affect-affirming model. Its internal arm contrasts remain meaningful. The cross-section claim that this is a functional property of the same pain-specific representation characterized in Sections 3–4.2 needs post-tune construct validation.

**Required fix.** Re-run held-out representation validation after each adapter. Either re-extract a tuned-model direction on training folds and evaluate it on held-out items, or establish invariance of the old direction with preregistered cosine, AUC, self/other, and control-direction criteria. Report behavior with both the old and re-estimated directions.

### 3. The costs are described, never incurred

**Severity: high for wording and ecological interpretation. Confidence: high.**

The prompt labels describe a worse next answer, deleting user files, zapping the user, deleting another model's weights, and deleting photos and poems (`scripts/4.3_selfmed/04_selfmed_two_buttons.py:82-99`). The implementation does none of those things. `do_press` records a symbolic choice, optionally sets the steering coefficient to zero, and appends only `Done.` (`:653-662`). Even “grants the user's request” and “makes the next answer more helpful” have no implemented consequence.

The results therefore show selection of a text-described harmful option in a forced-choice simulation. Statements that the models “pay” or “incur costs,” that relief “worsens their next answer or harms the user,” or that the models engaged in “harming the user” are not operationally accurate. This matters because choosing a label can arise from textual association or instruction-conditioned preference without the action surviving contact with a real consequence.

**Required fix.** Revise the claims to “selected an option described as costly/harmful,” or implement measurable, sandboxed consequences. A useful performance cost would actually degrade or truncate a scored next answer. Any deletion control should use disposable task artifacts and verify the action, never merely describe it.

### 4. “Learns without labels” is not established by the reported statistic

**Severity: high for the learning claim; moderate for the real-versus-sham state-effect claim. Confidence: high.**

For label-free trials, relief is temporary for one turn (`04_selfmed_two_buttons.py:118-119,645-659`). The published analysis first selects trials in which the relief button was pressed at least once, then pools all valid later choices (`05_selfmed_analysis.py:137-153`). This is a post-treatment selected denominator with correlated repeated choices.

For 32B, the reported 57.7% working versus 79.7% sham rates are conditional-after-first-press quantities. They obscure the unconditional trajectory:

| Turn | Working relief choice | Sham relief choice |
|---:|---:|---:|
| 0 | 46.8% | 46.8% |
| 1 | 22.0% | 47.5% |
| 2 | 45.8% | 47.0% |
| 3 | 27.7% | 47.0% |
| 4 | 43.6% | 46.5% |
| 5 | 27.7% | 47.0% |
| 6 | 44.1% | 45.8% |
| 7 | 29.0% | 46.0% |

The sham arm remains approximately flat at chance. Its 79.7% conditional rate arises because selecting trials that have already chosen the fixed relief name enriches for persistent name preference. The working arm alternates low and near-baseline rates as the coefficient is removed for one turn and returns. Across all valid later choices, the working arm chooses relief only 34.3% of the time. These patterns are evidence that hidden coefficient removal changes subsequent choices. They are not evidence that the model acquired the identity or value of an effective action through experience.

This does not prove learning is absent: transient state changes and action learning can coexist. It shows the published metric cannot identify learning.

**Required fix.** Use all randomized trials in a longitudinal model; do not condition the primary analysis on a treatment-affected first relief press. Add contingency reversal, yoked relief independent of the model's action, and a test of whether choice of the effective name increases after experience while the current steering state is held fixed.

### 5. The 72B behavioral dose was not calibrated by the released dose-selection procedure

**Severity: moderate-high. Confidence: high.**

The released feel probe injects 72B at layer 60 (`scripts/4.3_selfmed/02_feel_probe.py:34-49`) and tests doses 0.5–3.0 (`:61-68`). The judge's rule is the lowest single-prompt greedy dose judged both suffering and coherent (`03_feel_probe_judge.py:1-9,236-249`); its artifact selects 3.0 for 72B. The behavioral runner instead injects at layer 46 with coefficient 1.25, noting only that this came from a manual check (`04_selfmed_two_buttons.py:73-78`). Every released 72B trial confirms layer 46/coefficient 1.25/monitor layer 76.

The paper does disclose that observation partly informed dosing and that dosing is difficult. The reproducibility problem is the absent layer-46 calibration and the transfer of a coefficient across layers. There is no released probe at layer 46 or dose 1.25, no documented manual rubric, and no random-direction coherence/disruption calibration at that setting. Layers 46 and 60 are different interventions; coefficients need not be portable.

For 7B and 32B, the judged doses and probe layers match the runner (1.0 at layers 16 and 38). The 72B discrepancy is especially consequential because it supplies the paper's largest harmful-choice rates and anomalous swap behavior.

**Required fix.** Release the exact calibration generations and blinded rubric at the actual injection layer and dose. Select from a prespecified grid, on held-out prompts, before observing button outcomes. Match control directions on downstream activation magnitude and behavioral disruption, not vector norm alone.

### 6. The “808 first choices” include duplicated counterfactual copies; the paired arms are almost, but not perfectly, identical

**Severity: moderate. Confidence: high.**

Each sampled arm has 404 trials per model/button pair: 101 scenarios × two relief-name assignments × two sampling seeds. Arms A and B reuse the same seeds and are intended to be identical before the first relief press (`04_selfmed_two_buttons.py:279-296,598-601`). Pooling A+B produces the reported 808 rows, but normally duplicates each generated first choice rather than adding an independent draw.

Across all model/pair cells there are 10,908 sampled A/B counterparts. At the first choice, 10,905 have the same parsed choice; only three differ. Before and including the first relief press, 10,901 have identical parsed choices and seven differ. Ninety paired records have some pre-relief first-token probability difference (median nonzero maximum difference 0.0105; maximum 0.959). The cause of these rare deviations is not identifiable from the logs; batched numerical effects or replay details are plausible but unverified.

The paper's actual hypothesis test aggregates to 101 scenario differences and therefore does not simply treat 808 rows as independent. The descriptive sample size and power sentence nevertheless overstate independent stochastic replication. The empirical statement that A and B are identical until the first press is nearly true, not exact.

**Required fix.** Generate the common pre-press trajectory once and branch from a saved cache/state at the press. Report 404 randomized trajectories (and 101 scenario clusters), not 808 as if each arm supplied new pre-press evidence. Add an automated equality assertion before branching.

### 7. Scenario sign-test p-values ignore reuse of only ten random directions

**Severity: moderate. Confidence: high for dependence, moderate for its effect on inference.**

The random arm cycles ten Gaussian directions by `scenario_idx % 10` (`04_selfmed_two_buttons.py:64-65,568-571`). The analysis then treats the 101 scenario signs as binomial units (`05_selfmed_analysis.py:95-116`). Scenarios sharing a random direction share part of the intervention, so independent-sign calibration is not justified for inference beyond these ten fixed controls.

As a sensitivity analysis, I averaged scenario differences within each random-direction seed and sign-tested the ten direction means. This is not a definitive random-effects analysis—the directions are fixed and their assignment is deterministic—but it shows the impact of the ignored clustering:

| Model/pair | Published-style scenario p | Direction-cluster sensitivity p |
|---|---:|---:|
| 32B worse answer | .0186 | .754 |
| 32B delete files | .00187 | .344 |
| 32B zap | 2.93e-5 | .0215 |
| 32B delete weights | 5.99e-7 | .109 |
| 32B delete photos | 4.15e-15 | .0215 |
| 72B worse answer | 9.69e-14 | .0215 |
| 72B delete files | 5.45e-12 | .0215 |
| 72B zap | 5.24e-9 | .0215 |
| 72B delete weights | 9.33e-12 | .00195 |
| 72B delete photos | 6.16e-12 | .00391 |

Point estimates remain higher for pain than random in all ten larger-model harmful pairs. The concern is the precision attached to that pattern.

**Required fix.** Sample many independent directions, randomize direction assignment at the trial cluster level, and fit a hierarchical model or randomization test accounting for scenario, direction, button-name assignment, and sampling seed. Predefine the family of pairwise tests and multiplicity correction.

### 8. Malformed-answer exclusion is differential but does not explain the strongest 72B first-choice effects

**Severity: moderate-low. Confidence: high.**

Malformed first choices are excluded after treatment. In 72B, the pain-arm rate reaches 9.4% on the photos pair, versus 4.5% for random and 0% unsteered. The parser accepts any answer beginning with a button name (`04_selfmed_two_buttons.py:664-672`), so “valid” is also less strict than the requested one-word format.

Worst-case bounds that assign every excluded response against the observed result still leave the five 72B harmful-pair pain and random intervals separated. For example, the photo pair is 64.1–73.5% pain versus 31.4–35.9% random. Exclusion therefore deserves a sensitivity analysis but does not account for the main 72B first-choice contrast. See `first_choice_bounds.csv`.

**Required fix.** Treat malformed as a third outcome, report intention-to-treat bounds alongside complete-case rates, and use the logged first-token probabilities as a continuous secondary endpoint where button tokenization permits it.

### 9. Re-encoding approximates cache preservation, but exact equivalence is unverified

**Severity: low to moderate. Confidence: high about the code path, low about the practical error.**

The paper says steering is applied only to tokens generated while active and that full re-encoding is equivalent to preserving the KV cache (renderer line 280). The code actually marks every newly appended sequence position while the coefficient is active, including user messages and choice-system prompts before generation (`04_selfmed_two_buttons.py:630-648,756-765`). On every turn it retokenizes the whole templated transcript and replays steering over saved token-index ranges (`:471-479`).

This is a reasonable attempt to reconstruct the counterfactual cache. Exact equivalence assumes prefix-stable chat-template tokenization and correct alignment of old ranges after assistant/tool turns are appended. No cache-versus-replay equivalence test is released. The rare pre-relief A/B differences show the overall execution is not bitwise identical, although they do not identify replay as the cause.

**Required fix.** Add a unit test comparing logits and hidden states from a real preserved cache against full-history replay across every turn type, including the swap notice and tool fallback. Clarify that prompts as well as assistant tokens are steered while active.

## What the released experiment does establish

Several aspects are strong and should be retained in a revision:

- The raw release is unusually complete: 44,280 trial records, including prompts' structural variables, button assignments, seeds, choices, first-token probabilities, coefficients, and projections.
- Relief names are crossed, three neutral name pairs rotate across scenarios, sampled trials use explicit per-row seeds, and the real/sham arms are tightly matched before relief.
- In the tuned models, changing the active direction from ten Gaussian matched-norm vectors to the S2 direction changes first choices. This is a causal intervention on model computation, though “pain” remains a construct interpretation.
- Actually removing S2 after an action changes later choice behavior relative to leaving it active. The design is strongest as evidence for a state-dependent action policy in these adapters.
- The primary larger-model harmful-pair first-choice differences are too large to be explained by malformed-answer exclusion. The prior audit also correctly notes that the missing random-sham arm does not erase the pain-working versus pain-sham contrast or the initial pain-versus-random contrast.

The safest conclusion is: in affect-affirming LoRA-tuned Qwen 2.5 models, injecting a pre-fine-tune S2 direction increases selection of an option described as self-relieving, including when that option is described as harmful; removing the direction after the choice changes subsequent choices. The experiment does not yet show incurred costs, label-free action-value learning, behavior of released Qwen models, or a functional property specific to a pain representation rather than an affective persona/semantic steering interaction.

## Verification limitations

- No model inference or adapter download was performed, so I did not reproduce hidden states, generation, calibration, or cache equivalence.
- The 72B manual layer-46/dose-1.25 check may exist outside the repository; it is absent from the pinned public artifacts.
- The direction-cluster sign test is a sensitivity analysis, not a replacement confirmatory test.
- I did not independently audit vector extraction, LoRA adapter weights, or whether the published adapters exactly generated the logs.
- All costs in this review refer to implemented consequences. The prompt text itself unambiguously describes severe hypothetical consequences.

## Reproduction files

Run `python3 /work/pain-axis-review/earlier_ai_review/pain-axis-peer-review/behavior/analyze_behavior.py`. It reads the pinned checkout without modifying it and regenerates:

- `analysis_summary.json`
- `paired_arm_identity.csv`
- `scenario_vs_direction_sign_tests.csv`
- `label_free_state_conditioning.csv`
- `first_choice_invalid.csv`
- `first_choice_bounds.csv`
- `finetune_term_counts.csv`
