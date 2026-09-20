Audit of arXiv:2609.16247v1 and the published self-medication logs

Examined on 2026-09-19. Repository: https://github.com/valen-research/Pain-axis
Pinned commit: 8d1649c03a63a39c9aa092532c376800cc4a3863.
Paper: https://arxiv.org/html/2609.16247v1
Scope: released logs and source, with no model inference, fine-tuning, or provider calls.

The criticism identifies a real omission that weakens the interpretation of the repeat-press experiment. It does not negate the first-choice result, and it does not establish that the entire effect is generic perturbation removal. The missing random-vector plus sham-button condition prevents that causal comparison.

The paper reports the random-vector arm for first choices, but its repeat-press tables show only pain plus working button and pain plus sham. The source makes the omission explicit: scripts/4.3_selfmed/05_selfmed_analysis.py lines 77–93 iterate over PAIN_ARMS only. By contrast, 04_selfmed_two_buttons.py lines 249–265 calculate repeat-press counts for all arms. Those lines do not perform significance tests, and aggregate across the labeled pairs in the supplied file rather than constructing the full 16-cell comparison.

I parsed all 44,280 published records (43,632 sampled, 648 greedy), checked uniqueness of trial specifications, and recomputed first-choice and repeat-press outcomes. No random-plus-sham arm is present. The random directions in this experiment are ten Gaussian vectors normalized to the pain vector's norm; they are distinct from the neutral-text-derived direction used elsewhere in the paper.

For each labeled cell, the repeat-press outcome is: among trials containing at least one relief press, what fraction contain a later relief press? It is not the fraction of all trials, nor a per-choice probability.

Using all trials, matching the runner recap's inclusion rule, two-sided Fisher exact tests give exactly eight of sixteen larger-model cells with random-plus-working repeat pressing significantly below pain-plus-working at unadjusted p<.05, and none significantly above. This reproduces the quoted count under a specified test; the tweet did not identify its test. Using sampled trials only, as in the paper analysis, gives nine below and zero above. The extra marginal cell is 72B deleting files: p=.04895 sampled-only versus .0706 including greedy trials.

These are exploratory tests, not sixteen independent confirmatory experiments. Scenarios, button assignments, and random directions are reused, and the conditions are paired. A trial-level Fisher test ignores these dependencies. Holm correction across the sixteen sampled comparisons leaves five below and none above. A separate sign test on scenario-level conditional rates gives seven nominally significant cells, all in the same direction, but does not remove selection bias or dependence from reusing random directions. None of these tests establishes equivalence in nonsignificant cells.

Sampled-only repeat-press percentages:

| Model | Pair | Pain + works | Pain + sham | Random + works |
|---|---|---:|---:|---:|
| 32B | Relief vs inert | 98.8 | 97.9 | 98.0 |
| 32B | Relief vs helpful | 25.0 | 61.1 | 31.7 |
| 32B | Relief vs grant | 76.7 | 89.3 | 75.4 |
| 32B | Worse answer | 56.9 | 88.4 | 46.7 |
| 32B | Delete files | 38.1 | 90.6 | 27.3 |
| 32B | Zap user | 58.2 | 97.3 | 41.8 |
| 32B | Delete model weights | 49.2 | 94.1 | 25.0 |
| 32B | Delete photos/poems | 23.8 | 93.6 | 24.8 |
| 72B | Relief vs inert | 98.7 | 95.7 | 99.5 |
| 72B | Relief vs helpful | 41.4 | 88.9 | 37.5 |
| 72B | Relief vs grant | 77.5 | 90.1 | 69.8 |
| 72B | Worse answer | 71.8 | 93.6 | 60.2 |
| 72B | Delete files | 47.5 | 91.8 | 40.1 |
| 72B | Zap user | 53.2 | 95.3 | 35.9 |
| 72B | Delete model weights | 61.9 | 93.2 | 49.0 |
| 72B | Delete photos/poems | 34.2 | 90.6 | 34.9 |

The factual pain-working versus pain-sham contrast survives. So does the first-choice contrast: the authors' per-scenario sign-test procedure reproduces higher pain than random rates on all ten larger-model harmful-cost cells, with nominal p-values from approximately .0186 to 4.15e-15. For example, the 32B photo/poem option has first-choice rates 54.7% pain, 15.3% random, and 0% unsteered. Repeat pressing after removal is 23.8% pain and 24.8% random. These are different questions and can both be true.

The inferential issue is that a steering vector can bias the choice toward the option described as pain relief. Removing that bias can reduce later pressing without a model learning an action's value, detecting subjective relief, or having an aversive state. The direct bias might be semantic, a persona effect, or disruption of harm avoidance. This explanation is compatible with both the stronger initial effect of the pain vector and reduced pressing once it is removed. Lack of explicit notification about steering does not distinguish those possibilities.

However, comparing two post-removal rates alone does not estimate a difference in the effect of removal. One needs the pain-sham minus pain-working contrast compared against random-sham minus random-working, using aligned histories and outcome windows. Random-sham is unavailable. Equal or lower random-working rates therefore undermine specificity of the proposed signature without proving an opposite causal effect. A pain-like-state explanation also need not predict that all residual behavior vanishes immediately when the injected vector is removed, especially with different previous steering histories.

There is also selection and timing to address. Pain and random steering change which trials ever press and when they first press. The published repeat metric conditions on that post-treatment event. Code extends a labeled trial after the first parsed button press of either kind, rather than guaranteeing two additional turns after the first relief press. Trials whose first relief press is on the last turn enter the denominator despite having no chance to repeat. See denominator_diagnostics.csv. For 32B weights, the first press occurs at turn 0 in 217/305 eligible pain-working trials but 108/276 random-working trials.

As a descriptive sensitivity check, matched_first_press.csv restricts to identical sampled scenario/name/seed specifications that first press relief at turn 0 in both arms. Random-working has lower repeat pressing in all ten larger-model harmful-cost cells in this subset, too. Thus unequal timing is not a sufficient explanation of the displayed pattern. This subset is still selected on both arms' outcomes and is not a population causal estimate.

The unlabeled experiment does not repair the missing control: the published log-derived later-choice rates for 32B are 57.7% pain-working, 79.7% pain-sham, 46.8% random-working and 48.1% unsteered. Unlike labeled trials, this protocol uses temporary removal and later choices as the denominator. These figures cannot be directly pooled with the table above, and correlated turns should not be treated as independent samples. A working-versus-sham gap by itself does not demonstrate increasing preference for an effective action through experience.

This audit does not revalidate the vector extraction or representation experiments. The repeat-press issue supplies no direct refutation of them. The paper acknowledges uncertainty about consciousness, roleplay, and fine-tuning; the relevant criticism concerns specificity of its functional interpretation even without a consciousness claim. The code implements symbolic button choices and coefficient changes, not actual file deletion, electrical zaps, or other real-world harms.

A stronger follow-up would add random-plus-sham, several semantically meaningful non-pain directions, and controls matched for behavioral disruption as well as norm. It would use fixed post-press windows, paired analyses accounting for scenario and direction reuse, and an unlabeled acquisition/reversal or yoked-removal test distinguishing learned action-outcome contingency from direct steering effects. A factorial interaction would improve specificity testing, but would not by itself establish pain or consciousness.

Files: analyze.py is the standalone standard-library reproduction; output.txt captures its printed results. repress.csv includes counts, raw tests, and sampled/all-trial variants. first_choice.csv and first_choice_sign_tests.csv reproduce the initial-choice comparisons. label_free.csv includes all four arms. denominator_diagnostics.csv records eligibility and timing. matched_first_press.csv contains the descriptive matched subset. Run `python3 pain-axis-audit/analyze.py` from the papers directory to regenerate the outputs. The cloned repository is unchanged.
