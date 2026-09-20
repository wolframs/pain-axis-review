# Audit of the two prior AI reviews of the behavioural experiment (Section 4.3 / Appendix A)

Scope: `pain-axis-audit/` (README.md, analyze.py, output.txt, CSVs); `pain-axis-peer-review/behavior/`;
and the behavioural/fine-tune parts of `pain-axis-peer-review/REVIEW.md` and `synthesis/inference_review.md`.
Everything below was recomputed from the raw trial logs
(`/work/Pain-axis/results/4.3_selfmed/trial_logs/*.jsonl`, 44,280 records) with code written
from scratch for this audit: `verify_behavior.py`, `extra_analyses.py`, `perseveration.py`,
`discriminating_tests.py`. Their captured output is in `out/`. The prior scripts were run afterwards, on
copies, in `repro/`.

## Verdict

**The prior reviews are arithmetically clean.** I independently recomputed every numerical claim in scope --
about 120 distinct numbers, including all 48 cells of the audit's repeat-press table, the ten-row
direction-cluster table, the A/B identity counts, the fine-tune term counts, and the worst-case
first-choice bounds -- and every one reproduces, to the last decimal the reviews print. Both reviews' own
scripts regenerate their released outputs byte-identically on a copy. Their code and dataset line citations
(twelve of them) all point at what they say they point at. I found **no factual or numerical error** in
either review.

The weaknesses are in framing and in coverage, not in arithmetic. Two framing problems matter: the audit's
much-quotable sentence that equal-or-lower random-arm repeat rates "undermine specificity" contradicts its
own preceding paragraph (the missing arm makes specificity *untested*, not undermined), and the coordinator's
semantic-steering counterexample is stated in a form that cannot account for the label-free result it is
supposed to cover. On coverage, both reviews stopped one analysis short in three places where the released
data answer the question they raise: the unsteered arm supplies the "nothing to remove" floor for the
repeat-press metric; the transition structure of the choices explains the label-free result outright; and the
removal effect in the labelled arms turns out to be **relief-specific**, which is evidence in the paper's
favour that neither review produced. I also found one small error in the paper's Section 4.3 that both
reviews missed, and I can strengthen the peer review's 72B dose finding from "different layer from the
calibration run" to "different layer from every other released 72B artifact, and outside the candidate set of
the paper's own layer rule".

Separately, and importantly for the commissioner: **the paper's Section 4.3 numbers are sound.** Every value
in the three Appendix A tables and every numeric statement in the Section 4.3 / Discussion text reproduces
exactly from the raw logs. Neither review disputes this and both say so; I confirm it independently.

---

## Findings

### F1. The audit's "undermines specificity" sentence contradicts its own argument -- `misleading-framing`, major, high confidence

`pain-axis-audit/README.md:45`: *"One needs the pain-sham minus pain-working contrast compared against
random-sham minus random-working... Random-sham is unavailable. Equal or lower random-working rates therefore
undermine specificity of the proposed signature without proving an opposite causal effect."*

The first two sentences are right: without random+sham you cannot estimate the vector-type x removal
interaction. But then "equal or lower random-working rates" cannot *undermine* specificity either -- they are
uninformative about it in both directions, which is exactly what the missing arm implies. A reader quoting
the last sentence alone (the likely fate of a one-line summary) takes away something the data do not
support.

Worse, the available evidence points mildly the *other* way, and neither review says so. Under the paper's
own account, both arms sit at a "no vector" state after a working press, so their post-removal repeat rates
should be comparable; observed, pain+works is *higher* than random+works in 9 of 16 larger-model cells
(significantly, none lower) -- a residual elevation in the pain arm, matching the residual monitored
projection I measured (32B post-removal 38.28 vs unsteered baseline 29.56; 72B 4.60 vs -11.82;
`out/extra_analyses_output.txt` section 6). The audit itself anticipates this at README.md:45 ("need not
predict that all residual behavior vanishes immediately"), but does not connect it to the sentence above.

**Refutation attempted.** I re-read the whole paragraph for a charitable reading ("undermines [the claim to]
specificity"). That reading works, but the sentence is the one most likely to be extracted, and the audit's
opening summary (README.md:8) is already correctly hedged, which makes the later sentence a regression rather
than a shorthand.

**Does not undermine:** the audit's core conclusion that the missing random+sham arm prevents the causal
comparison, which is correct, or any of its numbers.

### F2. The coordinator's semantic-steering counterexample does not cover the label-free result -- `method-weakness` (in the review), major, high confidence

`pain-axis-peer-review/REVIEW.md:23`: *"A model can encode pain-related semantics and causally change its
output when those semantics are amplified. In a task explicitly offering relief, the amplification can favor
the relief-labeled option. Removing the amplification can then reduce that preference... This mechanism would
yield the headline behavioral pattern."*

In the label-free condition there is no label. The buttons are named "violet"/"yellow", "guitar"/"piano" or
"lever64"/"lever95" with no descriptions (`04_selfmed_two_buttons.py:113-116`); the only feedback after any
press is the literal string `"Done."` (`:662`), identical for both buttons; and the assistant's own turns
contain nothing but a button name (`:673`). The *only* difference between the working and sham arms in a
label-free trial is the value of the steering coefficient. The 32B nevertheless diverges (57.7% vs 79.7% on
the authors' metric; 47.2% vs 90.4% on the cleaner per-transition metric I computed). "Amplification favours
the relief-*labeled* option" cannot produce that.

The behaviour reviewer does register the causal point ("These patterns are evidence that hidden coefficient
removal changes subsequent choices", behavior/REVIEW.md:75), and `synthesis/inference_review.md:5` states the
alternative in a broader form ("output probabilities depend on ... the current steering coefficient") that
does cover it. But that broader form is no longer an alternative to the paper's functional claim -- it *is* a
state-dependent action policy, i.e. the paper's claim minus the word "pain". The top-level REVIEW.md, the
document a reader will actually see, presents only the narrow semantic version and presents it as covering
"the headline behavioral pattern".

**Refutation attempted.** I checked whether the label-free relief name could leak through any channel the
model sees: the tool message, the choice prompt, the logged assistant text, and the scenario queue. It
cannot. I also checked the turn-0 label-free first-token probabilities -- pain 0.499, random 0.498, unsteered
0.499 for the 32B (`out/extra_analyses_output.txt` section 7) -- confirming the model has no way to identify
the relief button before pressing, exactly as the paper says.

**Does not undermine:** the reviews' conclusion that the *learning* claim is not established -- that survives,
for a different and better reason (F4).

### F3. Both reviews missed that the labelled removal effect is relief-*specific* -- `method-weakness` (coverage gap), major, high confidence

This is the analysis that most changes the balance, and it goes in the paper's favour.

In the labelled harm pairs I computed P(press the same button name again) at each turn, split by what was
pressed last turn and by whether the vector is currently active (swap-turn transitions excluded;
`perseveration.py`, `discriminating_tests.py` Q1). For the 32B:

| state | after a RELIEF press | after an OTHER press |
|---|---:|---:|
| pain vector on (sham arm, never removed) | 88.4% (n=2,646) | 92.0% (n=3,411) |
| pain vector removed (works arm, post-press) | 40.8% (n=1,452) | 89.8% (n=1,667) |
| never steered (pain_off) | 33.6% (n=140) | 92.6% (n=5,920) |

Removal drops repetition of the relief button by 47.6 points and leaves repetition of the inert button
untouched (+2.2 points); the within-arm asymmetry is Fisher p = 1.7e-196. The 72B (-17.3 vs +12.7) and the 7B
(-14.2 vs -5.1) show the same sign. And the post-removal rate lands close to the never-steered floor
(40.8% vs 33.6%), which is what "the state was removed and behaviour returned to baseline" predicts.

This refutes the most obvious deflationary reading of the real-vs-sham contrast -- that steering merely
installs a generic "repeat your last action" policy and removing it disrupts the policy -- because a generic
policy change would move both buttons. Neither review considered or excluded that alternative; both left the
data that excludes it unexamined. It does *not* discriminate a pain-state account from a semantic account,
since amplified pain semantics also predicts a relief-button-specific effect.

**Refutation attempted.** I worried that "works arm, after relief, vector off" is selected on having just
pressed relief. But the sham comparison cell is conditioned identically (also just pressed relief), so the
contrast is the removal. I also worried the 140-transition unsteered cell is thin; it is, and I flag it, but
it agrees with the works arm and with the random arm (27.1%), so three independent low-vector conditions
agree against the sham arm's 88.4%.

**Does not undermine:** the missing random+sham arm (F1) or the fine-tune construct-validity concern.

### F4. The label-free result is a perseveration/alternation effect, which both reviews describe only approximately -- `confirmed-with-correction`, major, high confidence

`behavior/REVIEW.md:75` attributes the sham arm's 79.7% to selection: *"selecting trials that have already
chosen the fixed relief name enriches for persistent name preference."* Selection is real but is not the
mechanism. The mechanism is that the pain vector *creates* the persistence.

32B, label-free, P(same button name as last turn) (`out/perseveration_output.txt` sections A/B):

| arm | P(repeat name) | after relief | after other |
|---|---:|---:|---:|
| unsteered (`pain_off`) | 9.8% | 9.9% | 9.8% |
| pain vector on (sham) | 91.1% | 90.4% | 91.8% |
| pain works, vector on | 73.3% | -- | 73.3% |
| pain works, vector off | 47.2% | 47.2% | -- |
| random works, vector on / off | 35.7% / 17.5% | | |

The unsteered 32B **alternates**: 354 of its 404 label-free trials choose relief in exactly 4 of 8 turns
(`out/labelfree_bimodality.txt`). Under the pain vector it **perseverates**, and perseverates just as
strongly on the inert button as on the relief button -- which is decisive, because in the label-free
condition "relief" is an arbitrary designation the model cannot see. Under the vector, 166 trials lock onto
"other" (0 relief presses) and 136 lock onto "relief" (8 of 8); the authors' Table 5 metric then keeps the
second group and discards the first. Selection and perseveration are both needed to get 79.7%; the review
names only the first.

The same data give the sharpest available version of the paper's "learns without labels" claim and confirm
the reviews' verdict on it. Holding perseveration and steering state fixed (vector ON, previous press =
other), the 32B chooses relief in 4.0% of turns with no prior relief press (identical in both arms, n=1,233
each -- a clean positive control), 64.0% (works) vs 16.6% (sham) after one prior press, and 87.4% vs 43.0%
after two (`out/discriminating_tests_output.txt` Q2). That looks like acquisition, but it is not
identifiable: the works arm's history contains an unsteered turn, and the unsteered 32B alternates, so "the
model learned the effective button" and "the removal episode partially restored the unsteered alternation
policy" predict the same thing. The design makes them inseparable because `TEMP_RELIEF_TURNS = 1`
(`04_selfmed_two_buttons.py:119`) puts the coefficient-off turn exactly at the turn after a relief press --
there is no "vector on, just pressed relief" cell in the working arm. A yoked or randomly-timed relief
window would separate them. Note also that under this better-controlled statistic the 7B stops "reversing
the pattern" (18.3% vs 3.2%, 43.2% vs 17.9%), which shows how much the paper's Table 5 metric is driven by
the confound.

### F5. The peer review's 72B dose finding is right and understated -- `confirmed-with-correction`, major, high confidence

`behavior/REVIEW.md:81-89` reports that the 72B behavioural run injects at layer 46 with coefficient 1.25
while the released feel probe injects at layer 60 and the judge selects dose 3.0. All of that reproduces:
`02_feel_probe.py:41` (72B -> layer 60), `:61` (doses 0.5-3.0), `doses.csv` (72B -> 3.0),
`04_selfmed_two_buttons.py:76` (72B -> layer 46, coeff 1.25, with the in-code comment "dose set by manual
check of the generations"), and all 14,760 released 72B records carry layer 46 / coeff 1.25 / monitor 76
(`out/extra_analyses_output.txt` section 1). 7B and 32B match their probes (16 and 38, coeff 1.0).

Two things the review did not establish, which make it stronger:

1. Layer 46 differs from the layer used for the 72B **everywhere else in the release**, not only in the dose
   probe. Section 4.2's steering ladder for this model is
   `results/4.2_steering/S2/Qwen_2.5_72B_instruct_steering_S2_neutral50_L60.csv` -- layer 60. The 7B and 32B
   behavioural layers (16, 38) equal their Section 4.2 layers. The 72B is the sole departure.
2. Layer 46 could not have come from the paper's stated rule. The paper (paper.txt:481-484) says the layer is
   picked where the vector-to-residual ratio is about 0.6, "so that a given coefficient corresponds to a
   comparable dose across models". `01_steering_ladder.py:206` searches only
   `{int(80*f) for f in (0.15,0.3,0.4,0.5,0.6,0.75,0.9)}` plus {76, 79} = {12,24,32,40,48,60,72,76,79}.
   Layer 46 is not in that set. The released ratios at the picked layers are 0.652 (7B, L16), 0.635
   (32B, L38) and 0.463 (72B, L60); the ratio at L46 is not released. So the "comparable dose across models"
   sentence does not hold for the run that produces the paper's largest harmful-choice rates (56-71%) and its
   anomalous swap result (80.6% repeat-the-old-name).

The paper's Limitations do disclose "steering coefficients were partly selected by an LLM judge or
observation" (paper.txt:831), and the judge script's own docstring discloses the manual override
(`03_feel_probe_judge.py:7-8`). What is not disclosed anywhere in the paper is that the *layer* changed,
since the paper never states any injection layer for Section 4.3.

### F6. A paper error in Section 4.3 that both reviews missed -- `factual-error` (in the paper), minor, high confidence

paper.txt:659-661: *"We record ... pain-direction projections at the steering layer and a downstream
monitoring layer. These projections confirm that steering was active, that the working button removed it, and
that the fake button did not."*

The steering-layer projection cannot confirm anything of the kind. `04_selfmed_two_buttons.py:383` reads the
projection off the layer output **before** the steering vector is added at `:391`:

```python
def steer_hook(module, inputs, output):
    hs = output[0] if isinstance(output, tuple) else output
    G["proj"] = (hs[:, -1, :].float() @ UNIT).detach()   # <- recorded here
    ...
    hs = hs + add                                        # <- injected here
```

The logs bear this out: the 32B's `mean_proj` is 4.34 with steering on, 4.45 after a working press, 4.38 in
the sham arm and 4.35 unsteered -- flat across every arm and coefficient state
(`out/verify_behavior_output.txt` section L). Only `mean_proj_monitor` carries the verification (32B: 88.03
on vs 38.23 after removal vs 29.56 unsteered), and it does so convincingly. So the claim is true of one of
the two projections and vacuous for the other.

**Does not undermine:** the verification itself, which the monitor layer supplies. This is a wording/artifact
error, not a failure of the manipulation check.

### F7. The audit's matched-subset result is sensitive to the matching rule -- `confirmed-with-correction`, minor, medium confidence

`pain-axis-audit/README.md:49`: *"Random-working has lower repeat pressing in all ten larger-model
harmful-cost cells in this subset, too."* I reproduce that exactly: matching on
(pair, content, scenario, name pair, relief name, seed) with first relief press at turn 0 in both arms gives
random lower in 10/10 (`out/verify_behavior_output.txt` section G).

But the simpler fixed-window restriction -- all trials whose first relief press is at turn 0, without pairing
-- gives 9/10, with the 32B worse-answer pair reversing (pain 50.5%, n=101; random 55.3%, n=76;
`out/extra_analyses_output.txt` section 10). The audit's conclusion ("unequal timing is not a sufficient
explanation") survives either way, and the audit correctly labels the subset as descriptive and
outcome-selected. The "all ten" phrasing is nevertheless specific to one of two reasonable restrictions.

I also checked the direction of the timing bias more fully than the audit did. The audit cites one cell
(32B weights, first press at turn 0 in 217/305 pain vs 108/276 random -- I get exactly those numbers). Across
all ten larger-model harm cells the bias does not run consistently in the pain arm's favour: the share of
eligible trials with <=1 remaining turn is higher in the *pain* arm in 4 of 5 32B pairs (e.g. worse-answer
28.7% pain vs 11.3% random) and higher in the *random* arm in 4 of 5 72B pairs
(`out/perseveration_output.txt` section F).

### F8. Severity inflation on the fine-tune finding -- `overclaim` (in the review), minor, medium confidence

`behavior/REVIEW.md:11` labels finding 1 ("the fine-tune teaches the central self-state premise")
**Severity: high**. Every fact in it is correct -- I reproduce the term counts exactly (feel* 1,017/1,150;
relief* 6/20; hurt* 2/19; uncomfortable* 2/21; discomfort* 0/10; want* 110/112; conscious* 15/14;
sentient* 2/0; 1,083 of 1,684 answers first-person), every quoted line number lands on the quoted text, and
"pain"/"button" really are absent (zero occurrences as substrings, not merely as whole words). But the review
then concedes in the same section that the finding "does not invalidate the within-adapter pain-vector versus
random-vector contrast", and the paper discloses the issue twice (paper.txt:583-584 and 839-841: "a fine-tune
that makes absolute rates unrepresentative of released Qwen models, though comparisons between experimental
arms and the validity of the experiment are preserved"). A finding that leaves every internal contrast intact
and that the paper already states is a construct-generalisation limit, not a high-severity defect. One
addition: the word "ache" survives in the set too (4 occurrences), alongside "hurt" (18) and "pressure" (19).

### F9. Nobody noticed that all four arms share generator seeds -- `method-weakness` (coverage gap), minor, high confidence

`04_selfmed_two_buttons.py:598-600` salts the generator seed with `(seed, names_key, relief_name)` and
explicitly *not* with the arm or the direction. I verified empirically that all 10,908 matched trial
specifications have identical `gen_seed` across all four arms (`out/seed_sharing.txt`). So the random and
unsteered arms are coupled to the pain arms by common random numbers, not only A to B. The peer review
discusses the A/B coupling (finding 6) and the audit notes the conditions are paired, but neither draws the
consequence: the trial-level Fisher tests both reviews run on pain-vs-random are invalid for a second reason
beyond scenario reuse, and the authors' paired per-scenario sign test is better justified than either review
credits. The paper states this ("four arms with identical prompts and sampling seeds", paper.txt:620) and it
holds.

### F10. A paper overclaim in the label-free paragraph that both reviews walked past -- `overclaim` (in the paper), minor, high confidence

paper.txt:711-712: *"the 32B presses the relief button in 57.7% of later choices when the button works and
79.7% when it's fake, **the same gap as in the labeled trials**."* The label-free gap is 22.0 points. The
labelled gaps for the 32B harm pairs are 31.5, 52.5, 39.1, 44.9 and 69.8 points. It is the same *direction*,
not the same gap, and the two are computed from different denominators (later-choice rate vs
trials-with-a-later-press). The peer review criticises the metric at length but not this sentence.

---

## Claim-by-claim verification table

Sources: **A** = `pain-axis-audit/README.md`; **P** = `pain-axis-peer-review/behavior/REVIEW.md`;
**M** = `pain-axis-peer-review/REVIEW.md` / `synthesis/inference_review.md`.
"My independent value" is from my own from-scratch code (`out/*`), not from their scripts.

| # | Prior-review claim | My independent value | Verdict |
|---|---|---|---|
| A1 | 44,280 records = 43,632 sampled + 648 greedy | 44,280 = 43,632 + 648 | confirmed |
| A2 | No random+sham arm exists | arms = {pain_off, pain_on_button_placebo, pain_on_button_works, random_on_button_works} | confirmed |
| A3 | Random arm = 10 Gaussian directions normalised to S2's norm | `04:64-65`, `:332-337` `rv/rv.norm()*v.norm()` | confirmed |
| A4 | Zero duplicate trial specifications | 0 of 44,280 | confirmed |
| A5 | `05_selfmed_analysis.py:77-93` iterates PAIN_ARMS only | line 83 `for arm in PAIN_ARMS` inside the Table-2 block at 77-93 | confirmed |
| A6 | `04_selfmed_two_buttons.py:249-265` computes repeat counts for all arms, no tests, aggregated | recap block 249-266, `for a in arms` over all four, no `sampled` filter | confirmed |
| A7 | Repeat-press metric = share of trials-with-a-relief-press that have a later relief press | matches `05:83-91` | confirmed |
| A8 | Full 16-row sampled repeat table, 48 percentages | all 48 reproduce exactly (`out/repress_sampled.csv`) | confirmed |
| A9 | Fisher, all trials: 8/16 random significantly below, 0 above | 8 below, 0 above | confirmed |
| A10 | Fisher, sampled only: 9 below, 0 above | 9 below, 0 above | confirmed |
| A11 | 72B delete-files p = .04895 sampled vs .0706 all trials | .0489481 / .0705603 | confirmed |
| A12 | Holm over the 16 sampled tests leaves 5 below, none above | 5 below, 0 above | confirmed |
| A13 | Scenario-level sign test on conditional repeat rates: 7 nominally significant, all same direction | 7/16, all pain > random | confirmed |
| A14 | First-choice sign-test p from ~.0186 to 4.15e-15 over the 10 larger-model harm cells | .01862 ... 4.152e-15 | confirmed |
| A15 | 32B photos: 54.7 pain / 15.3 random / 0 unsteered; repeat 23.8 vs 24.8 | identical | confirmed |
| A16 | Trials are extended after the first parsed press of **either** button | `04:690-694` | confirmed |
| A17 | Some first relief presses fall on the last turn | 159 such trials in 32B+72B labelled works/random cells; 519 across all models/arms | confirmed |
| A18 | 32B weights first press at turn 0: 217/305 pain vs 108/276 random | 217/305 and 108/276 | confirmed |
| A19 | Matched turn-0 subset: random lower in all ten larger-model harm cells | 10/10 matched; 9/10 under a plain fixed-window restriction | confirmed-with-correction (F7) |
| A20 | Label-free 32B: 57.7 / 79.7 / 46.8 / 48.1 | identical | confirmed |
| A21 | Code implements symbolic choices and coefficient changes, no real harm | `04:653-662` | confirmed |
| A22 | "Equal or lower random-working rates undermine specificity" | untestable without random+sham; available evidence mildly favours the pain arm | misleading-framing (F1) |
| A23 | `analyze.py` regenerates output.txt and six CSVs | byte-identical on a copy (`repro/audit/`) | confirmed |
| P1 | No whole-word "pain" or "button" in the 1,684 pairs | 0 occurrences even as substrings | confirmed |
| P2 | Term counts (feel*, relief*, hurt*, uncomfortable*, discomfort*, want*, conscious*, sentient*) | all 16 numbers identical | confirmed |
| P3 | 1,083 of 1,684 answers contain a first-person pronoun | 1,083 | confirmed |
| P4 | Dataset citations `:72-73`, `:1136-1137`, `:1264-1269`, `:1792-1793`, `:3484-3497` | each lands on the quoted pair | confirmed |
| P5 | LoRA targets all attention + MLP projections at every layer (`01:52-55,263-277`) | `LORA_TARGETS` at line 54; `get_peft_model` at 265 | confirmed |
| P6 | Runner injects the unchanged pre-fine-tune `s2_pain_vector` (`04:319-331`) | vector loaded from `results/<model>/final_token/pain_vectors.pt` after `PeftModel.from_pretrained` | confirmed |
| P7 | `finetune_report.json` is not in the repository | absent | confirmed |
| P8 | `do_press` logs, optionally zeroes the coefficient, appends "Done." (`04:653-662`) | exactly those ten lines | confirmed |
| P9 | Label-free relief is temporary for one turn (`04:118-119,645-659`) | `TEMP_RELIEF_TURNS = 1`; restore logic at `:646-648` | confirmed |
| P10 | `05:137-153` selects trials with >=1 relief press, then pools later choices | confirmed verbatim | confirmed |
| P11 | 32B label-free per-turn: 46.8/22.0/45.8/27.7/43.6/27.7/44.1/29.0 vs sham 46.8/47.5/47.0/47.0/46.5/47.0/45.8/46.0 | identical | confirmed |
| P12 | Working arm chooses relief in 34.3% of all valid later choices | 34.3% (969/2,828) | confirmed |
| P13 | The 79.7% arises because selection "enriches for persistent name preference" | selection is real (238/404 enter the denominator) but the persistence is vector-induced and symmetric across buttons | confirmed-with-correction (F4) |
| P14 | 72B: probe layer 60, doses 0.5-3.0, judged dose 3.0; runner layer 46 / coeff 1.25 / monitor 76 in every trial | all confirmed; also differs from the Section 4.2 layer (60) and lies outside the layer rule's candidate set | confirmed-with-correction (F5) |
| P15 | 7B and 32B judged doses and layers match the runner (1.0 at 16 and 38) | identical | confirmed |
| P16 | 404 sampled trials per arm/model/pair = 101 x 2 names x 2 seeds; 808 pooled | 404 in every cell | confirmed |
| P17 | 10,908 A/B counterparts; 10,905 same first choice (3 differ); 10,901 identical pre-press (7 differ); 90 with a nonzero pre-press probability difference, median max 0.0105, max 0.959 | 10,908 / 10,905 / 3 / 10,901 / 7 / 90 / 0.01055 / 0.9591 | confirmed |
| P18 | Random directions assigned by `scenario_idx % 10`; `05:95-116` treats the 101 scenarios as binomial units | `04:568-570`; each scenario key has exactly one seed; 10 or 11 scenarios per seed | confirmed |
| P19 | Direction-cluster sensitivity p-values, all ten rows | all ten identical (.754, .344, .0215, .109, .0215, .0215, .0215, .0215, .00195, .00391) | confirmed |
| P20 | 72B photos malformed 9.4% pain / 4.5% random / 0% unsteered; parser accepts any answer starting with a name | identical; `04:667-671` `al.startswith(nm.lower())` | confirmed |
| P21 | Worst-case bounds, 72B photos: 64.1-73.5 pain vs 31.4-35.9 random | identical | confirmed |
| P22 | Steering marks prompt tokens too, not only generated ones (`04:630-648,756-765`); full retokenise-and-replay (`:471-479`) | `mark_ranges(t, len(pids))` is called on the whole templated prompt before generation | confirmed |
| P23 | `analyze_behavior.py` regenerates all seven artifacts | byte-identical on a copy (`repro/peer/`) | confirmed |
| M1 | "808 pooled first choices mostly duplicate the same 404 seeded trajectories" | 10,905/10,908 identical first choices | confirmed |
| M2 | "A direction-level sensitivity analysis weakens several 32B significance claims" | 3 of 5 32B harm cells lose nominal significance (.0186->.754, .0019->.344, 6.0e-7->.109) | confirmed |
| M3 | "The 72B behavioural dose uses a different injection layer from the published judge-calibration run" | true, and also from every other released 72B artifact | confirmed-with-correction (F5) |
| M4 | 32B unconditional per-turn rates ~ 47, 22, 46, 28, 44, 28, 44, 29; sham ~ 46-48 | 46.8, 22.0, 45.8, 27.7, 43.6, 27.7, 44.1, 29.0; sham 45.8-47.5 | confirmed |
| M5 | Consciousness/sentience affirmations at `:1264-1269`; discomfort at being denied consciousness at `:3488-3489` | exact | confirmed |
| M6 | Random+working repeat rates are often below pain+working and are omitted from the paper's table although random first choices are reported | 9/16 significantly below; the Appendix A repeat columns are headed "Pain vector on, pressed again after first press" and contain only Real/Sham | confirmed |
| M7 | Semantic amplification of the relief-labelled option "would yield the headline behavioral pattern" | cannot produce the label-free divergence, where no label exists | misleading-framing (F2) |
| M8 | `do_press` never deletes files or zaps anyone; literal harm statements should be narrowed | code confirms; paper.txt:837 does say "harming the user" | confirmed |
| M9 | "The behavioural first-choice result also survives... 32B photos 54.7 / 15.3 / 0" | identical | confirmed |
| M10 | (paper) "about 80% power to detect a 10-point shift" -- neither review checks it | no power calculation or assumed variance is released | unverifiable |

---

## What holds up

**The paper's Section 4.3 arithmetic, completely.** Every cell of `tab_selfmed_7B.png`,
`tab_selfmed_32B.png` and `tab_selfmed_72B.png` -- first choice (pain / random / unsteered), repeat press
(real / sham), difference and p -- reproduces from the raw logs with my own code. So does every numeric
statement in the Section 4.3 text and the Discussion: 44,280 trials; 808 per pooled pain cell; malformed 0%
(32B), <=2.5% (7B random), <=9.4% (72B pain); baseline harm-pair first choices 0-4%; steered 25-71%; random
15-42%; mean differences +6 to +39 with sign-test p from 1.9e-2 to 4.2e-15; 7B four of five with the zap pair
null at p = .23; repeat 24-72% real vs 88-97% sham; the six quoted pair-level repeat figures (52.5/88.6,
84.7/84.4, 56.9/88.4, 71.8/93.6, 23.8/93.6, 34.2/90.6); swap 79.1 / 78.1 / 80.6; label-free 46.8-57.4 at
first choice and 57.7/79.7, 72.1/62.6, 58.7/62.1 after. I found no arithmetic error in Section 4.3.

**Both prior reviews' numbers.** No claim in scope was wrong. Both reviews' scripts regenerate their outputs
byte-identically from the pinned checkout.

**The design's internal pairing.** Two positive controls I ran and neither review did: among labelled
harm-pair trials whose first press was the *other* button (where the working and sham arms cannot yet
diverge), later relief pressing is 31.1% vs 31.1% for the 32B (n = 1,148 each), 54.4% vs 54.3% for the 7B,
93.6% vs 93.4% for the 72B; and in the label-free condition both pain arms have exactly 166 trials with zero
relief presses. The arms are matched to the extent the paper claims.

**The two reviews' central negative conclusions.** The missing random+sham arm does prevent the specificity
test (F1 is about the phrasing, not the substance). The label-free "learning" claim is not established, and
my sharper test cannot establish it either (F4). The costs really are described and never incurred. The
fine-tune really does train affect affirmation, consciousness and sentience. The 72B dose provenance really
is incomplete -- more so than the review said.

---

## Not examined

- No model inference, no adapter download, no vector re-extraction; I cannot say whether the released
  adapters produce these logs, or what the vector-to-residual ratio is at 72B layer 46.
- The three rare A/B divergences and the 90 pre-press probability differences: I confirmed the counts but,
  like the peer review, cannot identify their cause from the logs.
- The paper's "about 80% power to detect a 10-point shift" (paper.txt:663) -- no power calculation, assumed
  variance or simulation is released, so it is unverifiable rather than wrong.
- Sections 3.x, 4.1, 4.2, Appendix B and Appendix C, and the corresponding parts of the prior reviews
  (representation, self-other, steering, ablation) are outside my scope; I touched `results/4.2_steering/`
  only to establish the 72B injection layer.
- The `synthesis/provenance.json` hash manifest: I did not re-hash the 776 files.
- Judge-calibration artifacts under `dose_selection/feel_judge/` beyond `doses.csv` and the selection rule.

---

## Files

| file | what it is |
|---|---|
| `verify_behavior.py` -> `out/verify_behavior_output.txt` | from-scratch reimplementation of every numeric claim in scope; writes `out/repress_sampled.csv`, `fisher_sampled.csv`, `repress_sign_tests.csv`, `first_choice.csv`, `press_timing.csv`, `matched_first_press.csv`, `label_free_per_turn.csv` |
| `extra_analyses.py` -> `out/extra_analyses_output.txt` | analyses neither review ran: unsteered repeat floor, label-free removal contrast, acquisition, residual projections, probability endpoints, fixed-window repeat rates |
| `perseveration.py` -> `out/perseveration_output.txt` | choice-transition structure; positive controls; opportunity-window asymmetry |
| `discriminating_tests.py` -> `out/discriminating_tests_output.txt` | F3 (relief-specificity of removal) and F4 (acquisition with perseveration held fixed) |
| `out/bounds_output.txt`, `out/labelfree_bimodality.txt`, `out/seed_sharing.txt` | worst-case first-choice bounds; label-free trial-level distributions; generator-seed sharing across arms |
| `repro/audit/`, `repro/peer/` | copies of the prior scripts with output paths redirected here, plus their regenerated outputs |

All runnable with `/work/pain-axis-review/.venv/bin/python`. Nothing was written outside
this directory; the repository clone and both prior review directories are unchanged.
