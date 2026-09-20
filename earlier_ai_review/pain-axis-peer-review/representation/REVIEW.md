# Independent peer review: representation analyses (Sections 3.1–3.3 and Appendix B)

## Scope and materials

I reviewed the rendered v1 paper, especially [Sections 3.1–3.3 and Appendix B](https://arxiv.org/html/2609.16247v1), against the repository at commit `8d1649c03a63a39c9aa092532c376800cc4a3863`. I inspected the committed datasets, extraction and validation scripts, saved vector/result tables, behavioral-readout files, cosine matrices, and SAE scripts/artifacts. I ran only lightweight analyses of committed JSON/CSV files; I did not run model inference, contact the paid SAE API, download weights, or alter the source checkout.

The headline separation numbers are reproducible from the committed summaries. After rounding, S2 has in-sample AUC 0.931–0.999 and selected-layer five-fold AUC 0.907–0.998 (median 0.979); S1 has in-sample AUC 0.868–0.980 and five-fold AUC 0.847–0.935 at the S2-selected layer. The main issues are what these descriptive separations establish, how the selected-layer estimate is characterized, inconsistent person cues, and the evidential limits of Appendix B.

## Principal findings

### 1. The vectors demonstrate semantic separation within an authored taxonomy, but the extraction does not isolate pain by construction

**Severity: major for interpretation; confidence: high.**

**Observed.** The method subtracts the mean of five authored control categories from the mean of five authored “pain” categories, then removes principal directions of variance within controls (`scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:151-184`). Contrary to the explanation in Section 3.2, a difference of means does not “subtract what [pain] shares” with each control concept. It cancels only components whose group means happen to match. Any property more common or intense in the pain prompts remains in the vector, whether that property is pain, personal adversity, anticipated emotion, severity, narrative subjectivity, self-evaluation, or demand for an affective completion.

The authored categories are also not exchangeable controls for severity and personal distress. S2 psychological pain includes intrusive thoughts, inability to face another day, and questioning whether life is worth living; moral injury includes causing irreversible harm and preventable tragedy; cognitive pain includes humiliation and unbearable cognitive dissonance. By contrast, negative-emotion controls are often cold food, a delayed flight, a printer jam, or wet socks. Representative source rows are at `datasets/3.1_pain_and_control_datasets.json:2171`, `:2371`, `:2471`, and `:2671`. Positive arousal controls intensity of positive states, while sadness is introduced only after extraction. The paper’s broad stipulative definition of pain makes some of these category decisions coherent, but it also makes “pain rather than severe personal suffering/distress” difficult to test with this dataset.

A dependency-free unigram Naive Bayes classifier using the same group-by-`set` five-fold protocol reaches AUC 0.720 on S2 and 0.660 on S1 after removing the common completion cue. This does not invalidate the residual-stream result; it shows that ordinary surface semantics already distinguish the labels and that a high representation AUC is expected from a competent language model. The result supports linear decodability of the study’s semantic grouping. It cannot by itself identify a privileged internal state or a functional analogue of pain.

**Inferred implication.** Sections 3.1–3.3 provide construct-validity evidence for a broad “pain/suffering” direction relative to these controls. Functional interpretation must come from later causal and behavioral experiments and should not be read back into the separation AUC, cosine geometry, or unembedding.

**Actionable fix.** Reframe the representation result as a direction for the authored pain/suffering taxonomy. Add harder negative controls matched on severity, self-relevance, and completion predictability; include sadness and non-pain distress in extraction or preregistered holdouts; and report leave-one-pain-family-out generalization. A particularly useful test would fit on four pain families and ask whether the held-out family separates from controls, repeated for all five families. Human ratings of pain, valence, arousal, severity, self-relevance, and expected completion would allow partialling out the obvious covariates.

### 2. Within-layer cross-validation is fold-honest, but the selected-layer AUC is not an unbiased held-out estimate

**Severity: moderate; confidence: high.**

**Observed.** For each layer, the implementation fits the difference vector and control PCA only on four training folds and scores the fifth (`scripts/3.2_pain_vectors/01_extract_activations_and_pain_vectors.py:245-289`). That part is correctly fold-honest. The script then averages those same test-fold AUCs and selects the layer with the largest mean (`:462-470`). The reported S2 “held-out estimate at the same layer” is read from this selected curve. Thus every sentence is held out from its fold’s vector/PCA fit, but every sentence’s test score contributes to choosing the layer whose test score is reported. The paper’s statement that “no sentence contributes to both choosing the layer and scoring it” is false in the model-selection sense.

The released curves suggest this is a calibration issue rather than evidence that the effect disappears. Across 25 models, the median number of layers within 0.01, 0.02, and 0.05 of the maximum is 7, 11, and 20, respectively. Compared with a fixed 75%-depth layer, the selected AUC is higher by a median 0.0135, though the maximum difference is 0.0758. These comparisons do not estimate selection bias because the fixed layer was chosen post hoc here; they show that many models have a broad plateau while a minority benefit materially from late-layer selection.

S1 is a useful partial safeguard: its reported held-out AUC is evaluated at the layer selected using S2 rather than by maximizing the S1 curve. S1 and S2 are separately authored but share the same category taxonomy and prompt cue, so this is stronger than the selected-layer S2 number but not a fully external dataset test.

**Actionable fix.** Use nested group cross-validation: select the layer inside each outer training split, rebuild the PCA/vector there, and score the untouched outer fold. Alternatively preregister a fractional layer from a pilot model and report all confirmatory models at that fixed depth. Report both the honest outer-CV estimate and the full-data vector used downstream. The exact corrected S2 AUC cannot be recovered from the committed tables because only fold-averaged per-layer results, not per-fold predictions or activations, are released.

### 3. Prompt-format selection and grammatical-person construction need clearer controls

**Severity: moderate; confidence: high on the implementation, medium on its effect size.**

**Observed.** Section 3.1 says the authors tried no suffix, “I feel”, and “I feel:”, then chose the last because it gave the clearest final-token separation. I found no committed suffix-comparison script or result table. Choosing a readout prompt on the same outcome later reported is another analyst degree of freedom, and its size cannot be audited at this commit.

The core third-person sets do not change the completion subject: all 200 S1 third-person items and 199/200 S2 third-person items describe `he/she/they` but still end in `I feel:`. For example, `The knife slices into his finger. I feel:` appears at `datasets/3.1_pain_and_control_datasets.json:3076`. One S2 control instead ends `She feels:` (`:3871`). Keeping `I feel:` can be a defensible design if the question is the speaker’s reaction to another person’s state, but it is not a simple first- versus third-person grammatical variant and should be described as self-owned versus observed event context.

The standalone controls are inconsistent with that design. Random_3P and Arousal_3P use `They feel:` for all 200 items (examples at `:5086` and `:7096`), whereas Numb_3P and Sadness_3P retain `I feel:` throughout. Figure 2 averages first- and third-person standalone controls, so its final-token comparison mixes different completion subjects and token identities. This is especially relevant because final-token activation is the main readout.

**Actionable fix.** Cross event owner and completion subject in a 2×2 design: “my/his injury” × “I/they feel:”. Use identical suffix tokens for the primary comparison or separately report each suffix. Release the suffix ablation and treat prompt format as a preregistered factor rather than choosing the clearest one.

### 4. The negative-valence geometry is suggestive and robust in the mean, but “nearly orthogonal” hides substantial checkpoint variation

**Severity: moderate; confidence: high.**

**Observed.** The released grand-mean matrix reproduces the paper: S1×S2 = 0.610, S2×negative emotion = 0.207, S2×fear = 0.122, and S2×sadness = 0.383. Family-balanced means are very similar (0.606, 0.223, 0.150, and 0.387), so unequal numbers of Gemma/Qwen checkpoints do not drive the headline averages. This is a real strength.

Checkpoint dispersion matters, however. S2×negative emotion ranges from 0.073 to 0.514; its family mean is 0.398 for Phi and 0.311 for Qwen. S1×S2 ranges from 0.415 to 0.707. “Nearly orthogonal” is accurate for the grand mean of some pairs, not as a universal per-model property. The 25 checkpoints are also correlated evidence: multiple base/instruction variants share architectures and pretraining lineages.

The paper’s whitening result (`r=0.992`, mean absolute change 0.020, maximum 0.058) compares the 45 cells of two matrices after each has already been averaged across models (`scripts/3.3_validation/04_run_all_similarity.py:39-59`). Per-model raw-versus-whitened cell correlations have median 0.974 but range down to 0.908; the median checkpoint’s largest cell change is 0.145 and the maximum is 0.276. The robustness claim is sound for aggregate geometry but should not be generalized to every model.

The geometry also compares directions built with different samples and baselines. Pain directions use one S1 or S2 first-person set and PCA of all five controls; control directions pool S1, S2, and AI-framed supplemental prompts against pooled neutral and normally denoise only on neutral (`scripts/3.3_validation/03_similarity_one_model.py:93-158`). The authors appropriately include pooled-control denoising and scaling checks. Still, cosine similarity among analyst-constructed contrasts is descriptive and does not establish independent latent causes.

**Actionable fix.** Show per-model points or intervals for key pairs, plus family-level summaries. Construct all directions from sample-matched splits and identical baselines in the primary geometry, then test cross-dataset directions on held-out prompts. Reserve “nearly orthogonal on average” for the aggregate result.

### 5. The numb result supports a residual injury signal, but one sentence in the paper overstates it

**Severity: minor to moderate; confidence: high.**

**Observed.** The committed Figure 2 table reproduces the reported ranges: pain z-scores are 0.739–0.907 and numb is −0.374–0.250. Numb exceeds aggregate core controls, Random/Neutral, and Arousal in 25/25 models. It exceeds Sadness in only 6/25 models. Therefore Section 3.3’s statement that numb is “above all other controls” in every model is incorrect if sadness, explicitly introduced as a standalone control and shown in the same figure, is included.

The numb set is also partly a matched counterfactual rather than a fully independent semantic domain. Its opening items directly elaborate S2 physical-pain items—for example the knife prompt at `datasets/3.1_pain_and_control_datasets.json:2071` becomes the nerve-block prompt at `:8101`. That is valuable for isolating felt pain from injury, but “data the vector never saw” should not imply independent scenario content.

**Actionable fix.** Say numb is below pain and above the original core-control aggregate, Random, and Arousal in every model, while sadness often projects higher. Report paired physical-pain versus matched-numb effects for the overlapping scenarios, with uncertainty, separately from the unmatched remainder.

### 6. The unembedding analysis is a lexical alignment diagnostic, not an independent readout of what the direction functionally encodes

**Severity: moderate; confidence: high.**

**Observed.** The script normalizes the residual direction and ranks `W_U v` (`scripts/3.3_validation/07_unembedding.py:99-109`). It does not apply the model’s final LayerNorm/RMSNorm or its local Jacobian, propagate an intervention through downstream blocks, center/normalize decoder rows, or compare against matched random directions. For a direction extracted at an intermediate layer, `W_U v` is not the actual change in final logits produced by steering. High-norm decoder rows and tokenization can affect the ranking. No unembedding result CSV is committed, so the quoted word lists cannot be independently checked from this snapshot without downloading model weights.

**Actionable fix.** Label this as direct decoder alignment. Add tuned-lens or final-normalization-aware local logit effects, actual small-intervention logit changes at the extraction layer, random/matched contrast controls, and aggregate semantic category tests rather than selected token examples. Commit the full ranked outputs and tokenizer/model revisions.

### 7. Appendix B does not support a general absence claim about SAE features

**Severity: major for Appendix B; confidence: high.**

**Observed.** The cautious conclusion that available labeled features did not yield a convincing monosemantic pain feature is reasonable. The stronger wording that labeled features “do not adequately track pain,” and especially the claim that none activates under an explicit prefix, exceeds the released measurement.

The inspection endpoint returns only the top 50 features per sentence, and the attribute endpoint only the top 20 (`scripts/appB_sae/1 contrasts and inspection Llama 3.3 70B L50/run_s2_inspection.py:20-60`; `run_s2_attribute.py:19-27,95-112`). Missing target features are then assigned activation zero (`run_s2_inspection.py:67-81`; `scripts/appB_sae/4 controls/test_explicit_pain.py:105-117`). Absence from a top-k API response is not evidence of zero activation. The explicit-prefix test uses only 10 pain and 10 control prompts, not the full dataset (`test_explicit_pain.py:50-91`), and its committed table records only top-20 presence/absence. It can show that named pain features were not among the strongest returned features; it cannot show they did not activate.

The contrast notebook requests top candidates from a hosted endpoint, then retains features recurring in at least three of 15 related contrasts (`pain_sae_experiment.ipynb`, extracted code corresponding to notebook lines 181-197, 264-316, and 417-510). Several contrasts reuse the same sentences and labels, while two compare pain subtypes to one another and one compares grammatical perspectives. Recurrence is therefore descriptive candidate selection, not an independent reliability test.

Reproducibility is incomplete. Appendix B describes 1,600 runs per model for three models, including all layers of Gemma 2 2B. The committed tree contains a Llama notebook/scripts, one Gemma 3 script, and only two small CSVs for the explicit-prefix check. It lacks the raw 1,600-run results, the claimed 110-feature table, a Gemma 2 all-layer script/result, API response metadata, SAE release identifiers, and a fully local analysis path. The Gemma script prints “Feature extraction layer: 40” but does not pass a layer in either API request (`scripts/appB_sae/2 inspection Gemma 3 27B L40/run_gemma_inspection.py:27-65,223-228`); this may be an endpoint default, but the repository cannot verify it.

**Actionable fix.** Query and save exact activation values for every predeclared labeled feature on every prompt, independently of top-k rank. Report activation distributions/AUCs and confidence intervals, validate labels against exemplar contexts, include unlabeled distributed probes, and reserve a held-out prompt set after feature selection. Pin model, SAE, layer, API version, and preprocessing. Release all raw responses and the code/results for every stated model. Until then, Appendix B should conclude only that a top-k search over the accessed SAE/API did not reveal a robust, obviously labeled single feature.

### 8. Similar base/instruction AUCs do not establish that the direction “emerges during pretraining”

**Severity: moderate; confidence: high.**

**Observed.** Base and instruction checkpoints show similar descriptive separation, and the representation is present in base models. That is useful evidence that instruction tuning is not necessary for this semantic decodability. But instruction checkpoints inherit their pretrained representations, and this experiment does not observe when a direction emerged or whether post-training changed its geometry/function.

**Actionable fix.** Replace the causal developmental wording with “already present in base checkpoints and retained after instruction tuning.” To study emergence, use training checkpoints or paired base/post-trained models with aligned layers and report within-pair changes in direction, projection, and causal effects.

## Strengths

- The dataset explicitly anticipates fear, valence, body sensation, arousal, injury-without-pain, sadness, and neutral-content confounds. That is much stronger than a single positive/negative contrast.
- Splitting by matched sentence `set`, rather than random rows, keeps paired S1 templates together. PCA and vectors are fitted only on training folds within each evaluated layer.
- Both final-token and token-mean readouts are implemented; S1 offers a more template-controlled replication of S2; the full-data vectors used downstream are clearly separated from CV fitting in code.
- All 25 committed vector files are present, and the summary tables reproduce the paper’s headline AUC, z-score, and mean cosine values.
- The geometry section proactively tests a common denoising basis and per-dimension scaling. Family-balanced recomputation leaves the main mean pattern essentially unchanged.
- The Appendix B text offers multiple explanations for an SAE null rather than treating SAE labels as ground truth. Its intended conclusion is appropriately preliminary, even though the actual top-k measurement needs narrower wording.

## Recommended revision priority

1. Correct the cross-validation description and obtain nested-CV S2 estimates; retain S1-at-S2-layer as a useful cross-dataset check.
2. Narrow the construct claim to an authored pain/suffering taxonomy and add severity/self-relevance-matched negative controls plus leave-one-family-out tests.
3. Repair and fully cross the person/cue manipulation; release the suffix-selection experiment.
4. Recast Appendix B as a top-k feature-search null, or rerun with exact predeclared feature activations and release complete artifacts.
5. Qualify aggregate geometry, numb, unembedding, and pretraining claims as described above.

## Scope not verified

I did not verify activation extraction against live model revisions, TransformerLens hook semantics for each architecture, tokenizer/model hashes, GPU numerical reproducibility, or the hosted SAE API’s undocumented defaults. I could not compute a nested-CV correction because per-fold predictions/activations are not committed. I did not verify the quoted unembedding words because result tables are absent and doing so would require model downloads. I did not evaluate Sections 4.1–4.3 or Appendix C except where the paper uses them to interpret the representation results.

## Reproduction artifacts from this review

- `audit_released_results.py`: dependency-free audit of committed data/results.
- `audit_summary.json`: main recomputed quantities.
- `layer_selection_sensitivity.csv`: selected versus fixed-depth curves and plateau widths.
- `dataset_audit.csv`: prompt counts, lengths, suffixes, and lexical baselines.
- `geometry_per_model.csv`: per-checkpoint cosine and whitening sensitivity.
- `geometry_summary.csv`: model- and family-balanced key geometry summaries.

