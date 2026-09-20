# Scripts

All runnable with `/work/pain-axis-review/.venv/bin/python`.
Captured output in `../out/`. Repo paths are relative to `/work/Pain-axis/`.
Nothing in the authors' clone was modified.

| script | findings | what it does |
|---|---|---|
| `x1_geometry.py <model>` | 1 | fraction of the pain vs control directions inside the projected-out basis; four symmetric constructions on real activations; partial cosines; double leave-one-category-out with a random-grouping null; out-of-sample pain-vs-fear separability |
| `x2_layers.py` | 2, 11 | extraction vs S1/S2 steering layer for 25 models; held-out AUC **at the Section 4.1 layer** from the authors' released layerwise curves |
| `x3_selfother.py` | 3 | Section 4.1 stratum means and paired tests; unit-of-analysis alternatives; what "opposite pattern" requires |
| `x4_steering.py` | 4, 5 | three explicit ladder criteria over 10,000 S2 generations; the low-dose models' full sweeps; three nested bodily lexicons on S1 and S2; 40 flagged S1 generations for hand adjudication |
| `x5_ablation.py` | 6 | ablation residues per condition and model; the right comparator; the Gemma 2 2B instruct progression against amount of direction removed |
| `x6_50x_numb.py` | 8 | numb vs every Figure 2 column (its F9 block is superseded by `x6b`) |
| `x6b_50x.py` | 9 | settles r1 vs x2 on "50 times": full-softmax `p_pain` vs the truncated `top20` string, same rows |
| `x7_confound.py` | 7 | second-person density vs the pain axis; covariate horse-race; the 44 zero-"you" self-directed scenarios |
| `x8_supp_and_geometry2.py <model> <s1L> <s2L>` | 12, 1, 6 | control directions with/without `ControlSupplement_1P`; grand-mean construction; cross-layer S1xS2 cosine |
| `x9_qwen_labels.py` | 10 | every consequence of the Qwen 3 mislabel |
