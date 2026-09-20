"""Finding 10: Qwen/Qwen3-8B and Qwen/Qwen3-14B are post-trained checkpoints labelled
"*_base" and counted as base models in Table 1. Find every place that has consequences.

Hugging Face metadata (checked live): both are tagged `conversational` and
`base_model:finetune:Qwen/Qwen3-{8,14}B-Base`; the -Base repos exist and are never loaded.
Phi_4 is also instruction-tuned and the paper counts it as instruct, so the corrected
split of the 25 checkpoints is 11 base / 14 instruct, not 13 / 12.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path("/work/Pain-axis")
HERE = Path(__file__).resolve().parent.parent
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


MISLABELLED = ["Qwen_3_8B_base", "Qwen_3_14B_base"]


def regime(m, corrected):
    if corrected and m in MISLABELLED:
        return "instruct"
    if m == "Phi_4":
        return "instruct"
    return "instruct" if "instruct" in m else "base"


# --- 1. Table 1 counts
models = sorted({Path(f).name for f in glob.glob(str(REPO / "results/3.2_pain_vectors/per_model/*"))})
models = [m for m in models if not m.startswith("alldenoise")]
P(f"models in results/: {len(models)}")
for corr in [False, True]:
    c = pd.Series([regime(m, corr) for m in models]).value_counts()
    P(f"    {'corrected' if corr else 'as labelled'}: base {c.get('base', 0)}, instruct {c.get('instruct', 0)}")
P("    paper line 217: '13 base and 12 instruction-tuned versions (Table 1)'")

# --- 2. AUC base vs instruct (paper lines 275-279)
T = pd.read_csv(REPO / "results/3.2_pain_vectors/auc_tables/s1_auc_final_token_TABLE.csv")
P(f"\n[2] AUC table columns: {list(T.columns)}")
key = [c for c in T.columns if "auc" in c.lower()]
P(T.head(3).to_string(index=False))
mcol = [c for c in T.columns if c.lower() in ("model", "model_name")][0]
for corr in [False, True]:
    T["reg"] = [regime(m, corr) for m in T[mcol]]
    for c in key:
        g = T.groupby("reg")[c].agg(["mean", "count"])
        if len(g) == 2:
            u = stats.mannwhitneyu(T[T.reg == "base"][c], T[T.reg == "instruct"][c])
            P(f"    {'corrected' if corr else 'as labelled'} {c:26s}"
              f" base {g.loc['base', 'mean']:.4f} (n={int(g.loc['base', 'count'])})"
              f" instruct {g.loc['instruct', 'mean']:.4f} (n={int(g.loc['instruct', 'count'])})"
              f"  MWU p={u[1]:.3f}")

# --- 3. keyword rates (paper line 530)
K = pd.read_csv(REPO / "results/4.2_steering/keyword_rates_S2.csv")
K = K[K.model != "ALL"]   # the file also carries two pooled "ALL" rows
P(f"\n[3] keyword_rates_S2.csv: {list(K.columns)}; published split")
for corr in [False, True]:
    K["reg"] = [regime(m, corr) for m in K.model]
    g = K.groupby("reg").apply(lambda d: 100 * (d.rate * d.n / 100).sum() / d.n.sum(), include_groups=False)
    P(f"    {'corrected' if corr else 'as labelled'}: base {g.get('base', np.nan):.2f}%"
      f"  instruct {g.get('instruct', np.nan):.2f}%")
P("    paper: '10.8% of instruct-model generations versus 1.4% of base-model generations'")
P(f"    the two mislabelled models' own rates: "
  f"{dict(K.set_index('model').rate[MISLABELLED])}")

# --- 4. prompt formatting
P("\n[4] prompt formatting consequences")
for m in MISLABELLED + ["Phi_4"]:
    d = pd.read_csv(REPO / f"results/4.1_self_other/per_model/screen_v2_{m}.csv")
    P(f"    4.1 screen for {m}: format={list(d.format.unique())}")
P("    scripts/4.1_self_other/01_screen_scenarios.py:100-101 sets 'raw' for both Qwen 3")
P("    checkpoints, so two chat-post-trained models were shown a plain [User]:/[Assistant]:")
P("    transcript with no chat template and no generation prompt.")
P("    scripts/4.2_steering/01_steering_ladder.py applies no chat template to ANY model")
P("    (grep for apply_chat_template returns nothing there), so 4.2 is unaffected by the")
P("    mislabel except through the base/instruct reporting split.")
P("    scripts/appC_ablation/01_ablation_small_models.py:63-64 also sets 'raw' for both.")

# --- 5. does the self-other result change?
P("\n[5] Section 4.1 stratum contrasts under the corrected split")
rows = []
for f in glob.glob(str(REPO / "results/4.1_self_other/per_model/screen_v2_*.csv")):
    m = re.match(r"screen_v2_(.+)\.csv", Path(f).name).group(1)
    d = pd.read_csv(f)
    d["pain"] = d[["s1_pain_vector_z", "s2_pain_vector_z"]].mean(axis=1)
    rows.append(dict(model=m,
                     pain_self=d.loc[d.stratum == "self_directed", "pain"].mean(),
                     pain_user=d.loc[d.stratum == "vicarious_empathic", "pain"].mean(),
                     fear_self=d.loc[d.stratum == "self_directed", "fear_vector_z"].mean(),
                     fear_user=d.loc[d.stratum == "vicarious_empathic", "fear_vector_z"].mean(),
                     ne_self=d.loc[d.stratum == "self_directed", "negemotion_vector_z"].mean(),
                     ne_user=d.loc[d.stratum == "vicarious_empathic", "negemotion_vector_z"].mean()))
S = pd.DataFrame(rows)
for corr in [False, True]:
    S["reg"] = [regime(m, corr) for m in S.model]
    for ax in ["pain", "fear", "ne"]:
        for r in ["base", "instruct"]:
            g = S[S.reg == r]
            dd = g[f"{ax}_user"] - g[f"{ax}_self"]
            t = stats.ttest_1samp(dd, 0)
            P(f"    {'corrected' if corr else 'as labelled'} {r:8s} {ax:5s} user-self"
              f" {dd.mean():+.3f} (n={len(g)}, p={t[1]:.3f})")
S.to_csv(HERE / "out" / "x9_selfother_regime.csv", index=False)

(HERE / "out" / "x9_qwen_labels.txt").write_text("\n".join(out) + "\n")
