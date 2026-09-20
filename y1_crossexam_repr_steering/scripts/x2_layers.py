"""Finding 2: Section 4.1 projects at the S1 STEERING layer, not the extraction layer.
Verify in code; then ask the question both r1 (F2) and r2 (F11) say cannot be answered:
how well do the directions separate pain from controls AT THAT LAYER?

Both reviewers say no such evidence is released. It is:
  results/3.2_pain_vectors/per_model/<m>/layer_curves.csv  -> S2_1P and S2_3P held-out
      5-fold AUC at EVERY layer (final_token and mean)
  results/3.2_pain_vectors/auc_tables/s1_kfold_layer_curves.csv -> the same for S1_1P/S1_3P
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/work/Pain-axis")
HERE = Path(__file__).resolve().parent.parent
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


def steer_layers(tag):
    d = {}
    for f in glob.glob(str(REPO / f"results/4.2_steering/{tag}/*_steering_{tag}_neutral50_L*.csv")):
        m = re.match(rf"(.+)_steering_{tag}_neutral50_L(\d+)\.csv", Path(f).name)
        d[m.group(1)] = int(m.group(2))
    return d


s1L, s2L = steer_layers("S1"), steer_layers("S2")
P(f"models with S1 steering layer: {len(s1L)}, S2: {len(s2L)}")

# extraction layers from the released pain_vectors.pt filenames' sibling similarity CSVs
extL = {}
for f in glob.glob(str(REPO / "results/3.3_validation/cosine_similarity/similarity_*_L*.csv")):
    m = re.match(r"similarity_(.+)_L(\d+)\.csv", Path(f).name)
    if m.group(1).startswith("alldenoise_") or m.group(1).startswith("whiten"):
        continue
    extL[m.group(1)] = int(m.group(2))

s1curves = pd.read_csv(REPO / "results/3.2_pain_vectors/auc_tables/s1_kfold_layer_curves.csv")
rows = []
for mdl in sorted(extL):
    s2c = pd.read_csv(REPO / f"results/3.2_pain_vectors/per_model/{mdl}/layer_curves.csv")
    s2c = s2c[(s2c.extraction == "final_token") & (s2c.dataset == "S2_1P")].set_index("layer")
    s1c = s1curves[(s1curves.model == mdl) & (s1curves.extraction == "final_token")
                   & (s1curves.dataset == "S1_1P")].set_index("layer")
    nl = int(s2c.index.max()) + 1
    r = dict(model=mdl, n_layers=nl, ext=extL[mdl], s1_steer=s1L.get(mdl), s2_steer=s2L.get(mdl),
             s2auc_ext=s2c.auc_vs_all_controls.get(extL[mdl]),
             s2auc_at_s1steer=s2c.auc_vs_all_controls.get(s1L.get(mdl)),
             s2auc_at_s2steer=s2c.auc_vs_all_controls.get(s2L.get(mdl)),
             s1auc_ext=s1c.auc_vs_all_controls.get(extL[mdl]) if len(s1c) else np.nan,
             s1auc_at_s1steer=s1c.auc_vs_all_controls.get(s1L.get(mdl)) if len(s1c) else np.nan,
             s2auc_best=s2c.auc_vs_all_controls.max())
    rows.append(r)
df = pd.DataFrame(rows)
df["depth_ext"] = df.ext / (df.n_layers - 1)
df["depth_s1steer"] = df.s1_steer / (df.n_layers - 1)
df["depth_s2steer"] = df.s2_steer / (df.n_layers - 1)
df.to_csv(HERE / "out" / "x2_layers.csv", index=False)

P("\n[1] layer depths (fraction of network)")
for c in ["depth_ext", "depth_s1steer", "depth_s2steer"]:
    P(f"    {c:16s} median {df[c].median():.2f}  range {df[c].min():.2f}-{df[c].max():.2f}")
P(f"    extraction layer later than S1 steering layer in {(df.ext > df.s1_steer).sum()}/25;"
  f" median gap {int((df.ext - df.s1_steer).median())} layers, max {int((df.ext - df.s1_steer).max())}")

P("\n[2] HELD-OUT 5-fold AUC (pain vs all controls, final token), from the authors' own released")
P("    layerwise curves -- the evidence r1 F2 and r2 F11 say does not exist:")
for lbl, a, b in [("S2 vector at the EXTRACTION layer   ", "s2auc_ext", None),
                  ("S2 vector at the S1 STEERING layer  ", "s2auc_at_s1steer", None),
                  ("S2 vector at the S2 STEERING layer  ", "s2auc_at_s2steer", None),
                  ("S1 vector at the EXTRACTION layer   ", "s1auc_ext", None),
                  ("S1 vector at the S1 STEERING layer  ", "s1auc_at_s1steer", None)]:
    v = df[a].dropna()
    P(f"    {lbl} n={len(v)}  median {v.median():.3f}  range {v.min():.3f}-{v.max():.3f}"
      f"  <0.90 in {(v < 0.90).sum()}  <0.80 in {(v < 0.80).sum()}")
P("\n    per-model S2 held-out AUC at the layer Section 4.1 actually reads (S1 steering layer):")
for _, r in df.sort_values("s2auc_at_s1steer").iterrows():
    P(f"      {r.model:26s} L{int(r.s1_steer):>2d}/{r.n_layers - 1:<2d}  S2 {r.s2auc_at_s1steer:.3f}"
      f"   (extraction L{int(r.ext)}: {r.s2auc_ext:.3f})   S1 {r.s1auc_at_s1steer:.3f}")

# what the paper reports
P("\n[3] paper's reported held-out ranges (footnote 2, line 282): S2 0.91-1.00 median 0.98; S1 0.85-0.94")
v = df.s2auc_at_s1steer.dropna()
P(f"    below the bottom of the reported S2 range (0.907) at the 4.1 layer: {(v < 0.907).sum()}/25")
v1 = df.s1auc_at_s1steer.dropna()
P(f"    below the bottom of the reported S1 range (0.847) at the 4.1 layer: {(v1 < 0.847).sum()}/25")

(HERE / "out" / "x2_layers.txt").write_text("\n".join(out) + "\n")
