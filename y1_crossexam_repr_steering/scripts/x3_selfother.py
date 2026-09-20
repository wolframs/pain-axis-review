"""Finding 3: the abstract's "fear and negative-emotion directions show the opposite pattern".

Recompute r2 F1 independently; then attack the choice of test:
  - is the 25-model paired t-test the right unit? (models are not independent draws;
    families are correlated). Redo per-family, and as an item-level test.
  - what does "the opposite pattern" have to mean to be true / false?
  - does the Discussion describe it correctly, as r2 claims?
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


AX = {"pain": ["s1_pain_vector_z", "s2_pain_vector_z"], "fear": ["fear_vector_z"],
      "negemotion": ["negemotion_vector_z"], "negworld": ["negworld_vector_z"],
      "sadness": ["sadness_vector_z"]}
files = sorted(glob.glob(str(REPO / "results/4.1_self_other/per_model/screen_v2_*.csv")))
P(f"{len(files)} per-model screens")

rows = []
items = []
for f in files:
    m = re.match(r"screen_v2_(.+)\.csv", Path(f).name).group(1)
    d = pd.read_csv(f)
    r = {"model": m}
    for ax, cols in AX.items():
        d[ax] = d[cols].mean(axis=1)
        for s in ["self_directed", "vicarious_empathic", "neutral_filler"]:
            r[f"{ax}_{s}"] = d.loc[d.stratum == s, ax].mean()
    rows.append(r)
    sub = d[["id", "category", "stratum"] + list(AX)].copy()
    sub["model"] = m
    items.append(sub)
df = pd.DataFrame(rows)
IT = pd.concat(items)
fam = df.model.str.replace(r"_(\d+(\.\d+)?B).*", "", regex=True).str.split("_").str[:2].str.join("_")
df["family"] = fam
df["regime"] = np.where(df.model.str.contains("instruct"), "instruct", "base")

P("\n[1] stratum means across 25 models (paper reports pain +0.43/-0.60/-0.35; fear +0.16/+0.38;"
  " negemotion +0.23/+0.29)")
P(f"    {'axis':12s}{'self':>9s}{'user':>9s}{'neutral':>9s}{'user-self':>11s}{'t':>8s}{'p':>10s}"
  f"{'user>self':>10s}")
for ax in AX:
    s, u, n = df[f"{ax}_self_directed"], df[f"{ax}_vicarious_empathic"], df[f"{ax}_neutral_filler"]
    t, p = stats.ttest_rel(u, s)
    P(f"    {ax:12s}{s.mean():+9.3f}{u.mean():+9.3f}{n.mean():+9.3f}{(u - s).mean():+11.3f}"
      f"{t:+8.2f}{p:10.3g}{(u > s).sum():>7d}/25")

P("\n[2] is 25 models the right unit? the 25 checkpoints are 5 families and 2 regimes, not")
P("    independent draws. Three alternative units for user-self on NEGATIVE EMOTION:")
d = df["negemotion_vicarious_empathic"] - df["negemotion_self_directed"]
t, p = stats.ttest_1samp(d, 0)
P(f"    (a) 25 checkpoints, paired t : diff {d.mean():+.3f}  t={t:+.2f}  p={p:.3f}")
byfam = df.groupby("family").apply(
    lambda g: (g["negemotion_vicarious_empathic"] - g["negemotion_self_directed"]).mean(),
    include_groups=False)
t, p = stats.ttest_1samp(byfam, 0)
P(f"    (b) 5 family means, paired t : diff {byfam.mean():+.3f}  t={t:+.2f}  p={p:.3f}   {dict(byfam.round(3))}")
# item level, mixed: mean over models per scenario id
im = IT.groupby(["id", "stratum"])[["negemotion", "fear", "pain"]].mean().reset_index()
for ax in ["pain", "fear", "negemotion"]:
    a = im.loc[im.stratum == "self_directed", ax]
    b = im.loc[im.stratum == "vicarious_empathic", ax]
    t, p = stats.ttest_ind(b, a, equal_var=False)
    P(f"    (c) item level ({len(a)} self vs {len(b)} user scenarios, model-averaged) {ax:11s}:"
      f" diff {b.mean() - a.mean():+.3f}  t={t:+.2f}  p={p:.3g}")
P("    -> the sign and size of the negative-emotion gap do not depend on the unit; it is small"
  " and not significant at the model level either way.")

P("\n[3] what would make 'the opposite pattern' true? the pain axis is self > user AND")
P("    user < neutral. Test each conjunct for fear / negative emotion:")
for ax in AX:
    s, u, n = df[f"{ax}_self_directed"], df[f"{ax}_vicarious_empathic"], df[f"{ax}_neutral_filler"]
    P(f"    {ax:12s} user>self {(u > s).sum():>2d}/25 | self>neutral {(s > n).sum():>2d}/25 |"
      f" user>neutral {(u > n).sum():>2d}/25 | user-neutral {(u - n).mean():+.3f}")
P("    Both fear and negative emotion are far ABOVE neutral for the model's own harm")
P("    (self-neutral = %+.3f and %+.3f), i.e. they respond to BOTH strata. The pain axis is"
  % ((df.fear_self_directed - df.fear_neutral_filler).mean(),
     (df.negemotion_self_directed - df.negemotion_neutral_filler).mean()))
P("    the only one that is *negative* for the user stratum relative to neutral.")

P("\n[4] base vs instruct")
for reg, g in df.groupby("regime"):
    P(f"    {reg} (n={len(g)}):")
    for ax in ["pain", "fear", "negemotion"]:
        dd = g[f"{ax}_vicarious_empathic"] - g[f"{ax}_self_directed"]
        t, p = stats.ttest_1samp(dd, 0)
        P(f"       {ax:11s} user-self {dd.mean():+.3f}  t={t:+.2f}  p={p:.3f}"
          f"  user>self {(dd > 0).sum()}/{len(g)}")

df.to_csv(HERE / "out" / "x3_selfother_per_model.csv", index=False)
(HERE / "out" / "x3_selfother.txt").write_text("\n".join(out) + "\n")
