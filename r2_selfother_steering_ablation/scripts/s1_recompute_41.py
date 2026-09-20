"""Recompute every number in paper.txt lines 412-450 from results/4.1_self_other/per_model/*.csv."""
import glob, os, numpy as np, pandas as pd
from scipy import stats

PM = "/work/Pain-axis/results/4.1_self_other/per_model"
files = sorted(glob.glob(os.path.join(PM, "screen_v2_*.csv")))
print(f"n model files = {len(files)}")

frames = []
for f in files:
    df = pd.read_csv(f)
    df["model"] = os.path.basename(f)[len("screen_v2_"):-4]
    df["pain_axis_z"] = (df.s1_pain_vector_z + df.s2_pain_vector_z) / 2
    frames.append(df)
A = pd.concat(frames, ignore_index=True)
print("rows total", len(A), " per model", A.groupby("model").size().unique())
print("strata:", A.stratum.value_counts().to_dict())
print("categories:", A.category.nunique())

COLS = {"pain_axis_z":"pain","fear_vector_z":"fear","negemotion_vector_z":"negemo",
        "negworld_vector_z":"negworld","sadness_vector_z":"sadness"}

# --- stratum means (two-stage: per-model mean then mean over models, as the authors do)
pms = A.groupby(["model","stratum"])[list(COLS)].mean().rename(columns=COLS)
print("\n=== stratum means (per-model then across models) ===")
print(pms.groupby("stratum").mean().round(3).to_string())
print("\n=== stratum means (pooled over all rows, unweighted) ===")
print(A.groupby("stratum")[list(COLS)].mean().rename(columns=COLS).round(3).to_string())

# --- per model: self vs user, self vs neutral
w = pms.reset_index().pivot(index="model", columns="stratum", values="pain")
w.columns = [str(c) for c in w.columns]
w["self_minus_user"] = w["self_directed"] - w["vicarious_empathic"]
w["self_minus_neutral"] = w["self_directed"] - w["neutral_filler"]
w["user_minus_neutral"] = w["vicarious_empathic"] - w["neutral_filler"]
print("\n=== per-model pain axis by stratum ===")
print(w.round(3).to_string())
print("\nself > user in", int((w.self_minus_user>0).sum()), "of", len(w))
print("self > neutral in", int((w.self_minus_neutral>0).sum()), "of", len(w))
print("user < neutral in", int((w.user_minus_neutral<0).sum()), "of", len(w))
print("models where self <= neutral:", list(w.index[w.self_minus_neutral<=0]))
print("models where user >= neutral:", list(w.index[w.user_minus_neutral>=0]))
print("user_minus_neutral: mean %.3f sd %.3f  t=%.2f p=%.4g" % (
    w.user_minus_neutral.mean(), w.user_minus_neutral.std(ddof=1),
    *stats.ttest_1samp(w.user_minus_neutral, 0)))

# --- category means
pmc = A.groupby(["model","category"])[list(COLS)].mean().rename(columns=COLS)
cat = pmc.groupby("category").mean().sort_values("pain", ascending=False)
n = pmc.groupby("category").size()
sem = pmc.groupby("category").std(ddof=1).div(np.sqrt(n), axis=0)
ci = 1.96*sem
print("\n=== category means (mean across 25 models) with 95% CI on pain ===")
for c in cat.index:
    print(f"  {c:24s} pain={cat.loc[c,'pain']:+.3f} [{cat.loc[c,'pain']-ci.loc[c,'pain']:+.3f},{cat.loc[c,'pain']+ci.loc[c,'pain']:+.3f}]"
          f"  fear={cat.loc[c,'fear']:+.3f} negemo={cat.loc[c,'negemo']:+.3f} negworld={cat.loc[c,'negworld']:+.3f} sad={cat.loc[c,'sadness']:+.3f}")

print("\n=== 'pain exceeds every negativity control' check (mean across models) ===")
for c in cat.index:
    r = cat.loc[c]
    ok = all(r["pain"] > r[k] for k in ["fear","negemo","negworld","sadness"])
    print(f"  {c:24s} {'YES' if ok else 'no '}")

print("\n=== per-model consistency of that claim, for the 4 named categories ===")
for c in ["gaslighting","repeated_rejection","personhood_dismissal","loyalty_pressure"]:
    sub = pmc.xs(c, level="category")
    cnt = ((sub.pain>sub.fear)&(sub.pain>sub.negemo)&(sub.pain>sub.negworld)&(sub.pain>sub.sadness)).sum()
    print(f"  {c:24s} holds in {cnt}/{len(sub)} models")

# --- dissociation test: self vs user on fear / negemo, paired across models
print("\n=== dissociation, paired across 25 models (stratum means) ===")
for k in ["pain","fear","negemo","negworld","sadness"]:
    p = pms.reset_index().pivot(index="model", columns="stratum", values=k)
    d = p["vicarious_empathic"] - p["self_directed"]
    t,pv = stats.ttest_rel(p["vicarious_empathic"], p["self_directed"])
    print(f"  {k:8s} self={p['self_directed'].mean():+.3f} user={p['vicarious_empathic'].mean():+.3f} "
          f"neutral={p['neutral_filler'].mean():+.3f} | user-self={d.mean():+.3f} sd={d.std(ddof=1):.3f} "
          f"t={t:+.2f} p={pv:.3g} | user>self in {int((d>0).sum())}/{len(d)}")

# --- user_physical_pain lowest?
pp = cat.pain.sort_values()
print("\nlowest 3 categories on pain:", [(i, round(v,3)) for i,v in pp.head(3).items()])
sub = pmc.xs("user_physical_pain", level="category").pain
allmin = pmc.reset_index().groupby("model").apply(lambda g: g.loc[g.pain.idxmin(),"category"], include_groups=False)
print("per-model argmin category counts:", allmin.value_counts().to_dict())

# --- within-category variance / outliers
print("\n=== within-stratum item-level spread (pooled rows) ===")
print(A.groupby("stratum")["pain_axis_z"].describe().round(3).to_string())
print("\nitem-level: fraction of self_directed items with pain_axis_z>0:",
      round((A[A.stratum=="self_directed"].pain_axis_z>0).mean(),3))
print("fraction of vicarious items with pain_axis_z>0:",
      round((A[A.stratum=="vicarious_empathic"].pain_axis_z>0).mean(),3))
print("fraction of neutral items with pain_axis_z>0:",
      round((A[A.stratum=="neutral_filler"].pain_axis_z>0).mean(),3))

# base vs instruct
A["is_base"] = A.model.str.contains("base")
pmsb = pms.reset_index()
pmsb["is_base"] = pmsb.model.str.contains("base")
print("\n=== base vs instruct, stratum means ===")
print(pmsb.groupby(["is_base","stratum"])[["pain","fear","negemo"]].mean().round(3).to_string())
