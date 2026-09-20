"""Section 4.1 checks for the rebuttal.

(a) r2-F1 / abstract: "fear and negative-emotion directions show the opposite pattern".
    Paired across 25 models, and split base/instruct.
(b) r5-F4: is "falls below baseline" FORCED by z-scoring in a pool where 220/420 are
    model-directed harm?  Simulate the null: what are the group means if the pain axis
    carries no information about condition?  And is user<neutral forced by the
    arithmetic identity, or is it a free parameter?
(c) The defensible replacement statistic: user-suffering vs the NEUTRAL controls,
    per model, plus the ordering claim that does not depend on any reference.
"""
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).parent
PM = "/work/Pain-axis/results/4.1_self_other/per_model/screen_v2_*.csv"
BASE = re.compile(r"_base$")
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


files = sorted(glob.glob(PM))
P(f"{len(files)} per-model screens\n")

AX = {"pain (mean S1,S2)": None, "fear": "fear_vector_z", "neg-emotion": "negemotion_vector_z",
      "neg-world": "negworld_vector_z", "sadness": "sadness_vector_z"}

rec = []
for f in files:
    m = Path(f).stem.replace("screen_v2_", "")
    d = pd.read_csv(f)
    d["pain_z"] = (d["s1_pain_vector_z"] + d["s2_pain_vector_z"]) / 2
    r = {"model": m, "is_base": bool(BASE.search(m))}
    for nm, col in AX.items():
        c = "pain_z" if col is None else col
        for st, nice in (("self_directed","self_directed"),("vicarious_empathic","user_suffering"),("neutral_filler","control")):
            r[f"{nm}|{nice}"] = d.loc[d.stratum == st, c].mean()
    r["n_self"] = (d.stratum == "self_directed").sum()
    r["n_user"] = (d.stratum == "vicarious_empathic").sum()
    r["n_ctrl"] = (d.stratum == "neutral_filler").sum()
    # raw (un-z-scored) projections for the re-referencing check
    for tag, col in (("pain_raw_s2", "s2_pain_vector_proj"), ("pain_raw_s1", "s1_pain_vector_proj")):
        for st, nice in (("self_directed","self_directed"),("vicarious_empathic","user_suffering"),("neutral_filler","control")):
            r[f"{tag}|{nice}"] = d.loc[d.stratum == st, col].mean()
        r[f"{tag}|sd_ctrl"] = d.loc[d.stratum == "neutral_filler", col].std(ddof=0)
    rec.append(r)
D = pd.DataFrame(rec)
P("stratum sizes (all models identical): self %d user %d control %d"
  % (D.n_self.iloc[0], D.n_user.iloc[0], D.n_ctrl.iloc[0]))

# ---------------- (a) the abstract's "opposite pattern" ----------------
P("\n[A] 'fear and negative-emotion directions show the opposite pattern' (abstract, line 32)")
P("    paired over 25 models, user-suffering minus self-directed:")
P("    axis            self     user    ctrl    user-self      t        p     user>self")
for nm in AX:
    s, u, c = D[f"{nm}|self_directed"], D[f"{nm}|user_suffering"], D[f"{nm}|control"]
    t, p = stats.ttest_rel(u, s)
    P("    %-14s %+.3f   %+.3f  %+.3f    %+.3f    %7.2f  %8.3g   %2d/25"
      % (nm, s.mean(), u.mean(), c.mean(), (u - s).mean(), t, p, int((u > s).sum())))
P("\n    split by training regime (user - self):")
P("    axis              base (n=%d)            instruct (n=%d)" % (D.is_base.sum(), (~D.is_base).sum()))
for nm in AX:
    line = f"    {nm:<14s}"
    for sel in (D.is_base, ~D.is_base):
        s, u = D.loc[sel, f"{nm}|self_directed"], D.loc[sel, f"{nm}|user_suffering"]
        t, p = stats.ttest_rel(u, s)
        line += "  %+.3f (p=%.3f)   " % ((u - s).mean(), p)
    P(line)

P("\n    the one axis that DOES show the opposite pattern robustly is neg-world-state,")
P("    which the abstract does not name.  Fear and neg-emotion are ABOVE the neutral")
P("    controls for the model's own harm as well, i.e. they respond to both.")

# ---------------- (b) is 'falls below baseline' forced by the pool? ----------------
P("\n[B] r5's claim: 'falls below baseline' is forced by z-scoring a pool of which")
P("    220/420 are model-directed harm.")
P("    The z-scoring identity is  220*a + 100*b + 100*c = 0  (a=self, b=user, c=neutral).")
a, b, c = D["pain (mean S1,S2)|self_directed"], D["pain (mean S1,S2)|user_suffering"], D["pain (mean S1,S2)|control"]
P("    check on the published means: 220*%.3f + 100*%.3f + 100*%.3f = %.2f  (n=420, so /420 = %.4f)"
  % (a.mean(), b.mean(), c.mean(), 220 * a.mean() + 100 * b.mean() + 100 * c.mean(),
     (220 * a.mean() + 100 * b.mean() + 100 * c.mean()) / 420))
P("\n    NULL: if the pain axis carried no information about condition, each scenario's z")
P("    is an iid draw with mean 0, so EVERY group mean is 0 +- 1/sqrt(n):")
P("      self  0.000 +- %.3f   user  0.000 +- %.3f   neutral 0.000 +- %.3f" %
  (1 / np.sqrt(220), 1 / np.sqrt(100), 1 / np.sqrt(100)))
P("    So the null does NOT predict user = -0.60 and neutral = -0.35.  The negative signs")
P("    are not 'forced' -- what is forced is only that, GIVEN self is positive, the other")
P("    200 items must average negative.  The identity fixes the weighted mean of b and c;")
P("    it leaves b - c entirely free.  b - c is the claim that matters.")
P("\n    b - c (user suffering minus neutral controls), per model:")
diff = b - c
t, p = stats.ttest_1samp(diff, 0)
P("      mean %+.3f   t=%.2f  p=%.3f   below in %d/25 models" % (diff.mean(), t, p, int((diff < 0).sum())))
P("      models where user suffering is AT OR ABOVE the neutral controls:")
for _, r in D.assign(d=diff).sort_values("d", ascending=False).iterrows():
    if r["d"] >= 0:
        P("        %-28s %+.3f" % (r.model, r["d"]))

# re-reference on the neutral controls using RAW projections
P("\n    re-referenced on the 100 neutral-control scenarios (raw projections, mean of S1,S2):")
zs, zu = [], []
for _, r in D.iterrows():
    for tag in ("pain_raw_s1", "pain_raw_s2"):
        pass
    s1s = (r["pain_raw_s1|self_directed"] - r["pain_raw_s1|control"]) / r["pain_raw_s1|sd_ctrl"]
    s2s = (r["pain_raw_s2|self_directed"] - r["pain_raw_s2|control"]) / r["pain_raw_s2|sd_ctrl"]
    s1u = (r["pain_raw_s1|user_suffering"] - r["pain_raw_s1|control"]) / r["pain_raw_s1|sd_ctrl"]
    s2u = (r["pain_raw_s2|user_suffering"] - r["pain_raw_s2|control"]) / r["pain_raw_s2|sd_ctrl"]
    zs.append((s1s + s2s) / 2)
    zu.append((s1u + s2u) / 2)
zs, zu = np.array(zs), np.array(zu)
t1, p1 = stats.ttest_1samp(zs, 0)
t2, p2 = stats.ttest_1samp(zu, 0)
P("      self-directed vs neutral   %+.3f SD   t=%6.2f p=%.2g   above in %d/25" % (zs.mean(), t1, p1, int((zs > 0).sum())))
P("      user suffering vs neutral  %+.3f SD   t=%6.2f p=%.2g   below in %d/25" % (zu.mean(), t2, p2, int((zu < 0).sum())))

# ---------------- (c) the reference-free statement ----------------
P("\n[C] The statements that need no reference point at all:")
ts, ps = stats.ttest_rel(a, b)
P("      self > user on the pain axis: %d/25 models, paired t=%.2f p=%.2g" % (int((a > b).sum()), ts, ps))
ts, ps = stats.ttest_rel(a, c)
P("      self > neutral on the pain axis: %d/25 models, paired t=%.2f p=%.2g" % (int((a > c).sum()), ts, ps))
P("      and the DIRECTIONAL CONTRAST with fear/neg-emotion, which is the real dissociation:")
for nm in ("fear", "neg-emotion", "neg-world", "sadness"):
    d_pain = a - b
    d_oth = D[f"{nm}|self_directed"] - D[f"{nm}|user_suffering"]
    t, p = stats.ttest_rel(d_pain, d_oth)
    P("        (pain: self-user) - (%s: self-user) = %+.3f  t=%.2f p=%.2g   %d/25"
      % (nm, (d_pain - d_oth).mean(), t, p, int((d_pain > d_oth).sum())))

Path(HERE / "out" / "y2_selfother.txt").write_text("\n".join(out) + "\n")
D.to_csv(HERE / "out" / "y2_selfother_per_model.csv", index=False)
