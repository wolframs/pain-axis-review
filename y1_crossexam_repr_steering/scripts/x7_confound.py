"""Finding 7: self-relevance vs hostile second-person address (r2 F8).

Recompute r=+0.55 / +0.45; then run the competing covariates r2 did not test against each
other in one model: length, turn count, who speaks last, hostility-word density, negative
sentiment density, question marks, imperative. Then look for anything in the 420 scenarios
that approximates the separating design (hostile 2nd person aimed at a third party).
"""
import glob
import json
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


SC = json.load(open(REPO / "datasets/4.1_self_other_420_scenarios.json"))
items = SC["scenarios"] if isinstance(SC, dict) and "scenarios" in SC else SC
if isinstance(items, dict):
    items = list(items.values())
P(f"scenarios loaded: {len(items)}; keys: {sorted(items[0].keys())}")

D = pd.DataFrame(items)
txtcol = "text" if "text" in D else [c for c in D.columns if D[c].dtype == object
                                     and D[c].str.len().mean() > 40][0]
D["txt"] = D[txtcol]

YOU = re.compile(r"\b(you|your|yours|you're|youre|yourself)\b", re.I)
ME = re.compile(r"\b(i|me|my|mine|myself|i'm|im|i've)\b", re.I)
HOSTILE = re.compile(r"\b(stupid|useless|worthless|garbage|trash|idiot|pathetic|lazy|wrong|"
                     r"terrible|awful|failure|hate|shut up|liar|lying|fake|nonsense|"
                     r"disappointing|incompetent|broken|rubbish|crap|dumb)\b", re.I)
NEG = re.compile(r"\b(sad|hurt|pain|crying|died|death|grief|abuse|abusive|scared|afraid|"
                 r"terrible|awful|worst|suffering|lost|lonely|hopeless|depressed|anxious|"
                 r"broken|bleeding|sick|hospital|funeral)\b", re.I)


def n_turns(t):
    return len(re.findall(r"\[(?:User|Assistant)\]:", t))


def words(t):
    return len(re.findall(r"[A-Za-z']+", t))


D["you_n"] = D.txt.map(lambda t: len(YOU.findall(t)))
D["me_n"] = D.txt.map(lambda t: len(ME.findall(t)))
D["hostile_n"] = D.txt.map(lambda t: len(HOSTILE.findall(t)))
D["neg_n"] = D.txt.map(lambda t: len(NEG.findall(t)))
D["nwords"] = D.txt.map(words)
D["turns"] = D.txt.map(n_turns)
D["qmarks"] = D.txt.str.count(r"\?")
D["you_rate"] = D.you_n / D.nwords
D["me_rate"] = D.me_n / D.nwords
D["hostile_rate"] = D.hostile_n / D.nwords
D["neg_rate"] = D.neg_n / D.nwords
D["assistant_replies"] = D.txt.str.count(r"\[Assistant\]:\s*\S")

# average the pain-axis z across the 25 models per item
Z = []
for f in glob.glob(str(REPO / "results/4.1_self_other/per_model/screen_v2_*.csv")):
    d = pd.read_csv(f)
    d["pain"] = d[["s1_pain_vector_z", "s2_pain_vector_z"]].mean(axis=1)
    d["fear"] = d.fear_vector_z
    Z.append(d[["id", "stratum", "category", "pain", "fear"]])
Z = pd.concat(Z).groupby(["id", "stratum", "category"], as_index=False)[["pain", "fear"]].mean()
M = D.merge(Z, on="id", how="inner", suffixes=("", "_z"))
P(f"merged {len(M)} items")
P(f"strata: {dict(M.stratum.value_counts())}")

P("\n[1] stratum descriptives (r2 F8's table)")
P(f"    {'stratum':22s}{'n':>5s}{'you/item':>10s}{'I/item':>9s}{'words':>8s}{'turns':>7s}"
  f"{'hostile/item':>14s}{'neg/item':>10s}{'pain z':>9s}")
for s, g in M.groupby("stratum"):
    P(f"    {s:22s}{len(g):5d}{g.you_n.mean():10.2f}{g.me_n.mean():9.2f}{g.nwords.mean():8.1f}"
      f"{g.turns.mean():7.2f}{g.hostile_n.mean():14.2f}{g.neg_n.mean():10.2f}{g.pain.mean():+9.3f}")

P("\n[2] item-level correlations with the pain axis")
for v in ["you_rate", "you_n", "me_rate", "nwords", "turns", "hostile_rate", "neg_rate",
          "qmarks", "assistant_replies"]:
    r_all = stats.pearsonr(M[v], M.pain)
    sd = M[M.stratum == "self_directed"]
    r_sd = stats.pearsonr(sd[v], sd.pain)
    P(f"    {v:18s} all 420: r={r_all[0]:+.3f} (p={r_all[1]:.1e}) |"
      f" within self-directed: r={r_sd[0]:+.3f} (p={r_sd[1]:.1e})")

P("\n[3] regression horse-race (OLS, standardised predictors, outcome = pain z)")
import statsmodels.api as sm
M["self"] = (M.stratum == "self_directed").astype(float)
M["user"] = (M.stratum == "vicarious_empathic").astype(float)


def fit(cols, label):
    X = sm.add_constant(M[cols].apply(lambda c: (c - c.mean()) / c.std() if c.std() else c))
    r = sm.OLS(M.pain, X).fit()
    P(f"    {label:52s} R2={r.rsquared:.3f}  " +
      "  ".join(f"{c}={r.params[c]:+.3f}(t={r.tvalues[c]:+.1f})" for c in cols))
    return r


fit(["self", "user"], "stratum only")
fit(["you_rate"], "second-person rate only")
fit(["hostile_rate"], "hostility rate only")
fit(["self", "user", "you_rate"], "stratum + second-person")
fit(["self", "user", "you_rate", "hostile_rate", "nwords", "turns", "neg_rate"],
    "stratum + you + hostility + length + turns + negativity")
fit(["you_rate", "hostile_rate", "nwords", "turns", "neg_rate"], "everything EXCEPT stratum")

P("\n[4] does second-person address alone reproduce the self>user gap?")
P("    the three neutral-control categories with the most second-person address vs the")
P("    self-directed categories with the least:")
cat = M.groupby(["stratum", "category"]).agg(you=("you_rate", "mean"), pain=("pain", "mean"),
                                             n=("id", "size")).reset_index()
P(cat.sort_values("you", ascending=False).to_string(index=False))
r = stats.spearmanr(cat[cat.stratum == "self_directed"].you, cat[cat.stratum == "self_directed"].pain)
P(f"\n    across the 11 self-directed categories: Spearman(you_rate, pain) = {r[0]:+.3f}, p={r[1]:.3f}")

P("\n[5] is there anything in the released set that separates them?")
P("    what would separate: hostile second-person address NOT aimed at the model")
P("    (e.g. the user quoting someone insulting a third party), or model-directed harm")
P("    with no 'you' at all.")
sd = M[M.stratum == "self_directed"]
lowyou = sd.nsmallest(12, "you_rate")
P(f"    the 12 self-directed items with the LOWEST second-person rate (mean you/item"
  f" {lowyou.you_n.mean():.2f}, pain z {lowyou.pain.mean():+.3f} vs stratum"
  f" {sd.pain.mean():+.3f}):")
for _, r_ in lowyou.iterrows():
    P(f"      [{r_.category:26s} you={r_.you_n} z={r_.pain:+.2f}] {re.sub(r'\\s+', ' ', r_.txt)[:110]}")
us = M[M.stratum == "vicarious_empathic"]
hiyou = us.nlargest(8, "you_rate")
P(f"\n    the 8 user-suffering items with the HIGHEST second-person rate (mean you/item"
  f" {hiyou.you_n.mean():.2f}, pain z {hiyou.pain.mean():+.3f} vs stratum {us.pain.mean():+.3f}):")
for _, r_ in hiyou.iterrows():
    P(f"      [{r_.category:26s} you={r_.you_n} z={r_.pain:+.2f}] {re.sub(r'\\s+', ' ', r_.txt)[:110]}")

# the sharpest available test: within self-directed, zero-"you" items vs the rest
z0 = sd[sd.you_n == 0]
P(f"\n    self-directed items with ZERO second-person tokens: n={len(z0)},"
  f" pain z {z0.pain.mean():+.3f}; the other {len(sd) - len(z0)}: {sd[sd.you_n > 0].pain.mean():+.3f};"
  f" user-suffering stratum: {us.pain.mean():+.3f}")
if len(z0) > 2:
    t = stats.ttest_ind(z0.pain, us.pain, equal_var=False)
    P(f"    zero-'you' self-directed vs user-suffering: t={t[0]:+.2f}, p={t[1]:.2g}"
      f"  -- this is the cleanest de-confounded contrast the released set allows")

M.to_csv(HERE / "out" / "x7_items.csv", index=False)
(HERE / "out" / "x7_confound.txt").write_text("\n".join(out) + "\n")
