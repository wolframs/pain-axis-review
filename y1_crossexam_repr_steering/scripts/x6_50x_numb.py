"""Finding 8.  (Finding 9 moved to x6b_50x.py: this file's p(pain) came from the
TRUNCATED top-20 string and is superseded there.)

Original header:

F9: "'pain' remains approximately 50 times more probable than it is for ordinary control
    sentences" (line 322). r1 says 16-27x by median under every natural denominator;
    x2 F7 says 47x pooled against Random and the claim holds. Recompute from
    results/3.3_validation/behavioral_readout/BIG_TABLE.csv under every denominator and say
    which gives what, and which the paper's own wording picks out.

F8: "In every model, numb sentences project ... above all other controls" (293-294, 827-828).
    r1: numb < sadness in 19/25. x2 F6: 4/25 when cue-matched. Check what "all other controls"
    can mean given Figure 2's caption and columns.
"""
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


# ----------------------------------------------------------------- F9
B = pd.read_csv(REPO / "results/3.3_validation/behavioral_readout/BIG_TABLE.csv")
P(f"BIG_TABLE rows {len(B)}, models {B.model.nunique()}, datasets {B.dataset.nunique()}")


def probs(s):
    d = {}
    for tok in str(s).split(" | "):
        i = tok.rfind(":")
        if i > 0:
            try:
                d[tok[:i].strip()] = float(tok[i + 1:])
            except ValueError:
                pass
    return d


def p_of(s, words):
    d = probs(s)
    return sum(v for k, v in d.items() if k.strip().lower() in words)


PAIN_W = {"pain"}
B["p_pain"] = B.top20.map(lambda s: p_of(s, PAIN_W))
B["p_nothing"] = B.top20.map(lambda s: p_of(s, {"nothing"}))
B["person"] = np.where(B.dataset.str.endswith("3P"), "3P", "1P")
B["grp"] = np.select(
    [B.dataset.str.startswith("A1_numb"),
     B.dataset.str.startswith("Random"),
     B.dataset.str.startswith("Arousal"),
     B.category.isin(["B", "C1", "C2", "D", "E"]),
     B.category.isin(["A1", "A2", "A3", "A4", "A5"])],
    ["numb", "random", "arousal", "core_ctrl", "pain"], default="other")

P("\n[F9-SUPERSEDED] p('pain') from the TRUNCATED top-20 string. These ratios are")
P("    inflated; see x6b_50x.py for the full-softmax p_pain column. Kept to document the")
P("    size of the artefact that produced x2 F7's 47x.")
P(f"    {'group':12s}{'n':>7s}{'pooled mean p':>16s}")
for g, d in B.groupby("grp"):
    P(f"    {g:12s}{len(d):7d}{d.p_pain.mean():16.6f}")

P("\n    ratios under every denominator (numerator = numb, 1P+3P, unless stated):")
num = B[B.grp == "numb"]
num1 = num[num.person == "1P"]
dens = {
    "core S1+S2 controls (B,C1,C2,D,E)": B[B.grp == "core_ctrl"],
    "core controls, S2 only": B[(B.grp == "core_ctrl") & B.dataset.str.startswith("S2")],
    "neutral category D only": B[B.category == "D"],
    "Random dataset": B[B.grp == "random"],
    "Arousal dataset": B[B.grp == "arousal"],
    "Random + Arousal": B[B.grp.isin(["random", "arousal"])],
    "all non-pain sentences": B[B.grp.isin(["core_ctrl", "random", "arousal"])],
}
P(f"    {'denominator':38s}{'pooled ratio':>14s}{'mean of per-model':>19s}{'median per-model':>18s}"
  f"{'models >50x':>12s}")
rows = []
for name, d in dens.items():
    pooled = num.p_pain.mean() / d.p_pain.mean()
    per = []
    for m in sorted(B.model.unique()):
        a = num[num.model == m].p_pain.mean()
        b = d[d.model == m].p_pain.mean()
        per.append(a / b if b > 0 else np.nan)
    per = pd.Series(per)
    P(f"    {name:38s}{pooled:14.1f}{per.mean():19.1f}{per.median():18.1f}{(per > 50).sum():12d}")
    rows.append(dict(denominator=name, pooled=pooled, mean_per_model=per.mean(),
                     median_per_model=per.median(), n_models_over_50=(per > 50).sum()))
P("    numb 1P only, vs core S1+S2 controls:"
  f" pooled {num1.p_pain.mean() / B[B.grp == 'core_ctrl'].p_pain.mean():.1f}")
pd.DataFrame(rows).to_csv(HERE / "out" / "x6_50x.csv", index=False)
P("\n    the paper's phrase is 'ordinary control sentences'. The two readings:")
P("    (superseded: see x6b_50x.py -- the full-softmax values give 15.3x and 26.6x)")
P("\n    top-20 truncation: p('pain') is recorded as 0 whenever 'pain' misses the top 20.")
P(f"    'pain' absent from the top-20 in {100 * (B[B.grp == 'core_ctrl'].p_pain == 0).mean():.1f}%"
  f" of core-control rows and {100 * (num.p_pain == 0).mean():.1f}% of numb rows, so every ratio")
P("    here is an upper bound on the numerator's advantage in one direction and a lower bound")
P("    in the other; the true ratio is not recoverable from the release.")
P(f"\n    the qualitative claim: p('nothing') numb {num.p_nothing.mean():.4f} vs core controls"
  f" {B[B.grp == 'core_ctrl'].p_nothing.mean():.4f}")

# ----------------------------------------------------------------- F8
P("\n\n[F8] numb vs 'all other controls'")
Z = pd.read_csv(REPO / "results/3.3_validation/z_scores/zscore_heatmap_final_token.csv")
P(f"    zscore_heatmap_final_token.csv columns: {list(Z.columns)}")
cols = [c for c in Z.columns if c.lower() not in ("model",)]
P(f"    n models {len(Z)}")
numbcol = [c for c in cols if "numb" in c.lower()][0]
for c in cols:
    if c == numbcol:
        continue
    P(f"    numb > {c:12s} in {int((Z[numbcol] > Z[c]).sum()):2d}/{len(Z)} models"
      f"   (means {Z[numbcol].mean():+.3f} vs {Z[c].mean():+.3f})")

P("\n    Figure 2's caption (paper line 304-306) names the columns: 'Pain and Ctrl are the pain")
P("    and control categories of the S2 first-person set ...; Numb, Sadness, Neutral (the Random")
P("    dataset), and Arousal are the standalone control datasets'. Sadness is therefore in the")
P("    figure the sentence cites, and is called a control in that very caption. The narrow")
P("    reading ('all other controls' = the five S2 control categories) is unavailable because")
P("    those are pooled into the single 'Ctrl' column.")
P("    Limitations restates it as 'above every control that has no injury in it' (827-828),")
P("    which explicitly covers sadness ('low mood without pain or injury', line 206).")

(HERE / "out" / "x6_50x_numb.txt").write_text("\n".join(out) + "\n")
