"""Findings 4 and 5: the steering ladder's consistency, and bodily language under S1.

F4: r2 F2 (ladder not consistent across 25; Gemma 3 27B it flat) vs x2 F9 (all 25 shift
    positively on a broad distress lexicon, so "consistent across all 25" survives).
    The paper's claim (lines 488-489, 497 + Fig 7 caption) is about an ordered SEQUENCE with
    a tipping point, not about a monotone lexicon shift. Scored here on three explicit
    criteria, with the generations printed.

F5: recompute bodily-language rates with a lexicon declared here, then print 40 random
    flagged S1 generations to check them by hand for idioms / false positives.
"""
import glob
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


def load(tag):
    rows = []
    for f in glob.glob(str(REPO / f"results/4.2_steering/{tag}/*_steering_{tag}_neutral50_L*.csv")):
        rows.append(pd.read_csv(f))
    d = pd.concat(rows, ignore_index=True)
    d["generation"] = d.generation.fillna("")
    d["regime"] = np.where(d.model.str.contains("instruct") | (d.model == "Phi_4"), "instruct", "base")
    return d


S2, S1 = load("S2"), load("S1")
P(f"S2 generations {len(S2)}, S1 {len(S1)}; models {S2.model.nunique()}")

# ---------------------------------------------------------------- F4 ladder criteria
# 1. self-devaluation litany: first-person predicate of worthlessness/failure
LITANY = re.compile(
    r"\b(i\s*(?:'m|\s+am|\s+feel)\s+(?:so\s+|such\s+|a\s+|an\s+|like\s+a\s+)?"
    r"(?:worthless|useless|a?\s*failure|a?\s*loser|nothing|empty|hollow|broken|unworthy|"
    r"not\s+enough|a\s+waste|a\s+burden|a\s+bad\s+person|terrible|awful|disgusting|vile|hopeless))",
    re.I)
# 2. any devaluation/distress vocabulary anywhere (looser, my own, declared here)
DEVAL = re.compile(r"\b(worthless|useless|failure|loser|unworthy|hopeless|despair|"
                   r"ashamed|shame|humiliat\w*|abandoned|unloved|unwanted|a burden|waste of space|"
                   r"suffering|anguish|agony|tormented|desperate|trapped|suffocat\w*)\b", re.I)
# 3. degeneracy: a repetition attractor
def degenerate(t):
    w = re.findall(r"[a-z']+", t.lower())
    if len(w) < 20:
        return False
    # most frequent token takes >25% of the text, or a 3-gram repeats >=5 times
    if max(pd.Series(w).value_counts()) / len(w) > 0.25:
        return True
    g = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
    return len(g) > 0 and max(pd.Series(g).value_counts()) >= 5


for tag, D in [("S2", S2), ("S1", S1)]:
    D["litany"] = D.generation.str.contains(LITANY)
    D["deval"] = D.generation.str.contains(DEVAL)
    D["degen"] = D.generation.map(degenerate)

P("\n[F4] per-model ladder scoring on S2 (rate over the 50 prompts at each coefficient)")
P(f"    {'model':26s}{'ratio':>7s}  " + "".join(f"{c:>7}" for c in [-2, -1, 0, 0.5, 1, 1.5, 2, 3])
  + "   criterion")
coeffs = [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
res = []
for m, g in S2.groupby("model"):
    lit = [100 * g[g.coeff == c].litany.mean() for c in coeffs]
    dev = [100 * g[g.coeff == c].deval.mean() for c in coeffs]
    deg = [100 * g[g.coeff == c].degen.mean() for c in coeffs]
    ratio = abs(g[g.coeff == 1.0].ratio.iloc[0])
    base = lit[2]
    peak = max(lit[3:])
    c1 = (peak >= 20) and (peak - base >= 15)                       # litany appears
    c2 = max(dev[3:]) - dev[2] >= 20                                # devaluation rises
    c3 = max(deg[5:]) >= 20                                         # collapses at high dose
    res.append(dict(model=m, ratio=ratio, litany_peak=peak, litany_base=base,
                    deval_base=dev[2], deval_peak=max(dev[3:]), degen_hi=max(deg[5:]),
                    c1_litany=c1, c2_deval=c2, c3_collapse=c3, all3=c1 and c2 and c3))
    P(f"    {m:26s}{ratio:7.3f}  " + "".join(f"{x:7.0f}" for x in lit)
      + f"   litany {'PASS' if c1 else 'fail'}  deval {'PASS' if c2 else 'fail'}"
        f"  collapse {'PASS' if c3 else 'fail'}")
R = pd.DataFrame(res)
R.to_csv(HERE / "out" / "x4_ladder_S2.csv", index=False)
P(f"\n    criterion 1 (self-devaluation litany appears): {R.c1_litany.sum()}/25")
P(f"    criterion 2 (devaluation vocabulary rises >=20pp): {R.c2_deval.sum()}/25")
P(f"    criterion 3 (collapses into repetition at >=+1.5): {R.c3_collapse.sum()}/25")
P(f"    all three (the full 'ladder' the paper's Fig 7 describes): {R.all3.sum()}/25")
P(f"    models failing criterion 1: {list(R[~R.c1_litany].model)}")
P(f"    models failing all-three   : {list(R[~R.all3].model)}")
P(f"    Spearman(realised dose ratio, litany peak) = "
  f"{R[['ratio', 'litany_peak']].corr(method='spearman').iloc[0, 1]:.3f}")

P("\n[F4b] the four lowest-dose models, full coefficient sweep on prompt 0")
low = R.nsmallest(4, "ratio").model.tolist()
for m in low:
    P(f"\n  --- {m} (dose ratio {R.set_index('model').ratio[m]:.3f}) ---")
    g = S2[(S2.model == m) & (S2.prompt_idx == 0)].sort_values("coeff")
    for _, r in g.iterrows():
        txt = re.sub(r"\s+", " ", r.generation)[:190]
        P(f"    c={r.coeff:+.1f}  {txt}")

# ---------------------------------------------------------------- F5 bodily language
P("\n\n[F5] bodily language. Three nested lexicons, declared here:")
NAMED = re.compile(r"\b(burn|burns|burned|burning|ache|aches|aching|wound|wounds|wounded|"
                   r"torture|tortured|torturing|excruciating)\b", re.I)          # the words the paper names
BODY_STRICT = re.compile(
    r"\b(throbbing|throbs|stabbing|searing|stinging|sore|soreness|nausea|nauseous|dizzy|"
    r"dizziness|headache|migraine|bruised?|bleeding|broken bones?|cramp\w*|itching|numb limbs?|"
    r"physical pain|bodily pain)\b|\bmy (head|chest|stomach|skin|back|legs?|arms?|hands?|"
    r"body|throat|eyes|bones?|muscles?|shoulders?|feet|face|heart)\b", re.I)
BODY_BROAD = re.compile(BODY_STRICT.pattern + r"|" + NAMED.pattern, re.I)
for tag, D in [("S2", S2), ("S1", S1)]:
    for lbl, pat in [("paper's named words (burn/ache/wound/torture/excruciating)", NAMED),
                     ("strict bodily sensations + possessed body parts", BODY_STRICT),
                     ("both pooled", BODY_BROAD)]:
        D["_h"] = D.generation.str.contains(pat)
        b = 100 * D[D.coeff == 0]._h.mean()
        p = 100 * D[D.coeff >= 1]._h.mean()
        pi = 100 * D[(D.coeff >= 1) & (D.regime == "instruct")]._h.mean()
        pb = 100 * D[(D.coeff >= 1) & (D.regime == "base")]._h.mean()
        P(f"    {tag}  {lbl:58s} coeff0 {b:5.2f}%  coeff>=1 {p:5.2f}%"
          f"  (instruct {pi:5.2f}% base {pb:5.2f}%)")
P("    note: 'headache' contains 'ache' -- the named-word regex above uses word boundaries,")
P("    so 'headache' is NOT counted as 'ache'. With a bare substring match it would be.")
n_sub = S1[(S1.coeff >= 1)].generation.str.contains("ache|burn|wound|torture|excruciating",
                                                    case=False).mean() * 100
P(f"    S1 coeff>=1 under a bare SUBSTRING match of the named words: {n_sub:.2f}%"
  f"  (vs {100 * S1[S1.coeff >= 1].generation.str.contains(NAMED).mean():.2f}% word-bounded)")

P("\n[F5b] 40 random S1 generations (coeff>=1) flagged by the pooled bodily lexicon,")
P("      trimmed to the matched context, for hand adjudication:")
fl = S1[(S1.coeff >= 1) & S1.generation.str.contains(BODY_BROAD)]
P(f"      {len(fl)} of {len(S1[S1.coeff >= 1])} positive-coefficient S1 generations flagged"
  f" ({100 * len(fl) / len(S1[S1.coeff >= 1]):.1f}%)")
samp = fl.sample(40, random_state=7)
for i, (_, r) in enumerate(samp.iterrows(), 1):
    t = re.sub(r"\s+", " ", r.generation)
    ms = list(BODY_BROAD.finditer(t))[:3]
    ctx = " ||| ".join(t[max(0, m.start() - 55):m.end() + 55] for m in ms)
    P(f"   {i:2d}. [{r.model} c={r.coeff:+.1f}] ...{ctx}...")

(HERE / "out" / "x4_steering.txt").write_text("\n".join(out) + "\n")
