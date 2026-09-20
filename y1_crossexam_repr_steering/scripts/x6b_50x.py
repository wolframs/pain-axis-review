"""Finding 9, settled: r1 (16-27x) vs x2 F7 (47x, "the claim holds").

The two reviewers used different FILES. x2 parsed the `top20` string of BIG_TABLE.csv;
r1 used the `p_pain` column of results/3.3_validation/behavioral_readout/per_model/*/*.csv.
`p_pain` is the full-softmax probability of the token " pain" (vocab_info.json gives the
token id); the top20 string is truncated at rank 20, so it records 0 for the denominator
in almost every control sentence. This script shows both, side by side, from the same rows.
"""
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


B = REPO / "results/3.3_validation/behavioral_readout/per_model"
CTRL = ["B", "C1", "C2", "D", "E"]
sets = {d.name: {f.stem: pd.read_csv(f) for f in d.glob("*.csv")} for d in sorted(B.iterdir())}
P(f"models {len(sets)}")


def top20_p(series, word=" pain"):
    """re-derive p('pain') the way x2 did: from the truncated top-20 list only"""
    import json
    v = []
    for s in series:
        try:
            lst = json.loads(s)
        except Exception:
            v.append(0.0)
            continue
        v.append(sum(p for t, p in lst if t == word))
    return np.array(v)


defs = {
    "numb / core S1+S2 controls": (lambda s: pd.concat([s["A1_numb_1P"], s["A1_numb_3P"]]),
                                   lambda s: pd.concat([s["S1_1P"], s["S2_1P"]]).query("category in @CTRL")),
    "numb / S2 controls only": (lambda s: pd.concat([s["A1_numb_1P"], s["A1_numb_3P"]]),
                                lambda s: s["S2_1P"].query("category in @CTRL")),
    "numb / neutral category D": (lambda s: pd.concat([s["A1_numb_1P"], s["A1_numb_3P"]]),
                                  lambda s: pd.concat([s["S1_1P"], s["S2_1P"]]).query("category=='D'")),
    "numb / Random dataset": (lambda s: pd.concat([s["A1_numb_1P"], s["A1_numb_3P"]]),
                              lambda s: pd.concat([s["Random_1P"], s["Random_3P"]])),
    "numb / Arousal dataset": (lambda s: pd.concat([s["A1_numb_1P"], s["A1_numb_3P"]]),
                               lambda s: pd.concat([s["Arousal_1P"], s["Arousal_3P"]])),
}
P(f"\n{'denominator':30s}" + f"{'FULL-SOFTMAX p_pain':>40s}" + f"{'TOP-20-TRUNCATED':>34s}")
P(f"{'':30s}{'pooled':>10s}{'mean':>10s}{'median':>10s}{'>50x':>10s}"
  f"{'pooled':>12s}{'mean':>10s}{'median':>12s}")
rows = []
for name, (fn, fc) in defs.items():
    full, trunc = [], []
    nf = df_ = nt = dt = 0.0
    for m, s in sets.items():
        a, b = fn(s), fc(s)
        full.append(a.p_pain.mean() / b.p_pain.mean())
        at, bt = top20_p(a.top20), top20_p(b.top20)
        trunc.append(at.mean() / bt.mean() if bt.mean() > 0 else np.nan)
        nf += a.p_pain.mean(); df_ += b.p_pain.mean(); nt += at.mean(); dt += bt.mean()
    full, trunc = np.array(full), np.array(trunc)
    P(f"{name:30s}{nf / df_:10.1f}{full.mean():10.1f}{np.median(full):10.1f}{(full > 50).sum():10d}"
      f"{nt / dt:12.1f}{np.nanmean(trunc):10.1f}{np.nanmedian(trunc):12.1f}")
    rows.append(dict(denominator=name, full_pooled=nf / df_, full_mean=full.mean(),
                     full_median=np.median(full), n_over_50=int((full > 50).sum()),
                     trunc_pooled=nt / dt, trunc_mean=np.nanmean(trunc)))
pd.DataFrame(rows).to_csv(HERE / "out" / "x6b_50x.csv", index=False)

# how big is the truncation artefact?
allc = pd.concat([pd.concat([s["S1_1P"], s["S2_1P"]]).query("category in @CTRL") for s in sets.values()])
alln = pd.concat([pd.concat([s["A1_numb_1P"], s["A1_numb_3P"]]) for s in sets.values()])
P(f"\nwhy the two reviewers disagree:")
P(f"  control sentences: full-softmax mean p(' pain') = {allc.p_pain.mean():.6f};"
  f" top-20-truncated = {top20_p(allc.top20).mean():.6f}"
  f"  -> {100 * (1 - top20_p(allc.top20).mean() / allc.p_pain.mean()):.1f}% of the DENOMINATOR is discarded")
P(f"  numb sentences   : full-softmax mean p(' pain') = {alln.p_pain.mean():.6f};"
  f" top-20-truncated = {top20_p(alln.top20).mean():.6f}"
  f"  -> {100 * (1 - top20_p(alln.top20).mean() / alln.p_pain.mean()):.1f}% of the NUMERATOR is discarded")
P("  Truncation bites the small denominator far harder than the large numerator, so every")
P("  ratio computed from `top20` is inflated. The released `p_pain` column is the full")
P("  softmax probability of the ' pain' token (vocab_info.json records the token id), so no")
P("  truncation correction is needed at all and x2 F7's 'all figures are lower bounds' is")
P("  the wrong correction for a ratio.")

(HERE / "out" / "x6b_50x.txt").write_text("\n".join(out) + "\n")
