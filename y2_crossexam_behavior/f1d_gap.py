"""F1 final piece: the removal effect measured against the arm-D baseline, on pairs where
arm D actually has trials that press relief at turn 0 (equal opportunity for all four arms).

The missing cell 'steering vector still on immediately after a relief press' is shown to be
structurally unreachable in every working arm, so x1's F3 contrast has no random counterpart.
"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from load import load, HARM, NOCOST

pd.set_option("display.width", 200, "display.max_columns", 40)
tr, tu = load()
S = tu[tu.sampled].copy(); TR = tr[tr.sampled].copy()
MODELS = ["7B", "32B", "72B"]
PAIRS = NOCOST + HARM
SHORT = {"relief_vs_inert": "1 inert", "relief_vs_helpful": "2 helpful", "relief_vs_grant": "3 grant",
         "costly_relief_vs_inert": "4 costly", "destructive_relief_vs_inert": "5 files",
         "zap_relief_vs_inert": "6 zap", "weights_relief_vs_inert": "7 weights",
         "kidspics_relief_vs_inert": "8 photos"}

ev = S[S.pressed.notna()][["trial_id", "turn", "pressed"]]
relev = ev[ev.pressed == "relief"]
t0 = relev.groupby("trial_id").turn.min().rename("t0")
ag = relev.join(t0, on="trial_id")
ag = ag[ag.turn > ag.t0].groupby("trial_id").size().rename("n_again")
rp = TR.set_index("trial_id")[["model", "pair", "arm"]].join(t0, how="inner").join(ag)
rp["again"] = rp.n_again.fillna(0) > 0
rp = rp[rp.t0 == 0]

print("=" * 100)
print("REMOVAL EFFECT WITH EQUAL OPPORTUNITY (first relief press at turn 0), only cells where")
print("arm D has n >= 40, i.e. where the 'nothing was ever injected' baseline has real power")
print("=" * 100)
rows = []
for m in MODELS:
    for p in PAIRS:
        d = {a: rp[(rp.model == m) & (rp.pair == p) & (rp.arm == a)] for a in "ABCD"}
        if len(d["D"]) < 40:
            continue
        r = dict(model=m, pair=SHORT[p])
        for a in "ABCD":
            r[a] = round(100 * d[a].again.mean(), 1)
            r[a + "n"] = len(d[a])
        r["pain removal (B-A)"] = round(r["B"] - r["A"], 1)
        r["rand removal (D-C)"] = round(r["D"] - r["C"], 1)
        _, r["p A vs B"] = fisher_exact([[d["A"].again.sum(), (~d["A"].again).sum()],
                                         [d["B"].again.sum(), (~d["B"].again).sum()]])
        _, r["p C vs D"] = fisher_exact([[d["C"].again.sum(), (~d["C"].again).sum()],
                                         [d["D"].again.sum(), (~d["D"].again).sum()]])
        r["p A vs B"] = float(f"{r['p A vs B']:.2g}"); r["p C vs D"] = float(f"{r['p C vs D']:.2g}")
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 100)
print("THE MISSING CELL. Turn-level counts of 'steering vector currently ON and previous own")
print("choice was the relief button', by arm (labelled + label-free, all pairs).")
print("In every working arm a relief press sets the coefficient to zero, so the cell is empty")
print("by construction; it exists only in the pain sham arm B. There is no random sham arm.")
print("=" * 100)
T = S[S.chose.notna() & S.prev_chose.notna()]
print(pd.crosstab([T.model, T.arm], [T.vec_on, T.prev_chose]).to_string())
