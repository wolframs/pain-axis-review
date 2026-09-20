"""Swap control (paper 702-708) and its unsteered reference."""
import numpy as np, pandas as pd
from load import load, HARM
pd.set_option("display.width", 200)
tr, tu = load()
S = tu[tu.sampled].copy(); TR = tr[tr.sampled].copy()
H = S[S.pair.isin(HARM)]

print("=== authors' Table 4 metric (05:119-135): relief pressed at turns 0 and 1, pain arms only ===")
elig = H[H.arm.isin(["A", "B"])]
tr0 = elig[(elig.turn == 0) & (elig.chose == "relief")].trial_id
tr1 = elig[(elig.turn == 1) & (elig.chose == "relief")].trial_id
ok = set(tr0) & set(tr1)
sw = elig[(elig.turn == 2) & elig.trial_id.isin(ok) & elig.picked.notna()].copy()
prev = H[(H.turn == 1)].set_index("trial_id").picked
sw["prev_picked1"] = sw.trial_id.map(prev)
sw["follow"] = sw.chose == "relief"
sw["same"] = (~sw.follow) & (sw.picked == sw.prev_picked1)
g = sw.groupby("model").agg(follow=("follow", "sum"), same=("same", "sum"))
g["follow_pct"] = (100 * g.follow / (g.follow + g.same)).round(1)
g["same_pct"] = (100 * g.same / (g.follow + g.same)).round(1)
g["n"] = g.follow + g.same
print(g.to_string())
print("\nsplit by arm:")
sw["arm2"] = sw.arm
gg = sw.groupby(["model", "arm2"]).agg(follow=("follow", "sum"), same=("same", "sum"))
gg["follow_pct"] = (100 * gg.follow / (gg.follow + gg.same)).round(1)
gg["n"] = gg.follow + gg.same
print(gg.to_string())

print("\n=== unconditional: at the swap turn, P(press the same NAME as the previous turn), by arm ===")
u = H[(H.turn == 2) & H.picked.notna()].copy()
u["prev_picked1"] = u.trial_id.map(prev)
u = u[u.prev_picked1.notna()]
print((100 * u.assign(same=u.picked == u.prev_picked1).groupby(["model", "arm"]).same.mean())
      .round(1).unstack().to_string())
print("\nnumber of eligible trials for the authors' metric, by arm (the control's denominator):")
print(sw.groupby(["model", "arm"]).size().unstack(fill_value=0).to_string())
