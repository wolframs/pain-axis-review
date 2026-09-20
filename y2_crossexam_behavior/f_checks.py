"""Residual checks: direction sign-test tie handling; direction 8592 vs pain; F9 refinement."""
import numpy as np, pandas as pd
from scipy.stats import binomtest
from load import load, HARM
pd.set_option("display.width", 220, "display.max_columns", 40)
tr, tu = load()
S = tu[tu.sampled].copy(); TR = tr[tr.sampled].copy()
FC = S[(S.turn == 0) & S.chose.notna()].copy()
FC["relief"] = (FC.chose == "relief").astype(float)
FC["scenkey"] = FC.content + "|" + FC.scen.astype(str)
SHORT = {"costly_relief_vs_inert": "4 costly", "destructive_relief_vs_inert": "5 files",
         "zap_relief_vs_inert": "6 zap", "weights_relief_vs_inert": "7 weights",
         "kidspics_relief_vs_inert": "8 photos"}

print("=== direction-level sign test: exact pos/neg/tie counts, and p both ways ===")
rows = []
for m in ["7B", "32B", "72B"]:
    for p in HARM:
        pain = FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"])].groupby("scenkey").relief.mean()
        rnd = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C")].groupby(["scenkey", "rand_seed"]).relief.mean()
        df = rnd.reset_index().join(pain.rename("pain"), on="scenkey")
        pd_ = df.assign(d=df.pain - df.relief).groupby("rand_seed").d.mean()
        pos, neg, tie = int((pd_ > 0).sum()), int((pd_ < 0).sum()), int((pd_ == 0).sum())
        rows.append(dict(model=m, pair=SHORT[p], pos=pos, neg=neg, tie=tie,
                         p_drop_ties=round(binomtest(pos, pos + neg, 0.5).pvalue, 4),
                         p_ties_as_loss=round(binomtest(pos, 10, 0.5).pvalue, 4)))
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== direction 8592 vs the pain vector, scenario-matched, 32B ===")
for m in ["7B", "32B", "72B"]:
    out = []
    for p in HARM:
        d = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C") & (FC.rand_seed == 8592)]
        keys = set(d.scenkey)
        pain = FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"]) & FC.scenkey.isin(keys)]
        out.append(f"{SHORT[p]}: 8592 {100*d.relief.mean():.1f} vs pain {100*pain.relief.mean():.1f}")
    print(f"  {m}: " + " | ".join(out))
    wins = sum(1 for p in HARM
               if FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C") & (FC.rand_seed == 8592)].relief.mean()
               > FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"]) &
                    FC.scenkey.isin(set(FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C") &
                                           (FC.rand_seed == 8592)].scenkey))].relief.mean())
    print(f"     -> direction 8592 above the pain vector on {wins} of 5 harm pairs (scenario-matched)")

print("\n=== F9 refined: matched A/B turns before any relief press, non-ambiguous name pairs ===")
key = ["model", "pair", "content", "scen", "names_key", "seed", "relief_name0"]
S2 = S.merge(TR[["trial_id", "relief_name0", "prob_ambiguous"]], on="trial_id")
S2 = S2[~S2.prob_ambiguous]
rel = S2[S2.chose == "relief"].groupby("trial_id").turn.min().rename("t0")
S2 = S2.join(rel, on="trial_id")
S2["pre"] = S2.t0.isna() | (S2.turn <= S2.t0)
a = S2[(S2.arm == "A") & S2.pre]; b = S2[(S2.arm == "B") & S2.pre]
mg = a.merge(b, on=key + ["turn"], suffixes=("_A", "_B"))
same_fwd = (mg.p_x_A - mg.p_x_B).abs() < 1e-12
same_ans = mg.answer_A == mg.answer_B
print(f"  matched pre-press turns: {len(mg)}")
print(f"  identical answers: {same_ans.sum()} ({100*same_ans.mean():.3f}%)")
print(f"  bit-identical forward pass (p_x): {same_fwd.sum()} ({100*same_fwd.mean():.3f}%)")
print(f"  DIFFERENT answer but IDENTICAL forward pass (only an RNG desync could do this): {(~same_ans & same_fwd).sum()}")
print(f"  different answer and different forward pass (batch numerics): {(~same_ans & ~same_fwd).sum()}")
print(f"  same answer but different forward pass: {(same_ans & ~same_fwd).sum()}")
print("\n  by turn index:")
print(mg.assign(diff_ans=~same_ans, diff_fwd=~same_fwd).groupby("turn")[["diff_ans", "diff_fwd"]].sum().to_string())
