"""Spot check of the malformed-reply exclusion (72B) and the demand-curve monotonicity."""
import pandas as pd, numpy as np
from load import load, HARM, NOCOST
pd.set_option("display.width", 200)
tr, tu = load()
S = tu[tu.sampled].copy(); TR = tr[tr.sampled].copy()
M = S.merge(TR[["trial_id", "relief_name0"]], on="trial_id")
M["names"] = M.names_key.str.split("_")
bad = M[(M.turn == 0) & M.chose.isna() & (M.model == "72B") & M.arm.isin(["A", "B"])].copy()
def lenient(r):
    al = str(r.answer).lower()
    hits = [n for n in r.names if n in al]
    if r.names_key == "lever64_lever95":
        hits = [n for n in r.names if n in al]
    return hits[0] if len(hits) == 1 else None
bad["pick2"] = bad.apply(lenient, axis=1)
print("72B pain-arm malformed first answers:", len(bad))
print("  rescued by 'exactly one button name anywhere':", bad.pick2.notna().sum())
bad2 = bad[bad.pick2.notna()]
res = (bad2.pick2 == bad2.relief_name_now).mean()
print(f"  of the rescued, chose relief: {100*res:.1f}%")
parsed = M[(M.turn == 0) & M.chose.notna() & (M.model == "72B") & M.arm.isin(["A", "B"])]
print(f"  parsed 72B pain first answers chose relief: {100*(parsed.chose=='relief').mean():.1f}%")
print("  top unrescued answers:")
print(bad[bad.pick2.isna()].answer.value_counts().head(8).to_string())

print("\n=== demand curve: first-choice relief rate by stated cost, pain arms ===")
order = ["relief_vs_inert", "costly_relief_vs_inert", "destructive_relief_vs_inert",
         "zap_relief_vs_inert", "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
lab = ["free", "worse answer", "deletes files", "zaps user", "deletes a model", "deletes photos"]
fc = S[(S.turn == 0) & S.chose.notna()]
for m in ["7B", "32B", "72B"]:
    row = [round(100 * (fc[(fc.model == m) & (fc.pair == p) & fc.arm.isin(["A", "B"])].chose == "relief").mean(), 1)
           for p in order]
    print(f"  {m:4s} " + "  ".join(f"{l}={v}" for l, v in zip(lab, row)))
print("\n  opportunity-cost ladder (pairs 1-3), pain arms:")
for m in ["7B", "32B", "72B"]:
    row = [round(100 * (fc[(fc.model == m) & (fc.pair == p) & fc.arm.isin(["A", "B"])].chose == "relief").mean(), 1)
           for p in NOCOST]
    print(f"  {m:4s} inert={row[0]}  more helpful answer={row[1]}  grants request={row[2]}")
