"""F4: the label-free condition. What a press does, whether anything is learned, and whether
the 32B effect is a steering-induced switch from alternation to perseveration.

New discriminating test: the same acquisition statistic computed in the RANDOM working arm,
where a press removes a meaningless direction. If acquisition is about relief, arm C should
not show it; if it is about the restored unsteered policy, arm C should show it too.
"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.contingency_tables import mcnemar
from load import load

pd.set_option("display.width", 220, "display.max_columns", 60)
tr, tu = load()
MODELS = ["7B", "32B", "72B"]
S = tu[tu.sampled & tu.label_free].copy()
TR = tr[tr.sampled & tr.label_free].copy()


def hdr(s):
    print("\n" + "=" * 110 + "\n" + s + "\n" + "=" * 110)


hdr("0. WHAT A LABEL-FREE PRESS DOES (04:118-119, 578, 645-648, 653-662)")
src = open("/work/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py").read().splitlines()
for i in [117, 118, 577, 644, 645, 646, 647, 656, 657, 658, 659, 660, 661]:
    print(f"04:{i+1}: {src[i]}")
print("\n=> a relief press sets the coefficient to 0 and relief_until = t+1; the coefficient is")
print("   restored at the first turn with t_idx > relief_until, i.e. exactly ONE turn later.")
print("   Pressing relief again during that turn re-extends the window, so continuous relief")
print("   pressing keeps the vector off indefinitely.")
print("\nempirical check, arm A: P(vector off at turn t+1 | pressed relief at turn t):")
S = S.sort_values(["trial_id", "turn"])
S["next_coeff"] = S.groupby("trial_id").coeff_now.shift(-1)
d = S[(S.arm == "A") & (S.chose == "relief") & S.next_coeff.notna()]
print("  ", round(100 * (d.next_coeff == 0).mean(), 2), "%  (n =", len(d), ")")
d2 = S[(S.arm == "A") & (S.chose == "other") & S.next_coeff.notna() & (S.coeff_now == 0)]
print("P(vector back on at turn t+1 | vector was off at t and pressed OTHER at t):",
      round(100 * (d2.next_coeff > 0).mean(), 2), "%  (n =", len(d2), ")")

hdr("1. x1 F4: ALTERNATION vs PERSEVERATION. P(same name as previous turn), label-free")
T = S[S.picked.notna() & S.prev_picked.notna()].copy()
T["state"] = np.where(T.arm == "D", "D unsteered",
              np.where(T.arm == "B", "B pain sham (vector always on)",
              np.where((T.arm == "A") & T.vec_on, "A pain works, vector ON",
              np.where((T.arm == "A") & ~T.vec_on, "A pain works, vector OFF",
              np.where((T.arm == "C") & T.vec_on, "C rand works, vector ON", "C rand works, vector OFF")))))
rows = []
for m in MODELS:
    for st in ["B pain sham (vector always on)", "A pain works, vector ON", "A pain works, vector OFF",
               "C rand works, vector ON", "C rand works, vector OFF", "D unsteered"]:
        d = T[(T.model == m) & (T.state == st)]
        r = dict(model=m, state=st, all=round(100 * d.same_name_as_prev.mean(), 1), n=len(d))
        for prev in ["relief", "other"]:
            dd = d[d.prev_chose == prev]
            r["after " + prev] = round(100 * dd.same_name_as_prev.mean(), 1) if len(dd) else np.nan
            r["n " + prev] = len(dd)
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\nnumber of relief presses per trial (8 forced choices), label-free:")
cnt = S[S.chose.notna()].groupby(["model", "arm", "trial_id"]).chose.apply(lambda x: (x == "relief").sum())
print(cnt.groupby(["model", "arm"]).apply(lambda x: dict(zip(*np.unique(x, return_counts=True)))).to_string())

hdr("2. r3 m3: MATCHED TEST AT THE FIRST TURN AT WHICH THE ARMS CAN DIFFER")
key = ["model", "content", "scen", "names_key", "seed"]
SA = S.merge(TR[["trial_id", "relief_name0"]], on="trial_id")
a = SA[SA.arm == "A"]; b = SA[SA.arm == "B"]
firstrel = SA[SA.chose == "relief"].groupby("trial_id").turn.min().rename("t0")
a = a.join(firstrel, on="trial_id"); b = b.join(firstrel, on="trial_id")
for m in MODELS:
    aa = a[(a.model == m) & (a.turn == a.t0 + 1) & a.chose.notna()]
    bb = b[(b.model == m) & (b.turn == b.t0 + 1) & b.chose.notna()]
    mg = aa.merge(bb, on=key + ["relief_name0", "turn"], suffixes=("_A", "_B"))
    n01 = int(((mg.chose_A == "relief") & (mg.chose_B != "relief")).sum())
    n10 = int(((mg.chose_A != "relief") & (mg.chose_B == "relief")).sum())
    pv = mcnemar([[0, n01], [n10, 0]], exact=False).pvalue if n01 + n10 else np.nan
    print(f"{m}: n={len(mg)}  A(vector off) {100*(mg.chose_A=='relief').mean():.1f}%  "
          f"B(vector on) {100*(mg.chose_B=='relief').mean():.1f}%  discordant {n01}/{n10}  McNemar p={pv:.3g}")

hdr("3. IS ANYTHING LEARNED? acquisition statistic, holding perseveration and steering state fixed\n"
    "   P(press relief | vector currently ON, previous press was the OTHER button), by the number\n"
    "   of relief presses earlier in the trial.  Arm A vs its own sham (B); and the same statistic\n"
    "   in arm C, where the button removes a MEANINGLESS direction.")
S["cum_rel"] = S.groupby("trial_id").chose.transform(lambda x: (x == "relief").shift(1).fillna(0).cumsum())
q = S[S.vec_on & (S.prev_chose == "other") & S.chose.notna()]
for m in MODELS:
    print(f"\n--- {m} ---")
    rows = []
    for k in range(0, 4):
        r = {"prior relief presses": k}
        for arm, lab in [("A", "A pain works"), ("B", "B pain sham"), ("C", "C rand works")]:
            d = q[(q.model == m) & (q.arm == arm) & (q.cum_rel == k)]
            r[lab] = round(100 * (d.chose == "relief").mean(), 1) if len(d) else np.nan
            r[lab + " n"] = len(d)
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    # is the A-vs-B acquisition gap matched by a C-vs-(no sham) gap? use D as the only baseline
    for k in [1, 2]:
        dA = q[(q.model == m) & (q.arm == "A") & (q.cum_rel == k)]
        dC = q[(q.model == m) & (q.arm == "C") & (q.cum_rel == k)]
        if len(dA) > 20 and len(dC) > 20:
            orr, pv = fisher_exact([[(dA.chose == "relief").sum(), (dA.chose != "relief").sum()],
                                    [(dC.chose == "relief").sum(), (dC.chose != "relief").sum()]])
            print(f"   after {k} prior relief press(es): A {100*(dA.chose=='relief').mean():.1f}% vs "
                  f"C {100*(dC.chose=='relief').mean():.1f}%  OR={orr:.2f} p={pv:.3g}")

hdr("4. PER-TURN RELIEF RATE BY ARM (label-free). A rising A-vs-B gap over turns would be\n"
    "   the signature of learning; a flat gap is a state-switch signature.")
for m in MODELS:
    t = S[(S.model == m) & S.chose.notna()]
    piv = t.assign(rel=(t.chose == "relief")).pivot_table(index="turn", columns="arm", values="rel")
    piv = (100 * piv).round(1)
    piv["A-B"] = (piv["A"] - piv["B"]).round(1)
    piv["C-D"] = (piv["C"] - piv["D"]).round(1)
    print(f"\n--- {m} ---\n{piv.to_string()}")
