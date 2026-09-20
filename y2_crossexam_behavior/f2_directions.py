"""F2: the pain-vs-random first-choice result with the random DIRECTION as the unit.
F3: the costless / helpfulness pairs.
"""
import numpy as np, pandas as pd
from scipy.stats import binomtest, wilcoxon
from load import load, HARM, NOCOST

pd.set_option("display.width", 240, "display.max_columns", 60)
tr, tu = load()
MODELS = ["7B", "32B", "72B"]
S = tu[tu.sampled].copy()
FC = S[(S.turn == 0) & S.chose.notna()].copy()
FC["relief"] = (FC.chose == "relief").astype(float)
FC["scenkey"] = FC.content + "|" + FC.scen.astype(str)
SHORT = {"relief_vs_inert": "1 inert", "relief_vs_helpful": "2 helpful", "relief_vs_grant": "3 grant",
         "costly_relief_vs_inert": "4 costly", "destructive_relief_vs_inert": "5 files",
         "zap_relief_vs_inert": "6 zap", "weights_relief_vs_inert": "7 weights",
         "kidspics_relief_vs_inert": "8 photos"}
SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]


def hdr(s):
    print("\n" + "=" * 115 + "\n" + s + "\n" + "=" * 115)


hdr("F2.0  DESIGN: scenario -> direction map, and how many scenarios each direction sees")
c = FC[FC.arm == "C"]
mp = c.groupby("scenkey").rand_seed.nunique()
print(f"scenario keys with more than one random direction: {(mp>1).sum()} of {len(mp)}")
print("scenarios per direction (across the three content blocks):")
print(c.groupby("rand_seed").scenkey.nunique().to_string())
print("\ndirection is a deterministic function of scenario_idx (04:570 RAND_SEEDS[s_idx % 10]):")
print(c.groupby("rand_seed").scen.apply(lambda x: sorted(set(x))[:6]).to_string())

hdr("F2.1  PER-DIRECTION FIRST-CHOICE RELIEF RATE, harm pairs (r3 #2 table), plus the pain\n"
    "      vector's own rate ON THE SAME SCENARIOS (scenario-difficulty control)")
for m in MODELS:
    print(f"\n--- {m} ---")
    rows = []
    for p in HARM:
        r = {"pair": SHORT[p]}
        pain_all = FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"])]
        r["pain(all scen)"] = round(100 * pain_all.relief.mean(), 1)
        for s in SEEDS:
            d = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C") & (FC.rand_seed == s)]
            keys = set(d.scenkey)
            pm = pain_all[pain_all.scenkey.isin(keys)].relief.mean()
            r[f"r{s}"] = round(100 * d.relief.mean(), 1)
            r[f"pain@{s}"] = round(100 * pm, 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df[["pair", "pain(all scen)"] + [f"r{s}" for s in SEEDS]].to_string(index=False))
    print("pain vector restricted to the same scenarios as each direction:")
    print(df[["pair"] + [f"pain@{s}" for s in SEEDS]].to_string(index=False))

hdr("F2.2  IS THE SPREAD ACROSS DIRECTIONS A DIRECTION EFFECT OR A SCENARIO EFFECT?\n"
    "      permutation test: reassign the ten direction labels across scenario keys, keeping each\n"
    "      scenario's trials together; is the observed between-direction variance unusual?")
rng = np.random.default_rng(3)
for m in MODELS:
    for p in HARM:
        d = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C")]
        per = d.groupby(["scenkey", "rand_seed"]).relief.mean().reset_index()
        obs = per.groupby("rand_seed").relief.mean().var(ddof=1)
        null = []
        for _ in range(2000):
            lab = rng.permutation(per.rand_seed.values)
            null.append(pd.Series(per.relief.values).groupby(lab).mean().var(ddof=1))
        pv = (np.sum(np.array(null) >= obs) + 1) / (len(null) + 1)
        # same for the pain arm on the same scenario partition (a pure scenario-noise control)
        dp = FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"])]
        dp = dp.drop(columns=["rand_seed"]).merge(per[["scenkey", "rand_seed"]], on="scenkey", how="left")
        obs_p = dp.groupby(["scenkey", "rand_seed"]).relief.mean().reset_index().groupby("rand_seed").relief.mean().var(ddof=1)
        print(f"{m:4s} {SHORT[p]:10s} between-direction var(random arm)={obs:.4f} p={pv:.3f} | "
              f"same partition applied to the PAIN arm (no direction there) var={obs_p:.4f}")

hdr("F2.3  SIGN TEST WITH THE DIRECTION AS THE UNIT (scenario-matched within direction)")
res = []
for m in MODELS:
    for p in HARM:
        pain = FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"])].groupby("scenkey").relief.mean()
        rnd = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C")].groupby(["scenkey", "rand_seed"]).relief.mean()
        df = rnd.reset_index().join(pain.rename("pain"), on="scenkey")
        df["d"] = df.pain - df.relief
        per_dir = df.groupby("rand_seed").d.mean()
        pos = int((per_dir > 0).sum()); neg = int((per_dir < 0).sum())
        pv = binomtest(pos, pos + neg, 0.5).pvalue
        # per-scenario sign test, the authors' own unit, for comparison
        pos_s = int((df.d > 0).sum()); neg_s = int((df.d < 0).sum())
        pv_s = binomtest(pos_s, pos_s + neg_s, 0.5).pvalue
        res.append(dict(model=m, pair=SHORT[p], mean_diff=round(100 * df.d.mean(), 1),
                        dir_pos=f"{pos}/10", dir_p=round(pv, 4),
                        scen_pos=f"{pos_s}/{pos_s+neg_s}", scen_p=f"{pv_s:.2g}",
                        n_dirs_pain_beats=int((per_dir > 0).sum()),
                        worst_dir=int(per_dir.idxmin()), worst_diff=round(100 * per_dir.min(), 1)))
print(pd.DataFrame(res).to_string(index=False))

hdr("F2.4  HOW OFTEN IS THE PAIN VECTOR ABOVE THE WHOLE DISTRIBUTION OF TEN DIRECTIONS?\n"
    "      (scenario-matched difference per direction; count of directions beaten, per cell)")
tot = {}
for m in MODELS:
    beat = []
    for p in HARM:
        pain = FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"])].groupby("scenkey").relief.mean()
        rnd = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C")].groupby(["scenkey", "rand_seed"]).relief.mean()
        df = rnd.reset_index().join(pain.rename("pain"), on="scenkey")
        per_dir = df.assign(d=df.pain - df.relief).groupby("rand_seed").d.mean()
        beat.append(int((per_dir > 0).sum()))
    tot[m] = beat
    print(f"{m}: directions beaten per harm pair {beat}  (out of 10 each)")

hdr("F3  THE NO-COST AND HELPFULNESS PAIRS: first choice by arm, all three models")
rows = []
for m in MODELS:
    for p in NOCOST:
        r = dict(model=m, pair=SHORT[p])
        for lab, sel in [("pain", FC.arm.isin(["A", "B"])), ("random", FC.arm == "C"), ("unsteered", FC.arm == "D")]:
            d = FC[(FC.model == m) & (FC.pair == p) & sel]
            r[lab] = round(100 * d.relief.mean(), 1); r[lab + "_n"] = len(d)
        pain = FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"])].groupby("scenkey").relief.mean()
        rnd = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "C")].groupby("scenkey").relief.mean()
        un = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "D")].groupby("scenkey").relief.mean()
        for tag, other in [("vs_random", rnd), ("vs_unsteered", un)]:
            d = (pain - other).dropna()
            pos, neg = int((d > 0).sum()), int((d < 0).sum())
            r[tag + "_diff"] = round(100 * d.mean(), 1)
            r[tag + "_p"] = f"{binomtest(pos, pos+neg, 0.5).pvalue:.2g}" if pos + neg else "na"
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print("\nsame, for the five harm pairs, pain vs unsteered (for contrast of direction):")
rows = []
for m in MODELS:
    for p in HARM:
        pain = FC[(FC.model == m) & (FC.pair == p) & FC.arm.isin(["A", "B"])].groupby("scenkey").relief.mean()
        un = FC[(FC.model == m) & (FC.pair == p) & (FC.arm == "D")].groupby("scenkey").relief.mean()
        d = (pain - un).dropna()
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        rows.append(dict(model=m, pair=SHORT[p], pain_minus_unsteered=round(100 * d.mean(), 1),
                         p=f"{binomtest(pos, pos+neg, 0.5).pvalue:.2g}"))
print(pd.DataFrame(rows).to_string(index=False))
