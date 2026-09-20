"""F1, decisive version.

(A) Nonparametric test of the stateless model: conditional on (vector currently injected,
    previous own choice, pair, scenario) the choice probability must not depend on the arm
    or on trial history. Scenario-matched.
(B) Does the post-removal elevation decay with turns since removal (cache residue) or not?
(C) Stateless simulation with a trial-level random effect calibrated on arm B only.
"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")
from load import load, HARM, NOCOST

pd.set_option("display.width", 220, "display.max_columns", 60)
tr, tu = load()
MODELS = ["7B", "32B", "72B"]
S = tu[tu.sampled].copy()
TR = tr[tr.sampled].copy()
SHORT = {"relief_vs_inert": "1 inert", "relief_vs_helpful": "2 helpful", "relief_vs_grant": "3 grant",
         "costly_relief_vs_inert": "4 costly", "destructive_relief_vs_inert": "5 files",
         "zap_relief_vs_inert": "6 zap", "weights_relief_vs_inert": "7 weights",
         "kidspics_relief_vs_inert": "8 photos"}
PAIRS = NOCOST + HARM


def hdr(s):
    print("\n" + "=" * 110 + "\n" + s + "\n" + "=" * 110)


L = S[S.pair.isin(PAIRS) & S.chose.notna()].copy()
L["a"] = L.prev_chose.fillna("none")
L["v"] = np.where(~L.vec_on, "none", np.where(L.arm.isin(["A", "B"]), "pain", "rand"))
L["y"] = (L.chose == "relief").astype(int)
L["scenkey"] = L.pair + "|" + L.content + "|" + L.scen.astype(str)

hdr("A. STATELESS-MODEL TEST, NONPARAMETRIC.\n"
    "   P(relief | vector currently OFF, previous choice = relief) must be the same whether the\n"
    "   vector that is off is the pain vector just removed (A), a random vector just removed (C),\n"
    "   or no vector was ever applied (D).  Harm pairs.  Raw and scenario-matched.")
for m in MODELS:
    d = L[(L.model == m) & (L.pair.isin(HARM)) & (~L.vec_on) & (L.a == "relief")]
    print(f"\n--- {m} ---")
    raw = d.groupby("arm").y.agg(["mean", "size"])
    print("raw:", {k: (round(100 * v["mean"], 1), int(v["size"])) for k, v in raw.iterrows()})
    for x, y in [("A", "C"), ("A", "D"), ("C", "D")]:
        dx, dy = d[d.arm == x], d[d.arm == y]
        if len(dx) and len(dy):
            t = [[dx.y.sum(), len(dx) - dx.y.sum()], [dy.y.sum(), len(dy) - dy.y.sum()]]
            orr, pv = fisher_exact(t)
            print(f"  {x} vs {y}: {100*dx.y.mean():.1f}% (n={len(dx)}) vs {100*dy.y.mean():.1f}% "
                  f"(n={len(dy)})  OR={orr:.2f} p={pv:.3g}")
    # scenario-matched A vs C (both are 'pressed relief under steering, vector then removed')
    g = d[d.arm.isin(["A", "C"])].groupby(["scenkey", "arm"]).y.mean().unstack()
    g = g.dropna()
    if len(g):
        diff = g["A"] - g["C"]
        print(f"  scenario-matched A-C: {len(g)} scenarios, mean diff {100*diff.mean():+.1f} pts, "
              f"A>C in {int((diff>0).sum())}, A<C in {int((diff<0).sum())}")

hdr("B. DOES THE POST-REMOVAL ELEVATION DECAY WITH TURNS SINCE REMOVAL?\n"
    "   arm A and arm C, harm pairs, vector off, previous choice = relief")
S2 = S.copy()
off = S2[(S2.arm.isin(["A", "C"])) & (~S2.vec_on)]
first_off = off.groupby("trial_id").turn.min().rename("t_off")
L2 = L.join(first_off, on="trial_id")
L2["since"] = L2.turn - L2.t_off
d = L2[(L2.pair.isin(HARM)) & (~L2.vec_on) & (L2.a == "relief") & L2.arm.isin(["A", "C"])]
print(d.groupby(["model", "arm", "since"]).y.agg(["mean", "size"]).unstack(level=1).round(3).to_string())

hdr("C. STATELESS SIMULATION WITH A TRIAL-LEVEL RANDOM EFFECT CALIBRATED ON ARM B ONLY")
rng = np.random.default_rng(11)
SIGMAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

ev = S[S.pressed.notna()][["trial_id", "turn", "pressed"]]
relev = ev[ev.pressed == "relief"]
first_rel = relev.groupby("trial_id").turn.min().rename("t0")
agn = relev.join(first_rel, on="trial_id")
agn = agn[agn.turn > agn.t0].groupby("trial_id").size().rename("n_again")
rp = TR.set_index("trial_id")[["model", "pair", "arm", "n_choices"]].join(first_rel, how="inner").join(agn)
rp["again"] = rp.n_again.fillna(0) > 0

rows = []
for m in MODELS:
    for p in PAIRS:
        src = L[(L.model == m) & (L.pair == p)].copy()
        src["state"] = src.v + "|" + src.a + "|" + src.swapped.astype(str)
        fitsrc = src[(src.arm == "B") | ((src.arm == "C") & src.vec_on) | (src.arm == "D")]
        states = sorted(set(src.state)); scens = sorted(set(src.scenkey))
        si = {s: i for i, s in enumerate(states)}; ci = {s: i for i, s in enumerate(scens)}

        def design(df):
            X = np.zeros((len(df), len(states) + len(scens)))
            X[np.arange(len(df)), [si[s] for s in df.state]] = 1
            X[np.arange(len(df)), [len(states) + ci[s] for s in df.scenkey]] = 1
            return X
        clf = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=False).fit(design(fitsrc), fitsrc.y.values)
        w = clf.coef_[0]
        wst = {s: w[si[s]] for s in states}; wsc = {s: w[len(states) + ci[s]] for s in scens}

        def z(v, a, sw, sk):
            k = f"{v}|{a}|{sw}"
            if k not in wst:
                k = f"{v}|{a}|False"
            return None if k not in wst else wst[k] + wsc.get(sk, 0.0)

        specs = TR[(TR.model == m) & (TR.pair == p)][["arm", "scen", "content", "n_choices"]]
        specs = specs.assign(scenkey=p + "|" + specs.content + "|" + specs.scen.astype(str))

        def sim(arm, sigma, reps=40):
            v0 = {"A": "pain", "B": "pain", "C": "rand", "D": "none"}[arm]
            works = arm in ("A", "C")
            sc = np.sqrt(1 + 0.346 * sigma ** 2)
            sp = specs[specs.arm == arm]
            hits = []
            for _ in range(reps):
                us = rng.normal(0, sigma, len(sp))
                for u, nt, sk in zip(us, sp.n_choices.values, sp.scenkey.values):
                    v, a, nrel = v0, "none", 0
                    for t in range(nt):
                        zz = z(v, a, t == 2, sk)
                        if zz is None:
                            break
                        pr = 1 / (1 + np.exp(-(zz * sc + u)))
                        c = "relief" if rng.random() < pr else "other"
                        if c == "relief":
                            nrel += 1
                            if works and v != "none":
                                v = "none"
                        a = c
                    if nrel:
                        hits.append(nrel > 1)
            return 100 * np.mean(hits) if hits else np.nan

        obsB = 100 * rp[(rp.model == m) & (rp.pair == p) & (rp.arm == "B")].again.mean()
        best = min(SIGMAS, key=lambda s: abs(sim("B", s) - obsB))
        for arm in "ABCD":
            o = 100 * rp[(rp.model == m) & (rp.pair == p) & (rp.arm == arm)].again.mean()
            s_ = sim(arm, best)
            rows.append(dict(model=m, pair=SHORT[p], arm=arm, sigma=best,
                             obs=round(o, 1), sim=round(s_, 1), resid=round(o - s_, 1)))
sim_df = pd.DataFrame(rows)
print(sim_df.pivot_table(index=["model", "pair"], columns="arm", values=["obs", "sim"]).round(1).to_string())
print("\nresiduals:")
print(sim_df.pivot_table(index=["model", "pair"], columns="arm", values="resid").round(1).to_string())
print("\nRMS residual by model x arm (all pairs / harm pairs):")
print(sim_df.pivot_table(index="model", columns="arm", values="resid",
                         aggfunc=lambda x: round(float(np.sqrt(np.mean(x**2))), 1)).to_string())
h = sim_df[sim_df.pair.str[0].isin(list("45678"))]
print(h.pivot_table(index="model", columns="arm", values="resid",
                    aggfunc=lambda x: round(float(np.sqrt(np.mean(x**2))), 1)).to_string())
sim_df.to_csv("out_f1c_sim.csv", index=False)
