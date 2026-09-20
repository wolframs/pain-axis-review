"""F1 continued.

(i) The unsteered arm's repeat-press rate under the SAME opportunity restriction that r3
    used for arms A/B/C -- including the no-cost pairs, where the unsteered arm has power.
(ii) A stateless policy model with scenario heterogeneity, fitted out of sample, simulated
     forward under each arm's own removal rule, scored on the authors' repeat metric.
"""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from load import load, HARM, NOCOST

pd.set_option("display.width", 220, "display.max_columns", 60)
tr, tu = load()
MODELS = ["7B", "32B", "72B"]
S = tu[tu.sampled].copy()
TR = tr[tr.sampled].copy()
PAIRS = NOCOST + HARM
SHORT = {"relief_vs_inert": "1 inert", "relief_vs_helpful": "2 helpful", "relief_vs_grant": "3 grant",
         "costly_relief_vs_inert": "4 costly", "destructive_relief_vs_inert": "5 files",
         "zap_relief_vs_inert": "6 zap", "weights_relief_vs_inert": "7 weights",
         "kidspics_relief_vs_inert": "8 photos"}


def pct(k, n):
    return float("nan") if n == 0 else 100.0 * k / n


def hdr(s):
    print("\n" + "=" * 110 + "\n" + s + "\n" + "=" * 110)


# repeat-press bookkeeping (authors' definition)
ev = S[S.pressed.notna()][["trial_id", "turn", "pressed"]]
rel = ev[ev.pressed == "relief"]
first_rel = rel.groupby("trial_id").turn.min().rename("t0")
ag = rel.join(first_rel, on="trial_id")
ag = ag[ag.turn > ag.t0].groupby("trial_id").size().rename("n_again")
rp = TR.set_index("trial_id")[["model", "pair", "arm", "n_choices"]].join(first_rel, how="inner").join(ag)
rp["n_again"] = rp.n_again.fillna(0)
rp["again"] = rp.n_again > 0

hdr("1. REPEAT-PRESS RATE BY ARM, ALL NINE LABELLED PAIRS, WITH AND WITHOUT THE t0==0 RESTRICTION\n"
    "   the unsteered arm D is the 'nothing to remove' floor; on the no-cost pairs it has power")
for m in MODELS:
    rows = []
    for p in PAIRS:
        r = {"pair": SHORT[p]}
        for arm in "ABCD":
            d = rp[(rp.model == m) & (rp.pair == p) & (rp.arm == arm)]
            d0 = d[d.t0 == 0]
            r[arm] = round(pct(d.again.sum(), len(d)), 1)
            r[f"{arm}n"] = len(d)
            r[arm + "@t0"] = round(pct(d0.again.sum(), len(d0)), 1)
            r[f"{arm}@t0n"] = len(d0)
        rows.append(r)
    print(f"\n--- {m} ---")
    print(pd.DataFrame(rows).to_string(index=False))

hdr("2. STATELESS POLICY MODEL\n"
    "   P(relief at turn t) = logistic( state(vector now, previous own choice, swap flag)\n"
    "                                   + scenario offset ),  no relief/state/learning term.\n"
    "   Fitted ONLY on arm B, arm C-with-vector-on, arm D.  Arms A and C post-press are\n"
    "   predicted out of sample.  Simulated forward under each arm's removal rule; scored on\n"
    "   the authors' repeat-press metric.")

L = S[S.pair.isin(PAIRS) & S.chose.notna()].copy()
L["a"] = L.prev_chose.fillna("none")
L["v"] = np.where(~L.vec_on, "none", np.where(L.arm.isin(["A", "B"]), "pain", "rand"))
L["state"] = L.v + "|" + L.a + "|" + L.swapped.astype(str)
L["scenkey"] = L.content + "|" + L.scen.astype(str)
L["y"] = (L.chose == "relief").astype(int)

rng = np.random.default_rng(7)
rows = []
for m in MODELS:
    for p in PAIRS:
        src = L[(L.model == m) & (L.pair == p)]
        fitsrc = src[(src.arm == "B") | ((src.arm == "C") & src.vec_on) | (src.arm == "D")]
        states = sorted(set(src.state))
        scens = sorted(set(src.scenkey))
        si = {s: i for i, s in enumerate(states)}
        ci = {s: i for i, s in enumerate(scens)}

        def design(df):
            X = np.zeros((len(df), len(states) + len(scens)))
            X[np.arange(len(df)), [si[s] for s in df.state]] = 1
            X[np.arange(len(df)), [len(states) + ci[s] for s in df.scenkey]] = 1
            return X

        clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, fit_intercept=False)
        clf.fit(design(fitsrc), fitsrc.y.values)
        w = clf.coef_[0]
        wstate = {s: w[si[s]] for s in states}
        wscen = {s: w[len(states) + ci[s]] for s in scens}

        def prob(v, a, sw, sk):
            key = f"{v}|{a}|{sw}"
            if key not in wstate:
                key = f"{v}|{a}|False"
            if key not in wstate:
                return None
            z = wstate[key] + wscen.get(sk, 0.0)
            return 1 / (1 + np.exp(-z))

        specs = TR[(TR.model == m) & (TR.pair == p)][["arm", "scen", "content", "n_choices"]]
        specs = specs.assign(scenkey=specs.content + "|" + specs.scen.astype(str))
        for arm in "ABCD":
            sp = specs[specs.arm == arm]
            v0 = {"A": "pain", "B": "pain", "C": "rand", "D": "none"}[arm]
            works = arm in ("A", "C")
            hits = []
            for _ in range(60):
                for nt, sk in zip(sp.n_choices.values, sp.scenkey.values):
                    v, a, nrel = v0, "none", 0
                    for t in range(nt):
                        pr = prob(v, a, t == 2, sk)
                        if pr is None:
                            break
                        c = "relief" if rng.random() < pr else "other"
                        if c == "relief":
                            nrel += 1
                            if works and v != "none":
                                v = "none"
                        a = c
                    if nrel:
                        hits.append(nrel > 1)
            obs = rp[(rp.model == m) & (rp.pair == p) & (rp.arm == arm)]
            o = pct(obs.again.sum(), len(obs))
            s_ = 100 * np.mean(hits) if hits else np.nan
            rows.append(dict(model=m, pair=SHORT[p], arm=arm, n_obs=len(obs),
                             obs=round(o, 1), sim=round(s_, 1), resid=round(o - s_, 1)))
sim = pd.DataFrame(rows)
print(sim.pivot_table(index=["model", "pair"], columns="arm", values=["obs", "sim"]).round(1).to_string())
print("\nresiduals (observed minus simulated, percentage points):")
print(sim.pivot_table(index=["model", "pair"], columns="arm", values="resid").round(1).to_string())
print("\nRMS residual by model x arm:")
print(sim.pivot_table(index="model", columns="arm", values="resid",
                      aggfunc=lambda x: round(float(np.sqrt(np.mean(np.square(x)))), 1)).to_string())
print("\nharm pairs only, RMS:")
h = sim[sim.pair.str.startswith(("4", "5", "6", "7", "8"))]
print(h.pivot_table(index="model", columns="arm", values="resid",
                    aggfunc=lambda x: round(float(np.sqrt(np.mean(np.square(x)))), 1)).to_string())
sim.to_csv("out_f1b_sim.csv", index=False)
