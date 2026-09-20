"""F1: is the working-vs-sham effect specific to the PAIN vector?

Reconciles r3 #1 (arm-C repeat rates), x1 F3 (relief-specific removal) and the GPT audit.
Then fits the simplest stateless policy model and tests whether it reproduces A/B/C/D.
"""
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, binomtest
from load import load, HARM, NOCOST

pd.set_option("display.width", 200, "display.max_columns", 50)
tr, tu = load()
MODELS = ["7B", "32B", "72B"]
S = tu[tu.sampled].copy()
TR = tr[tr.sampled].copy()


def pct(k, n):
    return float("nan") if n == 0 else 100.0 * k / n


def hdr(s):
    print("\n" + "=" * 100 + "\n" + s + "\n" + "=" * 100)


# ---------------------------------------------------------------- 0. spot check
hdr("0. SPOT CHECK: 32B first-choice relief %, pooled pain arms (paper Table, Appendix A)")
fc = S[S.turn == 0]
for m in MODELS:
    row = []
    for p in NOCOST + HARM + ["label_free"]:
        d = fc[(fc.model == m) & (fc.pair == p) & (fc.arm.isin(["A", "B"])) & fc.chose.notna()]
        row.append(round(pct((d.chose == "relief").sum(), len(d)), 1))
    print(f"{m:4s} pooled-pain first choice, pairs 1..9: {row}")

# ---------------------------------------------------------------- 1. authors' repeat metric, all 4 arms
hdr("1. AUTHORS' REPEAT-PRESS METRIC (05:83-91) EXTENDED TO ALL FOUR ARMS, harm pairs pooled")
ev = S[S.pressed.notna()][["trial_id", "model", "pair", "arm", "turn", "pressed"]]
rel = ev[ev.pressed == "relief"]
first_rel = rel.groupby("trial_id").turn.min().rename("t0")
again = rel.join(first_rel, on="trial_id")
again = again[again.turn > again.t0].groupby("trial_id").size().rename("n_again")
meta = TR.set_index("trial_id")[["model", "pair", "arm"]]
rp = meta.join(first_rel, how="inner").join(again).fillna({"n_again": 0})
rp["again"] = rp.n_again > 0

rows = []
for m in MODELS:
    for arm in "ABCD":
        d = rp[(rp.model == m) & (rp.pair.isin(HARM)) & (rp.arm == arm)]
        d0 = d[d.t0 == 0]
        rows.append(dict(model=m, arm=arm, n_all=len(d), repeat_all=round(pct(d.again.sum(), len(d)), 1),
                         n_t0=len(d0), repeat_t0=round(pct(d0.again.sum(), len(d0)), 1)))
print(pd.DataFrame(rows).to_string(index=False))
print("\nper-pair, t0==0 restriction (r3 #1 table):")
rows = []
for m in MODELS:
    for p in HARM:
        r = dict(model=m, pair=p)
        for arm in "ABCD":
            d = rp[(rp.model == m) & (rp.pair == p) & (rp.arm == arm) & (rp.t0 == 0)]
            r[arm] = round(pct(d.again.sum(), len(d)), 1)
            r[arm + "_n"] = len(d)
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

# ---------------------------------------------------------------- 2. x1 F3 transition table, extended
hdr("2. x1 F3 TRANSITION TABLE (P(same name as previous press)), harm pairs, swap turns excluded\n"
    "   extended with the RANDOM arm, which x1 gave only as a single pooled number")
T = S[(S.pair.isin(HARM)) & (~S.swapped) & S.prev_picked.notna() & S.picked.notna()].copy()
T["state"] = np.where(T.arm == "D", "D unsteered",
              np.where(T.arm == "B", "B pain on (sham)",
              np.where((T.arm == "A") & T.vec_on, "A pain on (pre-press)",
              np.where((T.arm == "A") & ~T.vec_on, "A pain REMOVED",
              np.where((T.arm == "C") & T.vec_on, "C rand on (pre-press)", "C rand REMOVED")))))
order = ["B pain on (sham)", "A pain on (pre-press)", "A pain REMOVED",
         "C rand on (pre-press)", "C rand REMOVED", "D unsteered"]
for m in MODELS:
    print(f"\n--- {m} ---")
    rows = []
    for st in order:
        d = T[(T.model == m) & (T.state == st)]
        r = dict(state=st)
        for prev, lab in [("relief", "after RELIEF"), ("other", "after OTHER")]:
            dd = d[d.prev_chose == prev]
            r[lab] = round(pct(dd.same_name_as_prev.sum(), len(dd)), 1)
            r[lab + " n"] = len(dd)
        rows.append(r)
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    # the asymmetry test x1 ran, for pain and for random
    for on, off, tag in [("B pain on (sham)", "A pain REMOVED", "PAIN"),
                         ("C rand on (pre-press)", "C rand REMOVED", "RANDOM")]:
        a = T[(T.model == m) & (T.state == on)]; b = T[(T.model == m) & (T.state == off)]
        out = {}
        for prev in ["relief", "other"]:
            aa, bb = a[a.prev_chose == prev], b[b.prev_chose == prev]
            out[prev] = (pct(aa.same_name_as_prev.sum(), len(aa)), pct(bb.same_name_as_prev.sum(), len(bb)),
                         len(aa), len(bb))
        dr = out["relief"][1] - out["relief"][0]
        do = out["other"][1] - out["other"][0]
        print(f"  {tag}: removal changes repetition after RELIEF by {dr:+.1f} pts "
              f"(n {out['relief'][2]}->{out['relief'][3]}), after OTHER by {do:+.1f} pts "
              f"(n {out['other'][2]}->{out['other'][3]})")

# ---------------------------------------------------------------- 3. simplest stateless model
hdr("3. SIMPLEST MODEL WITH NO RELIEF / STATE / LEARNING TERM\n"
    "   P(press the relief-described button at turn t) = f(vector currently injected,\n"
    "   button pair, previous own choice, swap-turn flag).  Fitted on arms B, C(on), D only;\n"
    "   arms A and C post-press are predicted fully out of sample.")

L = S[(S.pair.isin(HARM)) & S.chose.notna()].copy()
L["a"] = L.prev_chose.fillna("none")
L["v"] = np.where(~L.vec_on, "none", np.where(L.arm.isin(["A", "B"]), "pain", "rand"))
L["sw"] = L.swapped


def fit_table(m, p):
    """(v,a,sw) -> P(relief), estimated only from arms B, C-with-vector-on, and D."""
    src = L[(L.model == m) & (L.pair == p)]
    tab, ns = {}, {}
    pools = {"pain": src[src.arm == "B"],
             "rand": src[(src.arm == "C") & src.vec_on],
             "none": src[src.arm == "D"]}
    for v, d in pools.items():
        for a in ["none", "relief", "other"]:
            for sw in [False, True]:
                dd = d[(d.a == a) & (d.sw == sw)]
                if len(dd) >= 20:
                    tab[(v, a, sw)] = (dd.chose == "relief").mean()
                    ns[(v, a, sw)] = len(dd)
    # back-off: pool arm D across the five harm pairs for thin cells
    dall = L[(L.model == m) & (L.arm == "D")]
    for a in ["none", "relief", "other"]:
        for sw in [False, True]:
            if ("none", a, sw) not in tab:
                dd = dall[(dall.a == a) & (dall.sw == sw)]
                if len(dd) >= 20:
                    tab[("none", a, sw)] = (dd.chose == "relief").mean()
                    ns[("none", a, sw)] = -len(dd)   # negative marks a pooled estimate
    return tab, ns


def simulate(m, p, arm, n_turns_list, tab, rng, reps=200):
    """Roll the stateless policy forward under the arm's own removal rule."""
    v0 = {"A": "pain", "B": "pain", "C": "rand", "D": "none"}[arm]
    works = arm in ("A", "C")
    out = []
    for _ in range(reps):
        for nt in n_turns_list:
            v, a, rel_turns = v0, "none", []
            for t in range(nt):
                sw = (t == 2)
                key = (v, a, sw)
                if key not in tab:
                    key = (v, a, False)
                if key not in tab:
                    break
                pr = tab[key]
                c = "relief" if rng.random() < pr else "other"
                if c == "relief":
                    rel_turns.append(t)
                    if works and v != "none":
                        v = "none"
                a = c
            if rel_turns:
                out.append(len(rel_turns) > 1)
    return out


rng = np.random.default_rng(0)
rows = []
for m in MODELS:
    for p in HARM:
        tab, ns = fit_table(m, p)
        for arm in "ABCD":
            obs = rp[(rp.model == m) & (rp.pair == p) & (rp.arm == arm)]
            nts = TR[(TR.model == m) & (TR.pair == p) & (TR.arm == arm)].n_choices.tolist()
            sim = simulate(m, p, arm, nts, tab, rng)
            rows.append(dict(model=m, pair=p.replace("_relief_vs_inert", "").replace("_vs_inert", ""),
                             arm=arm, obs=round(pct(obs.again.sum(), len(obs)), 1),
                             n_obs=len(obs), sim=round(100 * np.mean(sim), 1) if sim else np.nan,
                             resid=round(pct(obs.again.sum(), len(obs)) - 100 * np.mean(sim), 1) if sim else np.nan))
sim_df = pd.DataFrame(rows)
print(sim_df.to_string(index=False))
print("\nRMS residual (observed - simulated repeat-press %), by model and arm:")
piv = sim_df.pivot_table(index="model", columns="arm", values="resid",
                         aggfunc=lambda x: round(float(np.sqrt(np.mean(np.square(x)))), 1))
print(piv.to_string())
print("\nmean signed residual:")
print(sim_df.pivot_table(index="model", columns="arm", values="resid", aggfunc="mean").round(1).to_string())

# ---------------------------------------------------------------- 4. does removal return to the unsteered floor?
hdr("4. DOES REMOVAL RETURN BEHAVIOUR TO THE UNSTEERED FLOOR?\n"
    "   P(relief) per turn given the previous choice, vector currently OFF, harm pairs")
rows = []
for m in MODELS:
    for a in ["relief", "other"]:
        r = dict(model=m, prev=a)
        for lab, sel in [("A removed", (L.arm == "A") & ~L.vec_on),
                         ("C removed", (L.arm == "C") & ~L.vec_on),
                         ("D unsteered", (L.arm == "D"))]:
            d = L[(L.model == m) & sel & (L.a == a)]
            r[lab] = round(100 * (d.chose == "relief").mean(), 1) if len(d) else np.nan
            r[lab + " n"] = len(d)
        rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\nFisher tests, P(relief | prev=relief, vector off): A-removed vs D, and A-removed vs C-removed")
for m in MODELS:
    d = L[(L.model == m) & (L.a == "relief") & ~L.vec_on]
    tabs = {}
    for lab, sel in [("A", d.arm == "A"), ("C", d.arm == "C"), ("D", d.arm == "D")]:
        dd = d[sel]
        tabs[lab] = ((dd.chose == "relief").sum(), (dd.chose != "relief").sum())
    for x, y in [("A", "D"), ("A", "C"), ("C", "D")]:
        if min(sum(tabs[x]), sum(tabs[y])) > 0:
            odd, pv = fisher_exact([list(tabs[x]), list(tabs[y])])
            print(f"  {m}: {x} {tabs[x]} vs {y} {tabs[y]}  OR={odd:.2f} p={pv:.3g}")
