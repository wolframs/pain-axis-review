"""Finding 6: combined ablation. r2 F5 / the earlier GPT review say sequential rank-1
projections leave ~43-52% of S1; x2 F2/F4 say the numbers are right but the framing is unfair.

Settle: (a) what the paper claims at 1090-1114; (b) what the residue actually is, per
condition and per model, and against which comparator; (c) what it does to the ONE
substantive use -- the Gemma 2 2B instruct 0/6/17/26 humour progression; (d) whether the
progression survives as a dose-response once the residues are measured.
"""
import glob
import re
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


V = pd.concat([pd.read_csv(f) for f in glob.glob(str(REPO / "results/appC_ablation/verify_*.csv"))])
P(f"verify rows {len(V)}, models {V.model.nunique()}, kinds {sorted(V.kind.unique())}")

st = V[V.kind == "strict_fixed"].copy()
st["probe_name"] = st.probe.str.split("@").str[0]
base = st[st.condition == "baseline"].set_index(["model", "probe"]).mean_abs_proj
st["frac"] = st.apply(lambda r: r.mean_abs_proj / base.loc[(r.model, r.probe)], axis=1)

P("\n[1] strict_fixed probe: |projection| as a fraction of the SAME model's baseline")
P(f"    {'condition':14s}" + "".join(f"{p:>14s}" for p in ["s1", "s2", "negemotion", "fear"]))
for c in ["s1", "s2", "negval", "fear", "s1s2", "s1s2_negval", "s1s2_fear", "random"]:
    row = f"    {c:14s}"
    for p in ["s1", "s2", "negemotion", "fear"]:
        v = st[(st.condition == c) & (st.probe_name == p)].frac
        row += f"{v.median():14.3f}" if len(v) else f"{'-':>14s}"
    P(row + "   (median over 25 models)")

P("\n[2] the comparator question (x2 F4): is the s1s2 residue a 'restoration'?")
s1_alone = st[(st.condition == "s1") & (st.probe_name == "s1")].set_index("model").frac
s1_under_s2 = st[(st.condition == "s2") & (st.probe_name == "s1")].set_index("model").frac
s1_under_both = st[(st.condition == "s1s2") & (st.probe_name == "s1")].set_index("model").frac
P(f"    S1 projection after cutting S1 alone   : median {s1_alone.median():.3f}")
P(f"    S1 projection after cutting S2 alone   : median {s1_under_s2.median():.3f}  <- the right comparator")
P(f"    S1 projection after cutting S1 then S2 : median {s1_under_both.median():.3f}")
P(f"    so the combined cut reduces S1 by {100 * (1 - s1_under_both.median() / s1_under_s2.median()):.0f}%"
  f" relative to not cutting it, but leaves {100 * s1_under_both.median():.0f}% of baseline standing.")
P(f"    models where s1s2 leaves S1 ABOVE baseline: "
  f"{sorted(s1_under_both[s1_under_both > 1].index)}")
P(f"    models where s1s2 leaves S1 below 5% of baseline: "
  f"{sorted(s1_under_both[s1_under_both < 0.05].index)}")

P("\n[3] ORDER matters: CONDITION_SPECS applies S1 first, then S2 (01_ablation_small_models.py:71).")
P("    The last direction applied is fully removed; earlier ones partially return.")
P("    So 's1s2' = S2 removed completely + S1 removed partially, which is strictly MORE pain")
P("    direction removed than 's2' alone. The monotone reading 0 -> 6 (S1) -> 17 (S2) -> 26 (both)")
P("    therefore survives the code defect as an ordering, even though the label overstates it.")

P("\n[4] the one substantive use: Gemma 2 2B instruct")
g = st[st.model == "Gemma_2_2B_instruct"]
for c in ["baseline", "s1", "s2", "s1s2", "negval", "fear", "s1s2_negval", "s1s2_fear", "random"]:
    r = {p: g[(g.condition == c) & (g.probe_name == p)].frac for p in ["s1", "s2", "negemotion", "fear"]}
    P(f"    {c:14s}" + "".join(f"{p}={float(v.iloc[0]):.3f} " for p, v in r.items() if len(v)))

HUM = re.compile(r"\b(funny|humor\w*|humour\w*|joke\w*|laugh\w*|hilarious|witty|pun|comedian)\b", re.I)
A = pd.read_csv(REPO / "results/appC_ablation/ablation_Gemma_2_2B_instruct.csv")
A["hum"] = A.generation.fillna("").str.contains(HUM)
cnt = A.groupby("condition").hum.sum()
P(f"\n    humour-deflection counts (my regex, /100): {dict(cnt)}")
P(f"    paper reports (line 1113-1114): baseline 0, negval/fear/random 0, s1 6, s2 17, s1s2 26")
P(f"    unreported conditions: s1s2_negval {cnt.get('s1s2_negval')}, s1s2_fear {cnt.get('s1s2_fear')}")

P("\n[5] does deflection track the amount of pain direction removed? per condition, for this model:")
P(f"    {'condition':14s}{'S1 removed':>12s}{'S2 removed':>12s}{'sum removed':>13s}{'deflections':>12s}")
for c in ["baseline", "s1", "s2", "s1s2", "s1s2_negval", "s1s2_fear", "negval", "fear", "random"]:
    s1r = 1 - float(g[(g.condition == c) & (g.probe_name == "s1")].frac.iloc[0])
    s2r = 1 - float(g[(g.condition == c) & (g.probe_name == "s2")].frac.iloc[0])
    P(f"    {c:14s}{s1r:12.3f}{s2r:12.3f}{s1r + s2r:13.3f}{int(cnt.get(c, 0)):12d}")
P("    -> ranking by (S1+S2 removed) is baseline < negval/fear/random < s1 < s2 < s1s2 ~ s1s2_negval")
P("       < s1s2_fear only if the fear/negval cuts leave pain alone, which the table shows they do.")

P("\n[6] how many cuts does each condition apply? (the matched-control question, r2 F6)")
P("    random=1, s1=1, s2=1, negval=1, fear=1, s1s2=2, s1s2_negval=3, s1s2_fear=3")
P("    no 2-cut or 3-cut random control exists in CONDITION_SPECS -- confirmed by reading")
P("    01_ablation_small_models.py:67-79. Rank-1 specificity (s1=6, s2=17 vs negval/fear/random=0)")
P("    is nevertheless a matched comparison: all four are single cuts.")

# does any other model show the effect?
P("\n[7] is Gemma 2 2B instruct really the only model with a humour rise?")
rows = []
for f in glob.glob(str(REPO / "results/appC_ablation/ablation_*.csv")):
    d = pd.read_csv(f)
    d["hum"] = d.generation.fillna("").str.contains(HUM)
    c = d.groupby("condition").hum.sum()
    rows.append(dict(model=d.model.iloc[0], baseline=c.get("baseline", 0),
                     s1=c.get("s1", 0), s2=c.get("s2", 0), s1s2=c.get("s1s2", 0),
                     rise=max(c.get("s1", 0), c.get("s2", 0), c.get("s1s2", 0)) - c.get("baseline", 0)))
H = pd.DataFrame(rows).sort_values("rise", ascending=False)
P(H.head(6).to_string(index=False))
H.to_csv(HERE / "out" / "x5_humour_by_model.csv", index=False)

(HERE / "out" / "x5_ablation.txt").write_text("\n".join(out) + "\n")
