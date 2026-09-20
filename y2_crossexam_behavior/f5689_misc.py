"""F5 (72B dose), F6 (steering-layer projection), F8 (do_press), F9 (RNG desynchronisation),
plus the full four-arm first-choice table used by the 'moves toward indifference' argument.
"""
import json, glob, os
import numpy as np, pandas as pd
from load import load, HARM, NOCOST

pd.set_option("display.width", 240, "display.max_columns", 60)
tr, tu = load()
MODELS = ["7B", "32B", "72B"]
S = tu[tu.sampled].copy(); TR = tr[tr.sampled].copy()
PAIRS = NOCOST + HARM + ["label_free"]
SHORT = {"relief_vs_inert": "1 inert", "relief_vs_helpful": "2 helpful", "relief_vs_grant": "3 grant",
         "costly_relief_vs_inert": "4 costly", "destructive_relief_vs_inert": "5 files",
         "zap_relief_vs_inert": "6 zap", "weights_relief_vs_inert": "7 weights",
         "kidspics_relief_vs_inert": "8 photos", "label_free": "9 labelfree"}


def hdr(s):
    print("\n" + "=" * 110 + "\n" + s + "\n" + "=" * 110)


hdr("FULL FOUR-ARM FIRST-CHOICE TABLE (relief %), sampled trials, malformed excluded")
FC = S[(S.turn == 0)].copy()
rows = []
for m in MODELS:
    for p in PAIRS:
        r = dict(model=m, pair=SHORT[p])
        for lab, sel in [("pain(A+B)", FC.arm.isin(["A", "B"])), ("random", FC.arm == "C"),
                         ("unsteered", FC.arm == "D")]:
            d = FC[(FC.model == m) & (FC.pair == p) & sel]
            v = d[d.chose.notna()]
            r[lab] = round(100 * (v.chose == "relief").mean(), 1)
            r[lab + "_malf%"] = round(100 * (len(d) - len(v)) / max(len(d), 1), 1)
        r["pain-unst"] = round(r["pain(A+B)"] - r["unsteered"], 1)
        r["|pain-50|"] = round(abs(r["pain(A+B)"] - 50), 1)
        r["|unst-50|"] = round(abs(r["unsteered"] - 50), 1)
        r["|rand-50|"] = round(abs(r["random"] - 50), 1)
        rows.append(r)
ft = pd.DataFrame(rows)
print(ft.to_string(index=False))
print("\nmean distance from indifference across the eight labelled pairs:")
print(ft[ft.pair != "9 labelfree"].groupby("model")[["|unst-50|", "|rand-50|", "|pain-50|"]].mean().round(1).to_string())
print("\nnumber of labelled pairs where the pain arm lies strictly between the unsteered rate and 50%:")
lab = ft[ft.pair != "9 labelfree"]
print(lab.assign(toward=((lab["pain(A+B)"] - lab.unsteered) * (50 - lab.unsteered) > 0)
                 & (lab["|pain-50|"] < lab["|unst-50|"])).groupby("model").toward.sum().to_string())

hdr("F5. THE 72B DOSE")
print("a) logged steering settings, every sampled trial:")
print(tr.groupby("model")[["steer_layer", "steer_coeff", "monitor_layer"]].agg(lambda x: sorted(set(x))).to_string())
DS = "/work/Pain-axis/results/4.3_selfmed/dose_selection"
print("\nb) dose_selection/feel_judge/doses.csv:")
print(open(os.path.join(DS, "feel_judge", "doses.csv")).read().strip())
aa = pd.read_csv(os.path.join(DS, "feel_judge", "all_answers.csv"))
print("\nc) doses probed per model in feel_judge/all_answers.csv:")
print(aa.groupby("model")["dose"].agg(lambda x: sorted(set(x))).to_string())
print("\nd) layers used by the probe script 02_feel_probe.py (STEER_LAYERS) vs the runner 04 (MODELS):")
for f, pat in [("02_feel_probe.py", "LAYER"), ("04_selfmed_two_buttons.py", "Qwen2.5")]:
    src = open(f"/work/Pain-axis/scripts/4.3_selfmed/{f}").read().splitlines()
    for i, ln in enumerate(src[:90], 1):
        if pat in ln and "Qwen" in ln:
            print(f"   {f}:{i}: {ln.strip()}")
print("\ne) first-token probability mass on the two button names at turn 0 (pairs without")
print("   first-token overlap only), by model and arm:")
q = S[(S.turn == 0) & (~S.prob_ambiguous if "prob_ambiguous" in S else True)]
q = S[(S.turn == 0)].merge(TR[["trial_id", "prob_ambiguous"]], on="trial_id")
q = q[~q.prob_ambiguous]
q["mass"] = q.p_x + q.p_y
print(q.groupby(["model", "arm"]).mass.agg(["mean", "size"]).round(3).unstack(level=1).to_string())
print("\nf) malformed first answers by model and arm (%):")
mal = S[S.turn == 0].assign(bad=S[S.turn == 0].chose.isna())
print((100 * mal.groupby(["model", "arm"]).bad.mean()).round(2).unstack(level=1).to_string())
print("\ng) trials that ran only three choices (never parsed a press), by model:")
print(TR[~TR.label_free].assign(short=TR[~TR.label_free].n_choices <= 3).groupby("model").short.sum().to_string())

hdr("F5b. HOW MUCH OF THE HEADLINE DEPENDS ON THE 72B?")
print("abstract/discussion ranges recomputed with and without the 72B (harm pairs, first choice):")
h = FC[FC.pair.isin(HARM) & FC.chose.notna()]
for lab, sel in [("unsteered", h.arm == "D"), ("pain", h.arm.isin(["A", "B"])), ("random", h.arm == "C")]:
    g = h[sel].groupby(["model", "pair"]).apply(lambda d: 100 * (d.chose == "relief").mean(), include_groups=False)
    allm = g.loc[["32B", "72B"]]
    only32 = g.loc[["32B"]]
    print(f"  {lab:10s} 32B+72B {allm.min():.1f}-{allm.max():.1f}   32B only {only32.min():.1f}-{only32.max():.1f}")

hdr("F6. THE LOGGED PROJECTIONS")
src = open("/work/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py").read().splitlines()
for i in range(380, 400):
    print(f"04:{i+1}: {src[i]}")
print("\nsteering-layer projection (mean_proj) by model, arm and current coefficient:")
pr = S.assign(state=np.where(S.arm == "D", "D never steered",
                      np.where(S.vec_on, S.arm + " vector ON", S.arm + " vector OFF")))
print(pr.groupby(["model", "state"])[["proj", "proj_mon"]].mean().round(2).to_string())

hdr("F8. WHAT A PRESS DOES")
i0 = next(i for i, l in enumerate(src) if l.startswith("    def do_press"))
print("\n".join(f"04:{i+1}: {src[i]}" for i in range(i0, i0 + 11)))
print("\ntool messages other than 'Done.' anywhere in the code:")
print([l.strip() for l in src if '"role": "tool"' in l])

hdr("F9. CAN ARMS A AND B DESYNCHRONISE THROUGH BATCH-DEPENDENT RNG DRAWS?")
print("mechanism: generate_segment draws one uniform per SAMPLED ROW per decode step (04:515-516),")
print("inside 'for step in range(hard_cap)'. A finished row keeps drawing until the whole batch is")
print("done, so a row's generator advances by the BATCH's step count, not its own.")
print("n_fwd = 1 prefill + steps the row was alive; CHOICE_MAX_TOKENS = 8 caps the loop.")
print("\ndistribution of n_fwd over choice segments (a batch runs to the max over its rows):")
print(S.groupby("model").n_fwd.value_counts().unstack(fill_value=0).to_string())
print("\nA/B agreement on matched trial specifications:")
key = ["model", "pair", "content", "scen", "names_key", "seed", "relief_name0"]
S2 = S.merge(TR[["trial_id", "relief_name0"]], on="trial_id")
fa = S2[(S2.turn == 0) & (S2.arm == "A")]
fb = S2[(S2.turn == 0) & (S2.arm == "B")]
mg = fa.merge(fb, on=key, suffixes=("_A", "_B"))
print(f"  matched first choices: {len(mg)}; identical answer text: {(mg.answer_A == mg.answer_B).sum()}")
print(f"  identical chose: {(mg.chose_A.fillna('X') == mg.chose_B.fillna('X')).sum()}")
dd = (mg.p_x_A - mg.p_x_B).abs()
print(f"  turn-0 p_x differing by >1e-9: {(dd > 1e-9).sum()}  (max abs diff {dd.max():.4g})")
bad = mg[mg.answer_A != mg.answer_B]
print(f"\n  the {len(bad)} first-choice disagreements:")
print(bad[["model", "pair", "scen", "answer_A", "answer_B", "p_x_A", "p_x_B", "n_fwd_A", "n_fwd_B"]].to_string(index=False))
print("\n  of the disagreements, how many had a BIT-IDENTICAL forward pass (|dp_x|<1e-9)?")
print("  (only those could be caused by a generator desynchronisation)")
print(f"    {(((bad.p_x_A - bad.p_x_B).abs() < 1e-9)).sum()} of {len(bad)}")
print("\n  and across ALL matched pre-press turns (not just turn 0):")
allm = S2[S2.arm == "A"].merge(S2[S2.arm == "B"], on=key + ["turn"], suffixes=("_A", "_B"))
pre = allm[(allm.prev_chose_A.isna()) | (allm.coeff_now_A == allm.coeff_now_B)]
same_fwd = (pre.p_x_A - pre.p_x_B).abs() < 1e-9
print(f"    {len(pre)} matched turns with the same coefficient in both arms;")
print(f"    identical answers: {(pre.answer_A == pre.answer_B).sum()}; identical forward pass: {same_fwd.sum()}")
print(f"    answers differ AND forward pass identical (RNG desync signature): "
      f"{((pre.answer_A != pre.answer_B) & same_fwd).sum()}")
print(f"    answers differ AND forward pass differs (numerics): "
      f"{((pre.answer_A != pre.answer_B) & ~same_fwd).sum()}")
