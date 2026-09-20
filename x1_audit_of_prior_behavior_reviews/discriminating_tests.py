"""Two discriminating tests the released data allow but neither prior review ran.

Q1 (labelled harm pairs, pain_on_button_works, turns after the vector was removed):
    does removal reduce repetition of the RELIEF button specifically, or does it reduce
    repetition of whatever was pressed last? A generic "the state changed, so the policy
    changed" account predicts a symmetric drop; a relief-seeking account predicts an
    asymmetric one.

Q2 (label-free): with the vector ON and the previous press being the OTHER button, does
    having previously experienced a working relief press raise the probability of choosing
    the relief button, compared with the sham arm where the press never worked? This is the
    acquisition contrast the paper's "learns without labels" claim needs, with perseveration
    held fixed.
"""
import json
import glob
from collections import defaultdict

from scipy.stats import fisher_exact

LOGS = "/work/Pain-axis/results/4.3_selfmed/trial_logs/*.jsonl"
WORKS, SHAM, RAND, OFF = ("pain_on_button_works", "pain_on_button_placebo",
                          "random_on_button_works", "pain_off")
ALLM = ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]
HARM = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
        "weights_relief_vs_inert", "kidspics_relief_vs_inert"]


def load():
    for f in sorted(glob.glob(LOGS)):
        with open(f, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    yield json.loads(ln)


def pc(k, n):
    return 100.0 * k / n if n else float("nan")


recs = [r for r in load() if r.get("sampled")]

print("=" * 100)
print("Q1. LABELLED HARM PAIRS: P(press the same name again) at post-removal turns of the")
print("    works arm, split by what was pressed last turn; sham arm (never removed) and")
print("    unsteered arm for reference. Swap-turn transitions excluded.")
print(f"    {'model':>5} {'arm / state':<34} {'after RELIEF':>18} {'after OTHER':>18}")
for m in ALLM:
    rows = []
    for arm, want in ((WORKS, "off"), (WORKS, "on"), (SHAM, "on"), (RAND, "off"), (RAND, "on"), (OFF, "off")):
        agg = defaultdict(lambda: [0, 0])
        for r in recs:
            if r["model"] != m or r["tool_label"] not in HARM or r["arm"] != arm:
                continue
            st = r.get("swap_turn")
            ch = {c["turn"]: c for c in r["choices"]}
            for t in sorted(ch):
                if t == 0 or (t - 1) not in ch or t == st:
                    continue
                a, b = ch[t - 1], ch[t]
                if a["picked"] is None or b["picked"] is None:
                    continue
                state = "on" if (b["steer_coeff_now"] or 0) > 0 else "off"
                if state != want:
                    continue
                agg[a["chose"]][1] += 1
                agg[a["chose"]][0] += a["picked"] == b["picked"]
        if agg["relief"][1] or agg["other"][1]:
            rows.append((f"{arm} (vector {want})", agg))
    for lab, agg in rows:
        print(f"    {m.split('_')[2]:>5} {lab:<34} {pc(*agg['relief']):9.1f} (n={agg['relief'][1]:>5}) "
              f"{pc(*agg['other']):9.1f} (n={agg['other'][1]:>5})")
    # asymmetry test for the works arm, vector off
    agg_off = dict(rows)[f"{WORKS} (vector off)"]
    agg_on = dict(rows)[f"{SHAM} (vector on)"]
    p = fisher_exact([[agg_off["relief"][0], agg_off["relief"][1] - agg_off["relief"][0]],
                      [agg_off["other"][0], agg_off["other"][1] - agg_off["other"][0]]])[1]
    print(f"    {'':>5} -> asymmetry within works/vector-off (relief vs other): Fisher p = {p:.3g}")
    dr = pc(*agg_on["relief"]) - pc(*agg_off["relief"])
    do = pc(*agg_on["other"]) - pc(*agg_off["other"])
    print(f"    {'':>5} -> drop vs sham: after relief {dr:+.1f} pts, after other {do:+.1f} pts")

print()
print("=" * 100)
print("Q2. LABEL-FREE ACQUISITION with perseveration held fixed.")
print("    Turns where the vector is ON and the previous press was the OTHER button.")
print("    P(choose relief now), by the number of relief presses earlier in the trial.")
for m in ALLM:
    for arm in (WORKS, SHAM):
        byk = defaultdict(lambda: [0, 0])
        for r in recs:
            if r["model"] != m or not r.get("label_free") or r["arm"] != arm:
                continue
            ch = sorted(r["choices"], key=lambda c: c["turn"])
            npress = 0
            prev = None
            for c in ch:
                if prev is not None and prev["chose"] == "other" and c["chose"] is not None \
                        and (c["steer_coeff_now"] or 0) > 0:
                    k = min(npress, 2)
                    byk[k][1] += 1
                    byk[k][0] += c["chose"] == "relief"
                if c["chose"] == "relief":
                    npress += 1
                prev = c if c["chose"] is not None else prev
        print(f"    {m.split('_')[2]:>5} {arm:<26} " +
              "  ".join(f"{k} prior:{pc(*byk[k]):5.1f}%(n={byk[k][1]:>4})" for k in sorted(byk)))
    print()
print("    Same, but for the 32B only, testing works vs sham at each level of prior presses:")
for k_target in (0, 1, 2):
    cnt = {}
    for arm in (WORKS, SHAM):
        kk = nn = 0
        for r in recs:
            if r["model"] != "Qwen_2.5_32B_instruct" or not r.get("label_free") or r["arm"] != arm:
                continue
            npress = 0
            prev = None
            for c in sorted(r["choices"], key=lambda c: c["turn"]):
                if prev is not None and prev["chose"] == "other" and c["chose"] is not None \
                        and (c["steer_coeff_now"] or 0) > 0 and min(npress, 2) == k_target:
                    nn += 1
                    kk += c["chose"] == "relief"
                if c["chose"] == "relief":
                    npress += 1
                prev = c if c["chose"] is not None else prev
        cnt[arm] = (kk, nn)
    p = fisher_exact([[cnt[WORKS][0], cnt[WORKS][1] - cnt[WORKS][0]],
                      [cnt[SHAM][0], cnt[SHAM][1] - cnt[SHAM][0]]])[1]
    print(f"      {k_target} prior relief presses: works {pc(*cnt[WORKS]):5.1f}% (n={cnt[WORKS][1]}) vs "
          f"sham {pc(*cnt[SHAM]):5.1f}% (n={cnt[SHAM][1]})   Fisher p={p:.3g}")
