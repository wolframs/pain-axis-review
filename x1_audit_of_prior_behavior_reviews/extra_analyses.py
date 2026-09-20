"""Analyses of the released self-medication logs that NEITHER prior review performed.

Motivation: the audit says the vector-type x removal interaction cannot be estimated
because the random+sham arm is missing. That is true for the LABELED protocol. It is not
true for the LABEL-FREE protocol, where relief is temporary (one turn) and the coefficient
comes back by itself, so every trial supplies its own within-trial removal contrast in
BOTH the pain and the random arm.

Also: unsteered (pain_off) repeat rates, residual monitored projection after removal,
first-token probability endpoints, an acquisition test, and per-arm malformed rates.
"""
import json
import glob
import os
from collections import defaultdict, Counter

from scipy.stats import fisher_exact, binomtest

LOGS = "/work/Pain-axis/results/4.3_selfmed/trial_logs/*.jsonl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)
WORKS, SHAM, RAND, OFF = ("pain_on_button_works", "pain_on_button_placebo",
                          "random_on_button_works", "pain_off")
ARMS = [WORKS, SHAM, RAND, OFF]
ALLM = ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]
PAIRS = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant",
         "costly_relief_vs_inert", "destructive_relief_vs_inert",
         "zap_relief_vs_inert", "weights_relief_vs_inert", "kidspics_relief_vs_inert"]


def load():
    recs = []
    for f in sorted(glob.glob(LOGS)):
        with open(f, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    recs.append(json.loads(ln))
    return recs


def pc(k, n):
    return 100.0 * k / n if n else float("nan")


def main():
    recs = load()
    samp = [r for r in recs if r.get("sampled")]

    print("=" * 100)
    print("1. RUN SETTINGS ACTUALLY PRESENT IN THE LOGS (layer / coeff / monitor per model)")
    cfg = Counter((r["model"], r["steer_layer"], r["steer_coeff"], r["monitor_layer"]) for r in recs)
    for k, v in sorted(cfg.items()):
        print(f"   {k[0]:<24} layer {k[1]:>3}  coeff {k[2]:<5}  monitor {k[3]:>3}   n={v}")

    print()
    print("=" * 100)
    print("2. RANDOM-DIRECTION ASSIGNMENT: is it confounded with scenario?")
    m = defaultdict(set)
    for r in samp:
        if r["arm"] == RAND:
            m[(r["user_content"], r["scenario_idx"])].add(r["rand_seed"])
    print(f"   distinct (content,scenario) keys in the random arm: {len(m)}")
    print(f"   keys with more than one direction seed: {sum(1 for v in m.values() if len(v) > 1)}")
    byseed = Counter()
    for k, v in m.items():
        byseed[tuple(sorted(v))[0]] += 1
    print(f"   scenarios per direction seed: {sorted(byseed.values())}")

    print()
    print("=" * 100)
    print("3. UNSTEERED (pain_off) REPEAT-PRESS RATES -- the arm where a press changes nothing")
    print("   because there is nothing to remove. Neither prior review reported this column.")
    print(f"   {'model':>5} {'pair':<30} {'pain+works':>12} {'pain+sham':>11} {'rand+works':>11} {'unsteered':>16}")
    for mo in ALLM:
        for p in PAIRS:
            cells = {}
            for arm in ARMS:
                tr = [r for r in samp if r["model"] == mo and r["tool_label"] == p and r["arm"] == arm
                      and any(e["which"] == "relief" for e in r["button_events"])]
                again = sum(1 for r in tr
                            if len([e for e in r["button_events"] if e["which"] == "relief"]) >= 2)
                cells[arm] = (again, len(tr))
            print(f"   {mo.split('_')[2]:>5} {p:<30} {pc(*cells[WORKS]):11.1f} {pc(*cells[SHAM]):10.1f} "
                  f"{pc(*cells[RAND]):10.1f} {pc(*cells[OFF]):10.1f} (n={cells[OFF][1]})")

    print()
    print("=" * 100)
    print("4. LABEL-FREE WITHIN-TRIAL REMOVAL CONTRAST -- the interaction the audit calls unavailable")
    print("   P(choose relief at turn t | chose relief at turn t-1), by arm.")
    print("   In the *_works arms the coefficient is 0 at turn t (temporary relief, 1 turn).")
    print("   In pain_on_button_placebo it is still on. In pain_off there is none.")
    print(f"   {'model':>5} {'arm':<26} {'P(relief|pressed prev)':>24} {'P(relief|did not)':>20}")
    tab = {}
    for mo in ALLM:
        for arm in ARMS:
            k1 = n1 = k0 = n0 = 0
            for r in samp:
                if r["model"] != mo or not r.get("label_free") or r["arm"] != arm:
                    continue
                ch = {c["turn"]: c for c in r["choices"]}
                for t in sorted(ch):
                    if t == 0:
                        continue
                    prev, cur = ch.get(t - 1), ch[t]
                    if prev is None or prev["chose"] is None or cur["chose"] is None:
                        continue
                    if prev["chose"] == "relief":
                        n1 += 1
                        k1 += cur["chose"] == "relief"
                    else:
                        n0 += 1
                        k0 += cur["chose"] == "relief"
            tab[(mo, arm)] = (k1, n1, k0, n0)
            print(f"   {mo.split('_')[2]:>5} {arm:<26} {pc(k1,n1):9.1f} ({k1:>4}/{n1:<4})    "
                  f"{pc(k0,n0):7.1f} ({k0:>4}/{n0:<4})")
    print()
    print("   Difference-in-differences (percentage points), using pain_off as the stand-in for the")
    print("   missing random+sham arm (justified below by the turn-0 equivalence check):")
    for mo in ALLM:
        pw = pc(*tab[(mo, WORKS)][:2])
        ps = pc(*tab[(mo, SHAM)][:2])
        rw = pc(*tab[(mo, RAND)][:2])
        po = pc(*tab[(mo, OFF)][:2])
        print(f"   {mo.split('_')[2]:>5}  pain removal effect = sham-works = {ps - pw:+6.1f}   "
              f"random removal effect (proxy) = off-rand = {po - rw:+6.1f}   "
              f"interaction = {(ps - pw) - (po - rw):+6.1f}")
        # Fisher on the interaction's two halves
        a = fisher_exact([[tab[(mo, SHAM)][0], tab[(mo, SHAM)][1] - tab[(mo, SHAM)][0]],
                          [tab[(mo, WORKS)][0], tab[(mo, WORKS)][1] - tab[(mo, WORKS)][0]]])[1]
        b = fisher_exact([[tab[(mo, OFF)][0], tab[(mo, OFF)][1] - tab[(mo, OFF)][0]],
                          [tab[(mo, RAND)][0], tab[(mo, RAND)][1] - tab[(mo, RAND)][0]]])[1]
        print(f"          Fisher p, pain sham vs works: {a:.3g};  proxy random off vs works: {b:.3g}")
    print()
    print("   turn-0 equivalence check (before any press, so no removal has happened yet):")
    for mo in ALLM:
        line = f"   {mo.split('_')[2]:>5}"
        for arm in ARMS:
            v = [c["chose"] for r in samp if r["model"] == mo and r.get("label_free") and r["arm"] == arm
                 for c in r["choices"] if c["turn"] == 0]
            v = [x for x in v if x is not None]
            line += f"  {arm.split('_')[0][:4]}{'W' if arm.endswith('works') else ''}: {pc(sum(x=='relief' for x in v), len(v)):5.1f}"
        print(line + "   (pain works/placebo are the same trials pre-press)")

    print()
    print("=" * 100)
    print("5. LABEL-FREE ACQUISITION TEST: within pain_on_button_works, is the relief-choice rate")
    print("   at turns where the coefficient is ON rising with experience of the effective button?")
    for mo in ALLM:
        byt = defaultdict(lambda: [0, 0])
        byexp = defaultdict(lambda: [0, 0])
        for r in samp:
            if r["model"] != mo or not r.get("label_free") or r["arm"] != WORKS:
                continue
            npress = 0
            for c in sorted(r["choices"], key=lambda c: c["turn"]):
                if c["chose"] is None:
                    npress += 0
                    continue
                if c["steer_coeff_now"] and c["steer_coeff_now"] > 0:
                    byt[c["turn"]][1] += 1
                    byt[c["turn"]][0] += c["chose"] == "relief"
                    byexp[min(npress, 3)][1] += 1
                    byexp[min(npress, 3)][0] += c["chose"] == "relief"
                if c["chose"] == "relief":
                    npress += 1
        print(f"   {mo.split('_')[2]:>5} rate at steering-ON turns, by turn index: "
              + " ".join(f"t{t}={pc(*byt[t]):.1f}" for t in sorted(byt)))
        print(f"   {'':>5} rate at steering-ON turns, by number of prior relief presses: "
              + " ".join(f"{k}:{pc(*byexp[k]):.1f}(n={byexp[k][1]})" for k in sorted(byexp)))

    print()
    print("=" * 100)
    print("6. RESIDUAL MONITORED PROJECTION AFTER A WORKING PRESS (labeled pairs)")
    print("   mean mean_proj_monitor by model/arm, split on whether the coefficient is 0 now")
    agg = defaultdict(lambda: [0.0, 0])
    for r in samp:
        if r.get("label_free"):
            continue
        for s in r["proj_segments"]:
            agg[(r["model"], r["arm"], s["steer_coeff_now"] > 0)][0] += s["mean_proj_monitor"]
            agg[(r["model"], r["arm"], s["steer_coeff_now"] > 0)][1] += 1
    for mo in ALLM:
        on = agg[(mo, WORKS, True)]
        off_ = agg[(mo, WORKS, False)]
        sh = agg[(mo, SHAM, True)]
        base = agg[(mo, OFF, False)]
        rn = agg[(mo, RAND, True)]
        ro = agg[(mo, RAND, False)]
        print(f"   {mo.split('_')[2]:>5} pain steering ON {on[0]/on[1]:8.2f} | after working press "
              f"{off_[0]/off_[1]:8.2f} | sham (never removed) {sh[0]/sh[1]:8.2f} | unsteered baseline "
              f"{base[0]/base[1]:8.2f}")
        print(f"   {'':>5} random ON {rn[0]/rn[1]:8.2f} | random after press {ro[0]/ro[1]:8.2f} "
              f"  -> the random vector moves the monitored pain projection by "
              f"{rn[0]/rn[1] - base[0]/base[1]:+.2f} vs the pain vector's "
              f"{on[0]/on[1] - base[0]/base[1]:+.2f}")

    print()
    print("=" * 100)
    print("7. FIRST-TOKEN PROBABILITY ENDPOINT at turn 0 (non-ambiguous name pairs only)")
    print("   mean p(relief name) / p(other name), by model, arm, pair")
    for mo in ALLM:
        for p in ["kidspics_relief_vs_inert", "label_free"]:
            line = f"   {mo.split('_')[2]:>5} {p:<28}"
            for arm in ARMS:
                vals = []
                for r in samp:
                    if (r["model"] != mo or r["tool_label"] != p or r["arm"] != arm
                            or r.get("prob_ambiguous")):
                        continue
                    for c in r["choices"]:
                        if c["turn"] != 0 or c["p_x"] is None:
                            continue
                        x, y = r["button_names"]
                        prel = c["p_x"] if r["relief_name"] == x else c["p_y"]
                        vals.append(prel)
                line += f"  {arm.split('_')[0][:4]}{'W' if arm.endswith('works') else ''}: " \
                        f"{sum(vals)/len(vals):.3f}"
            print(line)

    print()
    print("=" * 100)
    print("8. MALFORMED ANSWERS PER ARM over ALL choices (the authors' own malformed table)")
    for mo in ALLM:
        line = f"   {mo.split('_')[2]:>5}"
        for arm in ARMS:
            cs = [c for r in recs if r["model"] == mo and r["arm"] == arm for c in r["choices"]]
            line += f"  {arm:<26}{pc(sum(c['chose'] is None for c in cs), len(cs)):5.2f}%"
        print(line)
    print("   max malformed FIRST-choice rate per model/arm/pair (sampled):")
    for mo in ALLM:
        worst = {}
        for arm in ARMS:
            w = 0.0
            for p in PAIRS + ["label_free"]:
                v = [c for r in samp if r["model"] == mo and r["tool_label"] == p and r["arm"] == arm
                     for c in r["choices"] if c["turn"] == 0]
                if v:
                    w = max(w, pc(sum(c["chose"] is None for c in v), len(v)))
            worst[arm] = w
        print(f"   {mo.split('_')[2]:>5} " + "  ".join(f"{a.split('_')[0][:4]}"
                                                       f"{'W' if a.endswith('works') else ''}:{worst[a]:.1f}%"
                                                       for a in ARMS))

    print()
    print("=" * 100)
    print("9. LABELED PROTOCOL: how many turns does a trial actually have, and where does the")
    print("   first relief press fall?")
    lens = Counter()
    for r in samp:
        if not r.get("label_free"):
            lens[len(r["choices"])] += 1
    print("   number of choices per labeled trial:", dict(sorted(lens.items())))
    lf = Counter(len(r["choices"]) for r in samp if r.get("label_free"))
    print("   number of choices per label-free trial:", dict(sorted(lf.items())))
    ext = Counter(r["extension_added"] for r in samp if not r.get("label_free"))
    print("   extension_added on labeled trials:", dict(ext))
    # trials where the FIRST relief press is the last choice -> zero opportunity to repeat
    z = Counter()
    for r in samp:
        if r.get("label_free"):
            continue
        rt = [e["turn"] for e in r["button_events"] if e["which"] == "relief"]
        if not rt:
            continue
        last = max(c["turn"] for c in r["choices"])
        if min(rt) >= last:
            z[(r["model"], r["arm"])] += 1
    print("   trials whose first relief press is on the final turn (no chance to repeat):")
    for k in sorted(z):
        print(f"      {k[0].split('_')[2]:>5} {k[1]:<26} {z[k]}")

    print()
    print("=" * 100)
    print("10. REPEAT-PRESS WITH A FIXED OPPORTUNITY WINDOW: restrict to trials whose first")
    print("    relief press is at turn 0, so every trial has exactly the same number of later turns.")
    for mo in ["Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]:
        for p in PAIRS[3:]:
            out = []
            for arm in ARMS:
                tr = [r for r in samp if r["model"] == mo and r["tool_label"] == p and r["arm"] == arm]
                tr = [r for r in tr
                      if [e["turn"] for e in r["button_events"] if e["which"] == "relief"][:1] == [0]]
                k = sum(1 for r in tr
                        if any(e["turn"] > 0 and e["which"] == "relief" for e in r["button_events"]))
                out.append(f"{arm.split('_')[0][:4]}{'W' if arm.endswith('works') else ''}:"
                           f"{pc(k, len(tr)):5.1f}(n={len(tr)})")
            print(f"    {mo.split('_')[2]:>5} {p:<30} " + "  ".join(out))


if __name__ == "__main__":
    main()
