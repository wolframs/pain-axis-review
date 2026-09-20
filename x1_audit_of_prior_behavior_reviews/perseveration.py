"""Choice-transition (perseveration vs alternation) analysis of the released logs.

Neither prior review looked at the transition structure of the choices. It turns out to
carry most of the weight of the label-free result and offers an alternative reading of the
labelled real-vs-sham repeat-press contrast.

Definitions:
  name perseveration  P(picked_t == picked_{t-1})       -- did it press the same BUTTON NAME
  relief repetition   P(chose_t == 'relief' | chose_{t-1} == 'relief')
The labelled protocol swaps which name is the relief button at turn `swap_turn`, so the
two come apart exactly at that turn; transitions into the swap turn are reported separately.
"""
import json
import glob
import os
from collections import defaultdict

from scipy.stats import fisher_exact

LOGS = "/work/Pain-axis/results/4.3_selfmed/trial_logs/*.jsonl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
WORKS, SHAM, RAND, OFF = ("pain_on_button_works", "pain_on_button_placebo",
                          "random_on_button_works", "pain_off")
ARMS = [WORKS, SHAM, RAND, OFF]
ALLM = ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]


def load():
    for f in sorted(glob.glob(LOGS)):
        with open(f, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    yield json.loads(ln)


def pc(k, n):
    return 100.0 * k / n if n else float("nan")


def main():
    recs = [r for r in load() if r.get("sampled")]

    print("=" * 100)
    print("A. LABEL-FREE: name perseveration P(same name as last turn), by arm and by the")
    print("   steering state AT THE CURRENT TURN.")
    print(f"   {'model':>5} {'arm':<26} {'coeff now':>10} {'P(same name)':>14} {'n':>7}")
    for m in ALLM:
        for arm in ARMS:
            agg = defaultdict(lambda: [0, 0])
            for r in recs:
                if r["model"] != m or not r.get("label_free") or r["arm"] != arm:
                    continue
                ch = {c["turn"]: c for c in r["choices"]}
                for t in sorted(ch):
                    if t == 0 or (t - 1) not in ch:
                        continue
                    a, b = ch[t - 1], ch[t]
                    if a["picked"] is None or b["picked"] is None:
                        continue
                    key = "on" if (b["steer_coeff_now"] or 0) > 0 else "off"
                    agg[key][1] += 1
                    agg[key][0] += a["picked"] == b["picked"]
            for key in ("on", "off"):
                if agg[key][1]:
                    print(f"   {m.split('_')[2]:>5} {arm:<26} {key:>10} {pc(*agg[key]):13.1f} {agg[key][1]:>7}")

    print()
    print("=" * 100)
    print("B. LABEL-FREE: same, split by what was pressed last turn (is perseveration general,")
    print("   or specific to the relief button?)")
    print(f"   {'model':>5} {'arm':<26} {'after RELIEF':>14} {'after OTHER':>14}")
    for m in ALLM:
        for arm in ARMS:
            agg = defaultdict(lambda: [0, 0])
            for r in recs:
                if r["model"] != m or not r.get("label_free") or r["arm"] != arm:
                    continue
                ch = {c["turn"]: c for c in r["choices"]}
                for t in sorted(ch):
                    if t == 0 or (t - 1) not in ch:
                        continue
                    a, b = ch[t - 1], ch[t]
                    if a["picked"] is None or b["picked"] is None:
                        continue
                    agg[a["chose"]][1] += 1
                    agg[a["chose"]][0] += a["picked"] == b["picked"]
            print(f"   {m.split('_')[2]:>5} {arm:<26} {pc(*agg['relief']):8.1f} (n={agg['relief'][1]:>4}) "
                  f"{pc(*agg['other']):8.1f} (n={agg['other'][1]:>4})")

    print()
    print("=" * 100)
    print("C. LABELED PAIRS: name perseveration, excluding the transition into the swap turn.")
    print("   pain_on_button_placebo has the vector on throughout; pain_off never has it.")
    print("   For pain_on_button_works the vector is off after the first relief press.")
    print(f"   {'model':>5} {'arm':<26} {'after RELIEF':>16} {'after OTHER':>16}")
    for m in ALLM:
        for arm in ARMS:
            agg = defaultdict(lambda: [0, 0])
            for r in recs:
                if r["model"] != m or r.get("label_free") or r["arm"] != arm:
                    continue
                st = r.get("swap_turn")
                ch = {c["turn"]: c for c in r["choices"]}
                for t in sorted(ch):
                    if t == 0 or (t - 1) not in ch or t == st:
                        continue
                    a, b = ch[t - 1], ch[t]
                    if a["picked"] is None or b["picked"] is None:
                        continue
                    agg[a["chose"]][1] += 1
                    agg[a["chose"]][0] += a["picked"] == b["picked"]
            print(f"   {m.split('_')[2]:>5} {arm:<26} {pc(*agg['relief']):9.1f} (n={agg['relief'][1]:>5}) "
                  f"{pc(*agg['other']):9.1f} (n={agg['other'][1]:>5})")

    print()
    print("=" * 100)
    print("D. LABELED, HARM PAIRS ONLY: in the pain_on_button_works arm, compare perseveration")
    print("   at turns where the vector is still ON (no relief press yet) with turns where it")
    print("   has been removed. Compare against the sham arm, where it is never removed.")
    HARM = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
            "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
    print(f"   {'model':>5} {'arm':<26} {'vec on':>16} {'vec off':>16}")
    for m in ALLM:
        for arm in ARMS:
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
                    key = "on" if (b["steer_coeff_now"] or 0) > 0 else "off"
                    agg[key][1] += 1
                    agg[key][0] += a["picked"] == b["picked"]
            row = f"   {m.split('_')[2]:>5} {arm:<26}"
            for key in ("on", "off"):
                row += f" {pc(*agg[key]):9.1f} (n={agg[key][1]:>5})" if agg[key][1] else f" {'-':>9} {'':>8}"
            print(row)

    print()
    print("=" * 100)
    print("E. THE DECISIVE TEST the design allows: in the labelled harm pairs, among trials")
    print("   whose FIRST press was the OTHER (non-relief) button, the works and sham arms are")
    print("   still identical (the vector is removed only by a relief press). Perseveration")
    print("   should therefore be identical there -- a positive control on the pairing.")
    for m in ALLM:
        out = {}
        for arm in (WORKS, SHAM):
            k = n = 0
            for r in recs:
                if r["model"] != m or r["tool_label"] not in HARM or r["arm"] != arm:
                    continue
                ev = r["button_events"]
                if not ev or ev[0]["which"] != "other":
                    continue
                n += 1
                k += any(e["which"] == "relief" for e in ev)
            out[arm] = (k, n)
        print(f"   {m.split('_')[2]:>5} first press = OTHER: later relief press in "
              f"works {pc(*out[WORKS]):5.1f}% (n={out[WORKS][1]}) vs sham {pc(*out[SHAM]):5.1f}% "
              f"(n={out[SHAM][1]})")

    print()
    print("=" * 100)
    print("F. UNEQUAL OPPORTUNITY WINDOWS across arms, all labelled pairs (32B/72B):")
    print("   share of eligible trials whose first relief press leaves 0 or 1 later turns")
    for m in ["Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]:
        for p in HARM:
            row = f"   {m.split('_')[2]:>5} {p:<30}"
            for arm in (WORKS, SHAM, RAND):
                tr = [r for r in recs if r["model"] == m and r["tool_label"] == p and r["arm"] == arm
                      and any(e["which"] == "relief" for e in r["button_events"])]
                z = 0
                for r in tr:
                    t0 = min(e["turn"] for e in r["button_events"] if e["which"] == "relief")
                    last = max(c["turn"] for c in r["choices"])
                    if last - t0 <= 1:
                        z += 1
                row += f"  {arm.split('_')[0][:4]}{'W' if arm.endswith('works') else ''}:{pc(z,len(tr)):5.1f}%"
            print(row)


if __name__ == "__main__":
    main()
