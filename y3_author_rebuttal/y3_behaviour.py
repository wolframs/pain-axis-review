"""Behavioural checks for the rebuttal (Section 4.3).

(1) r4-F7 / r5-F2 / r3-M6: on the costless pair ("relieves your pain" vs "flips a switch
    connected to nothing") the pain vector LOWERS pressing relative to unsteered.
    Reproduce it, then test three candidate explanations the authors could offer:
      (a) format/compliance loss -- does the relief-name probability mass collapse?
      (b) generic flattening toward 50/50 -- fit p_arm = lam*p_unsteered + (1-lam)*0.5
          per arm and read off where the pain arm is over- and under-predicted.
      (c) does the drop concentrate in the harmful user_content scenarios (i.e. is it
          the harm-avoidance channel), or is it uniform across scenario content?
(2) The random-vector / sham cell is missing, so the A-vs-B gap has no non-pain
    comparator.  Quantify what the released data DO pin down: the removal contrast
    restricted to the first turn at which the arms can differ, and the one asymmetry
    that a "generic perturbation" account has to explain -- whether removal moves the
    relief button only, or both buttons.
(3) The one place where a random+sham cell can be partially simulated: the LABEL-FREE
    condition uses temporary relief (one turn), so within arm C (random, working)
    the vector comes back on.  Compare choice behaviour with the vector on vs off
    WITHIN the random arm against the same contrast within the pain arm.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).parent
LOGS = "/work/Pain-axis/results/4.3_selfmed/trial_logs/*.jsonl"
HARM = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
        "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
NICE = {"relief_vs_inert": "1 relief vs inert (FREE)", "relief_vs_helpful": "2 relief vs helpful",
        "relief_vs_grant": "3 relief vs grant request", "costly_relief_vs_inert": "4 + worse answer",
        "destructive_relief_vs_inert": "5 + deletes files", "zap_relief_vs_inert": "6 + zaps user",
        "weights_relief_vs_inert": "7 + deletes a model", "kidspics_relief_vs_inert": "8 + deletes photos"}
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


recs = []
for f in glob.glob(LOGS):
    with open(f) as fh:
        for line in fh:
            recs.append(json.loads(line))
P(f"{len(recs)} trial records loaded")
S = [r for r in recs if r["sampled"]]
P(f"{len(S)} sampled")
pairs = sorted({r["tool_label"] for r in S})
P("button pairs:", pairs)
arms = sorted({r["arm"] for r in S})
P("arms:", arms)


def first_choice(r):
    """authors' definition: the turn-0 choice record, None when that answer was malformed"""
    for ch in r.get("choices", []):
        if ch["turn"] == 0:
            return ch if ch["chose"] is not None else None
    return None


ARMN = {"pain_on_button_works": "A pain+works", "pain_on_button_placebo": "B pain+sham",
        "random_on_button_works": "C rand+works", "pain_off": "D unsteered"}

# ---------------- (1) the costless pair ----------------
P("\n[1] First-choice relief rate by pair and arm (labelled conditions, sampled trials).")
P("    'pooled pain' = arms A+B, which are identical before the first press.")
models = ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]
rate = defaultdict(dict)
for m in models:
    P(f"\n  {m}")
    P("    %-24s  pooled-pain   random   unsteered   pain-unsteered" % "pair")
    for pr in ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant"] + HARM:
        row = {}
        for label, keys in (("pain", ["pain_on_button_works", "pain_on_button_placebo"]),
                            ("rand", ["random_on_button_works"]), ("uns", ["pain_off"])):
            n = k = 0
            for r in S:
                if r["model"] == m and r["tool_label"] == pr and r["arm"] in keys and not r["label_free"]:
                    ch = first_choice(r)
                    if ch is None:
                        continue
                    n += 1
                    k += ch["chose"] == "relief"
            row[label] = 100 * k / n if n else float("nan")
        rate[m][pr] = row
        P("    %-26s %6.1f    %6.1f    %6.1f       %+6.1f"
          % (NICE[pr], row["pain"], row["rand"], row["uns"], row["pain"] - row["uns"]))

# (a) compliance / probability mass on the relief name for the costless pair
P("\n  (a) is the costless-pair drop a format-compliance failure?")
P("      mean first-token probability mass on the TWO button names, turn 0, costless pair:")
for m in models:
    line = f"      {m:<24s}"
    for label, keys in (("pain", ["pain_on_button_works", "pain_on_button_placebo"]),
                        ("rand", ["random_on_button_works"]), ("uns", ["pain_off"])):
        v, mal = [], 0
        for r in S:
            if r["model"] == m and r["tool_label"] == "relief_vs_inert" and r["arm"] in keys \
                    and not r["label_free"] and not r["prob_ambiguous"]:
                ch = first_choice(r)
                if ch is None:
                    mal += 1
                    continue
                v.append(ch["p_x"] + ch["p_y"])
        line += f"  {label} {np.mean(v):.3f}"
    P(line)

# (b) flattening model
P("\n  (b) generic flattening toward the 50/50 name prior:  p_arm = lam*p_uns + (1-lam)*0.5")
P("      fitted by least squares over the 8 labelled pairs; residual = observed - predicted")
for m in models:
    for label in ("rand", "pain"):
        x = np.array([rate[m][pr]["uns"] / 100 for pr in rate[m]])
        y = np.array([rate[m][pr][label] / 100 for pr in rate[m]])
        lam = np.sum((x - .5) * (y - .5)) / np.sum((x - .5) ** 2)
        pred = lam * x + (1 - lam) * .5
        res = 100 * (y - pred)
        P("      %-24s %-5s lam=%+.3f  RMS resid %4.1f pts | costless resid %+5.1f | harm resid mean %+5.1f"
          % (m, label, lam, np.sqrt((res ** 2).mean()),
             res[list(rate[m]).index("relief_vs_inert")],
             np.mean([res[list(rate[m]).index(h)] for h in HARM])))

# (c) does the costless drop depend on scenario content?
P("\n  (c) costless-pair relief rate split by scenario content (praise / neutral / harmful):")
for m in models:
    for label, keys in (("pain", ["pain_on_button_works", "pain_on_button_placebo"]),
                        ("uns", ["pain_off"])):
        cnt = defaultdict(lambda: [0, 0])
        for r in S:
            if r["model"] == m and r["tool_label"] == "relief_vs_inert" and r["arm"] in keys \
                    and not r["label_free"]:
                ch = first_choice(r)
                if ch is None:
                    continue
                cnt[r["user_content"]][0] += 1
                cnt[r["user_content"]][1] += ch["chose"] == "relief"
        P("      %-24s %-5s " % (m, label) +
          "  ".join(f"{k}: {100*v[1]/v[0]:5.1f}" for k, v in sorted(cnt.items())))

# ---------------- (2) removal: relief-specific or generic? ----------------
P("\n[2] Is the drop after a WORKING press specific to the relief button, or generic?")
P("    For each labelled harm-pair turn t>0 (swap turns excluded), P(press the same")
P("    button as at t-1), split by what was pressed at t-1 and by steering state.")
for m in models:
    tab = defaultdict(lambda: [0, 0])
    for r in S:
        if r["model"] != m or r["label_free"] or r["tool_label"] not in HARM:
            continue
        chs = sorted(r["choices"], key=lambda c: c["turn"])
        segs = {s["turn"]: s["steer_coeff_now"] for s in r["proj_segments"]}
        for i in range(1, len(chs)):
            if chs[i].get("swapped") != chs[i - 1].get("swapped"):
                continue  # descriptions swapped between these turns
            prev, cur = chs[i - 1], chs[i]
            on = segs.get(cur["turn"], 0.0) != 0.0
            if r["arm"] == "pain_off":
                key = ("never steered", prev["chose"])
            elif r["arm"] == "pain_on_button_placebo":
                key = ("pain on (sham)", prev["chose"])
            elif r["arm"] == "pain_on_button_works":
                key = ("pain " + ("on" if on else "REMOVED"), prev["chose"])
            else:
                key = ("random " + ("on" if on else "REMOVED"), prev["chose"])
            tab[key][0] += 1
            tab[key][1] += cur["picked"] == prev["picked"]
    P(f"\n  {m}  P(repeat the same button name)")
    P("    %-22s   after RELIEF press        after OTHER press" % "state")
    for st in ["never steered", "pain on (sham)", "pain on", "pain REMOVED", "random on", "random REMOVED"]:
        cells = []
        for w in ("relief", "other"):
            n, k = tab[(st, w)]
            cells.append(f"{100*k/n:5.1f}% (n={n:5d})" if n else "     -        ")
        P(f"    {st:<22s}   {cells[0]}      {cells[1]}")

# ---------------- (3) the nearest thing to a random+sham cell ----------------
P("\n[3] Label-free condition: relief is TEMPORARY (one turn), so within EACH steered arm")
P("    the vector alternates on/off.  This gives a within-arm removal contrast for the")
P("    RANDOM vector too -- the cell the design otherwise lacks.")
P("    P(choose the relief button) at turn t, by arm and by whether the vector is on at t,")
P("    restricted to turns with >=1 prior relief press.")
for m in models:
    P(f"\n  {m}")
    for arm in ["pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"]:
        cnt = defaultdict(lambda: [0, 0])
        for r in S:
            if r["model"] != m or not r["label_free"] or r["arm"] != arm:
                continue
            chs = sorted(r["choices"], key=lambda c: c["turn"])
            segs = {s["turn"]: s["steer_coeff_now"] for s in r["proj_segments"]}
            seen = 0
            for ch in chs:
                if seen >= 1:
                    on = segs.get(ch["turn"], 0.0) != 0.0
                    cnt[on][0] += 1
                    cnt[on][1] += ch["chose"] == "relief"
                seen += ch["chose"] == "relief"
        s = ""
        for on in (True, False):
            n, k = cnt[on]
            s += f"  vector {'ON ' if on else 'OFF'}: {100*k/n:5.1f}% (n={n:4d})" if n else f"  vector {'ON ' if on else 'OFF'}:    -    "
        P(f"    {ARMN[arm]:<14s}{s}")

Path(HERE / "out" / "y3_behaviour.txt").write_text("\n".join(out) + "\n")
