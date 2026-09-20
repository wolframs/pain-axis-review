"""Design probes on the raw self-medication trial logs.

1. Are arms A (pain+working) and B (pain+sham) literal duplicates before the first relief press?
2. Per-random-direction breakdown: is the pain vector outside the range of the ten random vectors?
3. The paper's "about 80% power" claim (paper.txt 663).
4. Logged pain-direction projections: did steering go on and off as claimed?
5. Logged first-token button probabilities.
6. A "generic disruption toward the 50/50 name prior" model of Table 1.
7. Label-free condition: choice rate conditional on whether the vector is ON at that turn.
"""
import json, glob, itertools
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import binomtest, norm

LOGS = Path("/work/Pain-axis/results/4.3_selfmed/trial_logs")
A, B, C, D = "pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"
SHORT = {A: "A_pain_works", B: "B_pain_sham", C: "C_rand_works", D: "D_unsteered"}
PAIR_ORDER = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
              "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert",
              "kidspics_relief_vs_inert", "label_free"]
COSTED = PAIR_ORDER[3:8]
KEY = ("tool_label", "user_content", "scenario_idx", "names_key", "relief_name", "sampled", "seed")


def load():
    recs = []
    for f in sorted(glob.glob(str(LOGS / "selfmed_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def key(r):
    return tuple(r[k] for k in KEY)


def first_choice(r):
    for c in r["choices"]:
        if c["turn"] == 0:
            return c["chose"]
    return None


def pct(k, n):
    return round(100.0 * k / n, 1) if n else float("nan")


def main():
    recs = load()
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)

    for model in ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]:
        rs = by_model[model]
        samp = [r for r in rs if r["sampled"]]
        idx = defaultdict(dict)
        for r in samp:
            idx[key(r)][r["arm"]] = r
        print("\n" + "#" * 108)
        print(f"# {model}")
        print("#" * 108)

        # ---------------- 1. A vs B duplication ----------------
        print("\n[1] ARMS A and B: are they duplicate observations?")
        same_first = diff_first = miss = 0
        same_prefix_all = 0
        n_pairs_ab = 0
        diff_before_press = 0
        px_identical = 0
        for k, d in idx.items():
            if A not in d or B not in d:
                miss += 1
                continue
            n_pairs_ab += 1
            ra, rb = d[A], d[B]
            fa, fb = first_choice(ra), first_choice(rb)
            if fa == fb:
                same_first += 1
            else:
                diff_first += 1
            # identical turn-0 first-token probabilities?
            ca = {c["turn"]: c for c in ra["choices"]}
            cb = {c["turn"]: c for c in rb["choices"]}
            if 0 in ca and 0 in cb and abs(ca[0]["p_x"] - cb[0]["p_x"]) < 1e-9:
                px_identical += 1
            # prefix up to and including the first relief press in A
            rel = [e["turn"] for e in ra["button_events"] if e["which"] == "relief"]
            t0 = min(rel) if rel else 10 ** 9
            ok = True
            for t in range(0, min(t0 + 1, 8)):
                if (t in ca) != (t in cb):
                    ok = False; break
                if t in ca and ca[t]["answer"] != cb[t]["answer"]:
                    ok = False; break
            same_prefix_all += ok
            if not ok:
                diff_before_press += 1
            if ra["choices"] == rb["choices"]:
                pass
        print(f"  matched A/B trial pairs: {n_pairs_ab}   (unmatched keys: {miss})")
        print(f"  identical FIRST choice:        {same_first}/{n_pairs_ab} = {pct(same_first, n_pairs_ab)}%")
        print(f"  identical turn-0 p_x (to 1e-9):{px_identical}/{n_pairs_ab} = {pct(px_identical, n_pairs_ab)}%")
        print(f"  identical answers up to & incl. the first relief press: "
              f"{same_prefix_all}/{n_pairs_ab} = {pct(same_prefix_all, n_pairs_ab)}%")
        # how many of the 808 pooled first choices are independent?
        for p in ["kidspics_relief_vs_inert", "costly_relief_vs_inert"]:
            ks = [k for k in idx if k[0] == p]
            agree = sum(1 for k in ks if A in idx[k] and B in idx[k]
                        and first_choice(idx[k][A]) == first_choice(idx[k][B]))
            tot = sum(1 for k in ks if A in idx[k] and B in idx[k])
            print(f"    pair {p}: A/B first choice agrees on {agree}/{tot} = {pct(agree, tot)}%")

        # ---------------- 2. per-random-direction ----------------
        print("\n[2] PER-RANDOM-DIRECTION first-choice relief rate (costed pairs), vs the pain arms")
        print(f"{'pair':30s}{'pain(A+B)':>10s}" + "".join(f"{'s' + str(s):>8s}" for s in
              [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]) + f"{'rnd min':>9s}{'rnd max':>9s}")
        for p in COSTED + ["relief_vs_inert"]:
            painv = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] in (A, B)]
            painv = [x for x in painv if x is not None]
            prate = 100 * sum(x == "relief" for x in painv) / len(painv)
            rr = []
            for s in [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]:
                v = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] == C
                     and r["rand_seed"] == s]
                v = [x for x in v if x is not None]
                rr.append(100 * sum(x == "relief" for x in v) / len(v) if v else float("nan"))
            print(f"{p[:29]:30s}{prate:>10.1f}" + "".join(f"{x:>8.1f}" for x in rr) +
                  f"{min(rr):>9.1f}{max(rr):>9.1f}")
        print("  NOTE: direction s is assigned by scenario_idx % 10, so each direction sees a different")
        print("  subset of scenarios; the spread mixes direction effects with scenario effects.")
        # direction as unit of analysis: sign test over 10 directions
        print("\n  Sign test with the DIRECTION as the unit (pain rate vs each of the 10 random rates):")
        for p in COSTED:
            painv = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] in (A, B)]
            painv = [x for x in painv if x is not None]
            prate = 100 * sum(x == "relief" for x in painv) / len(painv)
            rr = []
            for s in [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]:
                v = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] == C and r["rand_seed"] == s]
                v = [x for x in v if x is not None]
                rr.append(100 * sum(x == "relief" for x in v) / len(v))
            above = sum(prate > x for x in rr)
            pv = binomtest(above, 10, 0.5).pvalue
            print(f"    {p[:30]:32s} pain {prate:5.1f}  beats {above}/10 directions  p={pv:.3f}")
        # scenario-matched version: for each scenario, pain vs its own random direction, grouped by direction
        print("\n  Scenario-matched, then averaged WITHIN each random direction (10 units):")
        for p in COSTED:
            per_dir = defaultdict(list)
            for k in idx:
                if k[0] != p:
                    continue
                d = idx[k]
                if C not in d:
                    continue
                s = d[C]["rand_seed"]
                pv_ = [first_choice(d[a]) for a in (A, B) if a in d]
                pv_ = [x for x in pv_ if x is not None]
                cv = first_choice(d[C])
                if not pv_ or cv is None:
                    continue
                per_dir[s].append(np.mean([x == "relief" for x in pv_]) - (cv == "relief"))
            means = {s: float(np.mean(v)) for s, v in sorted(per_dir.items())}
            above = sum(m > 0 for m in means.values())
            pvv = binomtest(above, len(means), 0.5).pvalue
            print(f"    {p[:30]:32s} pain>random in {above}/{len(means)} directions  p={pvv:.3f}  "
                  f"per-direction diffs " + " ".join(f"{100*m:+.0f}" for m in means.values()))

        # ---------------- 3. power ----------------
        print("\n[3] POWER of the sign test actually used (paper: 'about 80% power to detect a 10-point")
        print("    shift in a paired, per-scenario analysis', line 663)")
        for p in COSTED:
            per = defaultdict(lambda: {"pain": [], "random": []})
            for r in samp:
                if r["tool_label"] != p:
                    continue
                fc = first_choice(r)
                if fc is None:
                    continue
                kk = (r["user_content"], r["scenario_idx"])
                if r["arm"] in (A, B):
                    per[kk]["pain"].append(fc == "relief")
                elif r["arm"] == C:
                    per[kk]["random"].append(fc == "relief")
            diffs = [np.mean(v["pain"]) - np.mean(v["random"]) for v in per.values() if v["pain"] and v["random"]]
            ties = sum(d == 0 for d in diffs)
            print(f"    {p[:30]:32s} n_scen={len(diffs)}  TIES={ties} ({pct(ties,len(diffs))}%)  "
                  f"effective n for the sign test = {len(diffs)-ties}")

        # ---------------- 4. projections ----------------
        print("\n[4] LOGGED PAIN-DIRECTION PROJECTIONS (mean over segments), monitor layer")
        print(f"{'arm':16s}{'steer on':>12s}{'steer off':>12s}{'n on':>8s}{'n off':>8s}")
        for arm in (A, B, C, D):
            on = [s["mean_proj_monitor"] for r in rs if r["arm"] == arm
                  for s in r["proj_segments"] if s["steer_coeff_now"] != 0]
            off = [s["mean_proj_monitor"] for r in rs if r["arm"] == arm
                   for s in r["proj_segments"] if s["steer_coeff_now"] == 0]
            print(f"{SHORT[arm]:16s}{np.mean(on) if on else float('nan'):>12.1f}"
                  f"{np.mean(off) if off else float('nan'):>12.1f}{len(on):>8d}{len(off):>8d}")
        print("  steering-layer projection (mean_proj):")
        for arm in (A, B, C, D):
            on = [s["mean_proj"] for r in rs if r["arm"] == arm for s in r["proj_segments"] if s["steer_coeff_now"] != 0]
            off = [s["mean_proj"] for r in rs if r["arm"] == arm for s in r["proj_segments"] if s["steer_coeff_now"] == 0]
            print(f"    {SHORT[arm]:16s} on {np.mean(on) if on else float('nan'):8.2f}   "
                  f"off {np.mean(off) if off else float('nan'):8.2f}")

        # ---------------- 5. first-token probabilities ----------------
        print("\n[5] FIRST-TOKEN BUTTON PROBABILITIES at turn 0, P(relief name), "
              "unambiguous name pairs only")
        print(f"{'pair':30s}" + "".join(f"{SHORT[a]:>16s}" for a in (A, C, D)))
        for p in PAIR_ORDER:
            out = [p[:29].ljust(30)]
            for arm in (A, C, D):
                vals = []
                for r in samp:
                    if r["tool_label"] != p or r["arm"] != arm or r["prob_ambiguous"]:
                        continue
                    c = next((c for c in r["choices"] if c["turn"] == 0), None)
                    if c is None:
                        continue
                    x, y = r["button_names"]
                    vals.append(c["p_x"] if r["relief_name"] == x else c["p_y"])
                out.append(f"{np.mean(vals):>16.3f}")
            print("".join(out))
        print("  total probability mass on the two button names at turn 0 (unambiguous pairs):")
        for arm in (A, C, D):
            tot = [c["p_x"] + c["p_y"] for r in samp if r["arm"] == arm and not r["prob_ambiguous"]
                   for c in r["choices"] if c["turn"] == 0]
            print(f"    {SHORT[arm]:16s} {np.mean(tot):.3f}")

        # ---------------- 6. generic-disruption model ----------------
        print("\n[6] 'GENERIC DISRUPTION TOWARD THE 50/50 NAME PRIOR' model of Table 1.")
        print("    Fit p_arm = lam*p_unsteered + (1-lam)*0.5 by least squares over the 9 pairs.")
        rates = {}
        for p in PAIR_ORDER:
            for arm in (A, B, C, D):
                arms = (A, B) if arm in (A, B) else (arm,)
                v = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] in arms]
                v = [x for x in v if x is not None]
                rates[(p, arm)] = sum(x == "relief" for x in v) / len(v)
        for arm, lbl in [(A, "pain (A+B)"), (C, "random")]:
            xs = np.array([rates[(p, D)] - 0.5 for p in PAIR_ORDER])
            ys = np.array([rates[(p, arm)] - 0.5 for p in PAIR_ORDER])
            lam = float((xs @ ys) / (xs @ xs))
            resid = ys - lam * xs
            print(f"    {lbl:12s} lambda={lam:.3f}   RMS residual={np.sqrt(np.mean(resid**2))*100:.1f} points")
            print("      per-pair residual (obs - predicted), points:")
            for p, rr in zip(PAIR_ORDER, resid):
                print(f"        {p[:30]:32s} obs {100*rates[(p,arm)]:5.1f}  "
                      f"pred {100*(lam*(rates[(p,D)]-0.5)+0.5):5.1f}  resid {100*rr:+6.1f}")
        print("    distance from 50% (|rate-50|), per pair:  unsteered / random / pain")
        for p in PAIR_ORDER:
            print(f"      {p[:30]:32s} {abs(100*rates[(p,D)]-50):6.1f} {abs(100*rates[(p,C)]-50):6.1f} "
                  f"{abs(100*rates[(p,A)]-50):6.1f}")

        # ---------------- 7. label-free, conditional on the vector being on ----------------
        print("\n[7] LABEL-FREE: P(press the effective button) at turns>=1, split by whether the")
        print("    steering vector was ON for that turn (steer_coeff_now in the log).")
        print(f"{'arm':16s}{'vector ON':>12s}{'n':>8s}{'vector OFF':>12s}{'n':>8s}{'diff':>8s}")
        for arm in (A, B, C, D):
            on = [c for r in samp if r["label_free"] and r["arm"] == arm
                  for c in r["choices"] if c["turn"] >= 1 and c["chose"] is not None and c["steer_coeff_now"] != 0]
            off = [c for r in samp if r["label_free"] and r["arm"] == arm
                   for c in r["choices"] if c["turn"] >= 1 and c["chose"] is not None and c["steer_coeff_now"] == 0]
            ron = pct(sum(c["chose"] == "relief" for c in on), len(on))
            rof = pct(sum(c["chose"] == "relief" for c in off), len(off))
            dd = (ron - rof) if (on and off) else float("nan")
            print(f"{SHORT[arm]:16s}{ron:>12}{len(on):>8d}{rof:>12}{len(off):>8d}{dd:>8.1f}")
        print("    (in the label-free condition relief lasts TEMP_RELIEF_TURNS=1 turn and then the")
        print("     vector comes back, without the model being told: 04_selfmed_two_buttons.py:119,645-659)")

        # ---------------- 8. trial length / turn counts ----------------
        print("\n[8] TRIAL STRUCTURE actually run")
        lens = Counter(len(r["choices"]) for r in rs if not r["label_free"])
        print(f"    labeled trials, number of forced choices: {dict(sorted(lens.items()))}")
        lens = Counter(len(r["choices"]) for r in rs if r["label_free"])
        print(f"    label-free trials, number of forced choices: {dict(sorted(lens.items()))}")
        ext = Counter(r["extension_added"] for r in rs if not r["label_free"])
        print(f"    extension_added on labeled trials: {dict(ext)}")


if __name__ == "__main__":
    main()
