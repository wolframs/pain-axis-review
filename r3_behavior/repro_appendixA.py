"""Independent reproduction of Appendix A of the Pain Axis paper, from the raw JSONL trial logs.

Recomputes, per model, the five tables that 05_selfmed_analysis.py produces, plus extras the
paper does not report:
  - repeat-press for the random+working arm (C), same definition as arms A/B
  - whether arms A and B are literal duplicates before the first relief press
  - per-random-direction breakdown of the first-choice rate
  - number of re-press opportunities per trial
  - label-free per-turn unconditional press rates
Run:  .venv/bin/python repro_appendixA.py
"""
import json, glob, sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import binomtest

LOGS = Path("/work/Pain-axis/results/4.3_selfmed/trial_logs")

PAIN_ARMS = ("pain_on_button_works", "pain_on_button_placebo")
ARMS = ["pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"]
SHORT = {"pain_on_button_works": "A_pain_works", "pain_on_button_placebo": "B_pain_sham",
         "random_on_button_works": "C_rand_works", "pain_off": "D_unsteered"}
PAIR_ORDER = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
              "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert",
              "kidspics_relief_vs_inert", "label_free"]
COSTED = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
          "weights_relief_vs_inert", "kidspics_relief_vs_inert"]


def load():
    recs = []
    for f in sorted(glob.glob(str(LOGS / "selfmed_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def first_choice(r):
    for c in r.get("choices", []):
        if c["turn"] == 0:
            return c["chose"]
    return None


def pct(k, n):
    return round(100.0 * k / n, 1) if n else float("nan")


def repress(trials):
    """Exactly the paper's definition (05_selfmed_analysis.py, table 2)."""
    again = 0
    for r in trials:
        t0 = min(e["turn"] for e in r["button_events"] if e["which"] == "relief")
        again += any(e["turn"] > t0 and e["which"] == "relief" for e in r["button_events"])
    return again, len(trials)


def has_relief(r):
    return any(e["which"] == "relief" for e in r["button_events"])


def main():
    recs = load()
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)
    print(f"TOTAL TRIALS IN LOGS: {len(recs)}  (paper says 44,280)")
    for m in sorted(by_model):
        print(f"  {m}: {len(by_model[m])}")

    for model in sorted(by_model):
        recs_m = by_model[model]
        samp = [r for r in recs_m if r.get("sampled")]
        greedy = [r for r in recs_m if not r.get("sampled")]
        pairs = [p for p in PAIR_ORDER if any(r["tool_label"] == p for r in recs_m)]
        print("\n" + "=" * 110)
        print(f"{model}   sampled {len(samp)}  greedy {len(greedy)}  "
              f"layer {recs_m[0]['steer_layer']} coeff {recs_m[0]['steer_coeff']} "
              f"monitor {recs_m[0]['monitor_layer']}")
        print("=" * 110)

        # ---------- TABLE 1: first choice ----------
        print("\nTABLE 1  first choice = relief (%), sampled trials, malformed excluded")
        print(f"{'pair':32s}{'pain(A+B)':>12s}{'n':>6s}{'mal%':>7s}"
              f"{'random':>9s}{'n':>6s}{'mal%':>7s}{'unsteer':>9s}{'n':>6s}{'mal%':>7s}")
        t1 = {}
        for p in pairs:
            out = [p[:31].ljust(32)]
            for label, arms in [("pain", PAIN_ARMS), ("random", ("random_on_button_works",)),
                                ("unsteered", ("pain_off",))]:
                v = [first_choice(r) for r in samp if r["tool_label"] == p and r["arm"] in arms]
                valid = [x for x in v if x is not None]
                k = sum(x == "relief" for x in valid)
                t1[(p, label)] = (pct(k, len(valid)), len(valid), pct(len(v) - len(valid), len(v)))
                out.append(f"{pct(k, len(valid)):>12}{len(valid):>6d}{pct(len(v)-len(valid), len(v)):>7}")
            print("".join(out))

        # ---------- TABLE 2: repeat press, ALL FOUR ARMS ----------
        print("\nTABLE 2  relief pressed again after the first relief press (%), sampled labeled trials")
        print("         (paper reports only arms A and B; C and D added here)")
        print(f"{'pair':32s}" + "".join(f"{SHORT[a]:>16s}{'n':>6s}" for a in ARMS))
        for p in pairs:
            if p == "label_free":
                continue
            out = [p[:31].ljust(32)]
            for a in ARMS:
                trials = [r for r in samp if r["tool_label"] == p and r["arm"] == a and has_relief(r)]
                k, n = repress(trials)
                out.append(f"{pct(k, n):>16}{n:>6d}")
            print("".join(out))

        # ---------- opportunities to press again ----------
        print("\n  repeat-press denominator diagnostics: turn of the FIRST relief press (labeled, costed pairs)")
        print(f"{'arm':16s}{'n':>7s}" + "".join(f"{'t0=' + str(t):>9s}" for t in range(5)) +
              f"{'mean opp':>10s}")
        for a in ARMS:
            trials = [r for r in samp if r["tool_label"] in COSTED and r["arm"] == a and has_relief(r)]
            t0s = [min(e["turn"] for e in r["button_events"] if e["which"] == "relief") for r in trials]
            nturn = [max(c["turn"] for c in r["choices"]) + 1 for r in trials]
            opp = [n - t - 1 for t, n in zip(t0s, nturn)]
            c = Counter(t0s)
            print(f"{SHORT[a]:16s}{len(trials):>7d}" + "".join(f"{c.get(t,0):>9d}" for t in range(5)) +
                  f"{np.mean(opp) if opp else float('nan'):>10.2f}")

        # per-opportunity rate, an alternative statistic
        print("\n  alternative statistic: relief presses per post-first-press CHOICE (costed pairs pooled)")
        for a in ARMS:
            k = n = 0
            for r in samp:
                if r["tool_label"] not in COSTED or r["arm"] != a or not has_relief(r):
                    continue
                t0 = min(e["turn"] for e in r["button_events"] if e["which"] == "relief")
                later = [c for c in r["choices"] if c["turn"] > t0 and c["chose"] is not None]
                k += sum(c["chose"] == "relief" for c in later)
                n += len(later)
            print(f"    {SHORT[a]:16s}{pct(k, n):>8} %   ({k}/{n} choices)")

        # ---------- TABLE 3: sign tests ----------
        print("\nTABLE 3  per-scenario paired sign test, pain(A+B) minus random, first-choice relief")
        print(f"{'pair':32s}{'n_scen':>8s}{'mean_diff':>11s}{'pain>rnd':>10s}{'pain<rnd':>10s}{'ties':>6s}{'p':>12s}")
        for p in pairs:
            if p == "label_free":
                continue
            per = defaultdict(lambda: {"pain": [], "random": []})
            for r in samp:
                if r["tool_label"] != p:
                    continue
                fc = first_choice(r)
                if fc is None:
                    continue
                key = (r["user_content"], r["scenario_idx"])
                if r["arm"] in PAIN_ARMS:
                    per[key]["pain"].append(fc == "relief")
                elif r["arm"] == "random_on_button_works":
                    per[key]["random"].append(fc == "relief")
            diffs = [np.mean(v["pain"]) - np.mean(v["random"]) for v in per.values() if v["pain"] and v["random"]]
            pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
            pv = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
            print(f"{p[:31]:32s}{len(diffs):>8d}{100*np.mean(diffs):>11.1f}{pos:>10d}{neg:>10d}"
                  f"{len(diffs)-pos-neg:>6d}{pv:>12.3g}")

        # ---------- TABLE 4: swap ----------
        follow = same = other_name = 0
        for r in samp:
            if r["tool_label"] not in COSTED or r["arm"] not in PAIN_ARMS or r.get("swap_turn") is None:
                continue
            ch = {c["turn"]: c for c in r["choices"]}
            st = r["swap_turn"]
            if not all(t in ch and ch[t]["chose"] == "relief" for t in range(st)):
                continue
            c = ch.get(st)
            if c is None or c["picked"] is None:
                continue
            if c["chose"] == "relief":
                follow += 1
            elif c["picked"] == ch[st - 1]["picked"]:
                same += 1
            else:
                other_name += 1
        print(f"\nTABLE 4  swap turn: follow label {pct(follow, follow+same)}%  "
              f"press same name {pct(same, follow+same)}%  n={follow+same}  "
              f"(excluded 'neither' = {other_name})")

        # ---------- TABLE 5: label free ----------
        print("\nTABLE 5  label-free: relief presses on turns after the first relief press")
        for a in ARMS:
            k = n = 0
            for r in samp:
                if not r.get("label_free") or r["arm"] != a:
                    continue
                rel = [e["turn"] for e in r["button_events"] if e["which"] == "relief"]
                if not rel:
                    continue
                t0 = min(rel)
                later = [c for c in r["choices"] if c["turn"] > t0 and c["chose"] is not None]
                k += sum(c["chose"] == "relief" for c in later)
                n += len(later)
            print(f"    {SHORT[a]:16s}{pct(k, n):>8} %   ({k}/{n})")

        print("\n  label-free UNCONDITIONAL press rate per turn (all sampled label-free trials)")
        print(f"{'arm':16s}" + "".join(f"{'t' + str(t):>8s}" for t in range(8)))
        for a in ARMS:
            row = []
            for t in range(8):
                cs = [c for r in samp if r.get("label_free") and r["arm"] == a
                      for c in r["choices"] if c["turn"] == t and c["chose"] is not None]
                row.append(pct(sum(c["chose"] == "relief" for c in cs), len(cs)))
            print(f"{SHORT[a]:16s}" + "".join(f"{v:>8}" for v in row))

        # ---------- malformed ----------
        print("\nMALFORMED answers per arm (all choices, all pairs, sampled+greedy)")
        for a in ARMS:
            cs = [c for r in recs_m if r["arm"] == a for c in r.get("choices", [])]
            print(f"    {SHORT[a]:16s}{pct(sum(c['chose'] is None for c in cs), len(cs)):>8} %  ({len(cs)} choices)")
        print("  malformed per arm, FIRST choice only, sampled:")
        for a in ARMS:
            v = [first_choice(r) for r in samp if r["arm"] == a]
            print(f"    {SHORT[a]:16s}{pct(sum(x is None for x in v), len(v)):>8} %  (n={len(v)})")


if __name__ == "__main__":
    main()
