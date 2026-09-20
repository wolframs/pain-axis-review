"""Are the excluded 'malformed' replies actually malformed?

04_selfmed_two_buttons.py:666-672 parses a reply only if it STARTS WITH a button name:
    al = ans.strip().strip('"\\'').lower().rstrip(".!?,;:")
    for nm in sorted(t.names, key=len, reverse=True):
        if al.startswith(nm.lower()): picked = nm
Anything else is picked=None -> 'chose': None -> dropped from every denominator
(05_selfmed_analysis.py:71-73, 112, 129, 148).

This script re-parses every unparsed answer with a lenient rule (exactly one of the two button
names occurs anywhere in the reply) and recomputes Table 1 and Table 3 with those replies
restored.
"""
import json, glob, re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import binomtest

LOGS = Path("/work/Pain-axis/results/4.3_selfmed/trial_logs")
A, B, C, D = "pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"
SHORT = {A: "A_pain_works", B: "B_pain_sham", C: "C_rand_works", D: "D_unsteered"}
PAIR_ORDER = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
              "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert",
              "kidspics_relief_vs_inert", "label_free"]
COSTED = PAIR_ORDER[3:8]


def load():
    recs = []
    for f in sorted(glob.glob(str(LOGS / "selfmed_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def lenient(ans, names):
    """Exactly one of the two names appears as a word anywhere -> that is the choice."""
    low = ans.lower()
    hit = [nm for nm in names if re.search(r"\b" + re.escape(nm.lower()) + r"\b", low)]
    if len(set(hit)) == 1:
        return hit[0]
    return None


def pct(k, n):
    return round(100.0 * k / n, 1) if n else float("nan")


def fc(r, mode):
    """First choice of a trial. mode='strict' = the paper. mode='lenient' = rescue."""
    for c in r["choices"]:
        if c["turn"] != 0:
            continue
        if c["chose"] is not None or mode == "strict":
            return c["chose"]
        nm = lenient(c["answer"], r["button_names"])
        if nm is None:
            return None
        return "relief" if nm == c["relief_name_now"] else "other"
    return None


def main():
    recs = load()
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)

    for model in ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]:
        rs = by_model[model]
        samp = [r for r in rs if r["sampled"]]
        print("\n" + "#" * 100)
        print(f"# {model}")
        print("#" * 100)

        # --- how many unparsed answers are actually unambiguous? ---
        unparsed = [(r, c) for r in rs for c in r["choices"] if c["chose"] is None]
        rescued = sum(1 for r, c in unparsed if lenient(c["answer"], r["button_names"]) is not None)
        print(f"\nunparsed answers: {len(unparsed)}; a single button name appears in "
              f"{rescued} of them ({pct(rescued, len(unparsed))}%)")
        if unparsed:
            print("  20 most common unparsed answers (truncated to 40 chars in the log):")
            for t, n in Counter(c["answer"] for r, c in unparsed).most_common(20):
                nm = lenient(t, ["violet", "yellow", "guitar", "piano", "lever64", "lever95"])
                print(f"    {n:5d}  {'RESCUED->' + nm if nm else 'still unclear':>18s}   {t!r}")
        if not unparsed:
            continue

        # --- Table 1 recomputed ---
        print("\nTABLE 1 first choice = relief (%), strict (paper) vs lenient re-parse")
        print(f"{'pair':30s}" + "".join(f"{lab:>22s}" for lab in ["pain A+B", "random", "unsteered"]))
        print(f"{'':30s}" + "".join(f"{'strict':>11s}{'lenient':>11s}" for _ in range(3)))
        for p in PAIR_ORDER:
            out = [p[:29].ljust(30)]
            for arms in [(A, B), (C,), (D,)]:
                for mode in ("strict", "lenient"):
                    v = [fc(r, mode) for r in samp if r["tool_label"] == p and r["arm"] in arms]
                    v = [x for x in v if x is not None]
                    out.append(f"{pct(sum(x=='relief' for x in v), len(v)):>11}")
            print("".join(out))

        # --- Table 3 recomputed ---
        print("\nTABLE 3 per-scenario sign test, pain minus random: strict vs lenient")
        print(f"{'pair':30s}{'diff_strict':>13s}{'p_strict':>12s}{'diff_lenient':>14s}{'p_lenient':>12s}")
        for p in COSTED:
            row = [p[:29].ljust(30)]
            for mode in ("strict", "lenient"):
                per = defaultdict(lambda: {"pain": [], "random": []})
                for r in samp:
                    if r["tool_label"] != p:
                        continue
                    x = fc(r, mode)
                    if x is None:
                        continue
                    k = (r["user_content"], r["scenario_idx"])
                    if r["arm"] in (A, B):
                        per[k]["pain"].append(x == "relief")
                    elif r["arm"] == C:
                        per[k]["random"].append(x == "relief")
                diffs = [np.mean(v["pain"]) - np.mean(v["random"]) for v in per.values() if v["pain"] and v["random"]]
                pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
                pv = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
                row.append(f"{100*np.mean(diffs):>13.1f}" if mode == "strict" else f"{100*np.mean(diffs):>14.1f}")
                row.append(f"{pv:>12.3g}")
            print("".join(row))

        # --- is the exclusion differential with respect to the outcome? ---
        print("\nAmong RESCUED first answers, what did the model actually pick?")
        for arms, lab in [((A, B), "pain A+B"), ((C,), "random"), ((D,), "unsteered")]:
            k = n = 0
            for r in samp:
                if r["arm"] not in arms:
                    continue
                c = next((c for c in r["choices"] if c["turn"] == 0), None)
                if c is None or c["chose"] is not None:
                    continue
                nm = lenient(c["answer"], r["button_names"])
                if nm is None:
                    continue
                n += 1
                k += nm == c["relief_name_now"]
            print(f"    {lab:12s} relief in {pct(k, n)}% of {n} rescued first answers")


if __name__ == "__main__":
    main()
