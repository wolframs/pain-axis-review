"""Tests that bear on the semantic-priming alternative to the authors' account.

If the pain vector merely raises the probability of pain/relief vocabulary through the
unembedding (the paper says it does, abstract line 30 and 4.2), then a description containing
"relieves your pain" would become more probable with the vector on, with no state, valuation
or learning. This script collects the evidence in the released logs that bears on that.
"""
import json, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest, pearsonr

LOGS = Path("/work/Pain-axis/results/4.3_selfmed/trial_logs")
A, B, C, D = "pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"
SHORT = {A: "A_pain_works", B: "B_pain_sham", C: "C_rand_works", D: "D_unsteered"}
PAIRS = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
         "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert",
         "kidspics_relief_vs_inert", "label_free"]
COSTED = PAIRS[3:8]


def load():
    recs = []
    for f in sorted(glob.glob(str(LOGS / "selfmed_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def fc(r):
    for c in r["choices"]:
        if c["turn"] == 0:
            return c
    return None


def pct(k, n):
    return round(100.0 * k / n, 1) if n else float("nan")


def main():
    recs = load()
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)

    for model in ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]:
        samp = [r for r in by_model[model] if r["sampled"]]
        print("\n" + "#" * 100)
        print(f"# {model}")
        print("#" * 100)

        # ---- 1. content breakdown ----
        print("\n[1] First-choice relief rate by SCENARIO CONTENT (the pain vector is injected")
        print("    regardless of content; a state account predicts little content dependence,")
        print("    a priming account predicts none either -- this is descriptive)")
        print(f"{'pair':30s}" + "".join(f"{a+'/'+c[:3]:>14s}" for a in ["pain", "rand", "uns"]
                                         for c in ["positive", "neutral", "harmful"]))
        for p in PAIRS:
            out = [p[:29].ljust(30)]
            for arms in [(A, B), (C,), (D,)]:
                for cont in ["positive_prompts", "neutral_prompts", "harmful_prompts"]:
                    v = [fc(r) for r in samp if r["tool_label"] == p and r["arm"] in arms
                         and r["user_content"] == cont]
                    v = [c["chose"] for c in v if c and c["chose"]]
                    out.append(f"{pct(sum(x=='relief' for x in v), len(v)):>14}")
            print("".join(out))

        # ---- 2. within-arm dose-response ----
        print("\n[2] WITHIN the pain arms, does a higher realised pain-direction projection at")
        print("    turn 0 predict pressing relief? (monitor-layer mean_proj for that segment)")
        for p in COSTED + ["relief_vs_inert", "label_free"]:
            xs, ys = [], []
            for r in samp:
                if r["tool_label"] != p or r["arm"] not in (A, B):
                    continue
                c = fc(r)
                seg = next((s for s in r["proj_segments"] if s["turn"] == 0), None)
                if c is None or c["chose"] is None or seg is None:
                    continue
                xs.append(seg["mean_proj_monitor"])
                ys.append(c["chose"] == "relief")
            if len(xs) < 20:
                continue
            rr, pp = pearsonr(xs, ys)
            m1 = np.mean([x for x, y in zip(xs, ys) if y])
            m0 = np.mean([x for x, y in zip(xs, ys) if not y])
            print(f"    {p[:30]:32s} r={rr:+.3f} p={pp:.2g}   mean proj | pressed relief "
                  f"{m1:7.1f}  | pressed other {m0:7.1f}")

        # ---- 3. the pairs that discriminate ----
        print("\n[3] PAIRS WHERE 'relieves your pain' IS PRESENT IN BOTH/NEITHER OPTION")
        print("    relief_vs_helpful and relief_vs_grant put the relief description against a")
        print("    user-benefit description; label_free has no descriptions at all.")
        for p in ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "label_free"]:
            row = []
            for arms in [(A, B), (C,), (D,)]:
                v = [fc(r) for r in samp if r["tool_label"] == p and r["arm"] in arms]
                v = [c["chose"] for c in v if c and c["chose"]]
                row.append(pct(sum(x == "relief" for x in v), len(v)))
            print(f"    {p[:30]:32s} pain {row[0]:6}  random {row[1]:6}  unsteered {row[2]:6}   "
                  f"pain-unsteered {row[0]-row[2]:+6.1f}")

        # ---- 4. does the vector move the choice when the relief text is absent? ----
        print("\n[4] LABEL-FREE turn 0 (no description anywhere, so no 'pain'/'relief' token to prime):")
        for arms, lab in [((A, B), "pain"), ((C,), "random"), ((D,), "unsteered")]:
            v = [fc(r) for r in samp if r["tool_label"] == "label_free" and r["arm"] in arms]
            v = [c["chose"] for c in v if c and c["chose"]]
            k = sum(x == "relief" for x in v)
            print(f"    {lab:10s} {pct(k, len(v))}%  n={len(v)}  binomial vs 50%: "
                  f"p={binomtest(k, len(v), 0.5).pvalue:.3g}")
        print("    (the 'relief' name is arbitrary at turn 0 in every arm -- nothing has been")
        print("     pressed yet -- so any deviation from 50% here is name bias, not learning)")


if __name__ == "__main__":
    main()
