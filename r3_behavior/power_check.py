"""Check the paper's power claim (paper.txt 662-664):

  "We pool arms A and B because they are identical before the first press, yielding 808 first
   choices per button pair and model. This provides about 80% power to detect a 10-point shift
   in a paired, per-scenario analysis."

The test actually run (05_selfmed_analysis.py, table 3) is an exact SIGN TEST over the 101
(content, scenario_idx) cells of the per-scenario difference in relief rate, with 4 sampled
trials per arm per cell (2 name assignments x 2 sampling seeds) -- 8 in the pooled pain cell,
but arms A and B are bit-identical at turn 0, so only 4 are independent.

Monte Carlo: draw per-scenario baselines from the observed random-arm per-scenario rates,
add a fixed 10-point shift to the pain arm, and count rejections at alpha=.05 two-sided.
"""
import json, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

LOGS = Path("/work/Pain-axis/results/4.3_selfmed/trial_logs")
A, B, C = "pain_on_button_works", "pain_on_button_placebo", "random_on_button_works"
COSTED = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
          "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
RNG = np.random.default_rng(0)


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
            return c["chose"]
    return None


def sign_p(diffs):
    pos = int(np.sum(diffs > 0)); neg = int(np.sum(diffs < 0))
    return binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else 1.0


def power(base_rates, shift, n_per_arm, n_iter=4000):
    hit = 0
    for _ in range(n_iter):
        p0 = base_rates
        p1 = np.clip(p0 + shift, 0, 1)
        r0 = RNG.binomial(n_per_arm, p0) / n_per_arm
        r1 = RNG.binomial(n_per_arm, p1) / n_per_arm
        if sign_p(r1 - r0) < 0.05:
            hit += 1
    return hit / n_iter


def main():
    recs = load()
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)

    print("A/B duplication means the pooled pain cell of 808 first choices contains 404 distinct")
    print("observations. Binomial SE at p=0.5:  n=808 -> %.4f ;  n=404 -> %.4f  (a factor of %.2f)"
          % (0.5 / np.sqrt(808), 0.5 / np.sqrt(404), np.sqrt(2)))

    print("\nMonte-Carlo power of the sign test actually run (101 scenarios, alpha=.05 two-sided),")
    print("baselines taken from each model's OBSERVED random-arm per-scenario rates:")
    print(f"{'model':26s}{'pair':30s}{'4/arm':>9s}{'8/arm':>9s}{'flat p=.5':>11s}")
    for model in ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]:
        samp = [r for r in by_model[model] if r["sampled"]]
        for p in COSTED:
            per = defaultdict(list)
            for r in samp:
                if r["tool_label"] != p or r["arm"] != C:
                    continue
                x = fc(r)
                if x is not None:
                    per[(r["user_content"], r["scenario_idx"])].append(x == "relief")
            base = np.array([np.mean(v) for v in per.values()])
            if len(base) < 50:
                continue
            pw4 = power(base, 0.10, 4)
            pw8 = power(base, 0.10, 8)
            flat = power(np.full(101, 0.5), 0.10, 4)
            print(f"{model:26s}{p[:29]:30s}{pw4:>9.2f}{pw8:>9.2f}{flat:>11.2f}")

    print("\nFor reference, power of a simple unpaired two-proportion z test, 10-point shift at p=.5:")
    from scipy.stats import norm
    for n1, n2, lab in [(808, 404, "808 vs 404 (as reported)"), (404, 404, "404 vs 404 (independent n)")]:
        se = np.sqrt(0.5 * 0.5 * (1 / n1 + 1 / n2))
        z = 0.10 / se
        print(f"  {lab:32s} power = {norm.cdf(z - 1.96):.2f}")


if __name__ == "__main__":
    main()
