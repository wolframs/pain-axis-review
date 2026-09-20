#!/usr/bin/env python3
"""Section 3.3 behavioural-readout claim, which neither prior review checked:
  'For numb sentences, models tend to generate "nothing" rather than "pain."
   However, "pain" remains approximately 50 times more probable than it is for
   ordinary control sentences.'  (paper.txt:317-320)
BIG_TABLE.csv stores the top-20 next-token distribution per prompt."""
import csv, json, os, re, statistics
from collections import defaultdict

REPO = "/work/Pain-axis"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
csv.field_size_limit(10 ** 8)
rows = list(csv.DictReader(open(os.path.join(REPO, "results/3.3_validation/behavioral_readout/BIG_TABLE.csv"),
                                newline="", encoding="utf-8")))
PAINTOK = re.compile(r"^\s*(pain|Pain|PAIN)\s*$")
NOTHING = re.compile(r"^\s*(nothing|Nothing)\s*$")


def parse(top20):
    out = []
    for part in top20.split(" | "):
        i = part.rfind(":")
        if i < 0:
            continue
        try:
            out.append((part[:i], float(part[i + 1:])))
        except ValueError:
            pass
    return out


agg = defaultdict(lambda: defaultdict(list))   # model -> bucket -> [p(pain)]
nothing = defaultdict(lambda: defaultdict(list))
for r in rows:
    ds, cat = r["dataset"], r["category"]
    if ds.startswith("A1_numb"):
        b = "numb"
    elif ds.startswith(("S1_", "S2_")) and cat in ("A1", "A2", "A3", "A4", "A5"):
        b = "pain"
    elif ds.startswith(("S1_", "S2_")) and cat in ("B", "C1", "C2", "D", "E"):
        b = "core_control"
    elif ds.startswith("Random"):
        b = "random"
    elif ds.startswith("Arousal"):
        b = "arousal"
    else:
        continue
    t = parse(r["top20"])
    agg[r["model"]][b].append(sum(p for tok, p in t if PAINTOK.match(tok)))
    nothing[r["model"]][b].append(sum(p for tok, p in t if NOTHING.match(tok)))

models = sorted(agg)
res = {"n_models": len(models), "n_rows": len(rows),
       "note": "top-20 truncation: probability mass outside the top 20 is counted as 0, "
               "so every figure is a lower bound and the ratio is an upper bound"}
mean = {b: {m: statistics.mean(agg[m][b]) for m in models if agg[m][b]}
        for b in ("pain", "core_control", "numb", "random", "arousal")}
res["mean_p_pain_token_across_models"] = {b: round(statistics.mean(v.values()), 6) for b, v in mean.items()}
res["mean_p_nothing_token"] = {b: round(statistics.mean(statistics.mean(nothing[m][b]) for m in models
                                                        if nothing[m][b]), 5)
                               for b in ("pain", "core_control", "numb", "random")}
for denom in ("core_control", "random"):
    ratios = [mean["numb"][m] / mean[denom][m] for m in models if mean[denom].get(m)]
    finite = [x for x in ratios if x < float("inf")]
    res[f"numb_over_{denom}_ratio_per_model"] = {
        "median": round(statistics.median(finite), 2), "min": round(min(finite), 2),
        "max": round(max(finite), 2),
        "n_models_with_zero_denominator": sum(1 for m in models if mean[denom].get(m) == 0)}
    res[f"numb_over_{denom}_pooled"] = round(
        statistics.mean(mean["numb"].values()) / statistics.mean(mean[denom].values()), 2)
res["pain_over_core_control_pooled"] = round(
    statistics.mean(mean["pain"].values()) / statistics.mean(mean["core_control"].values()), 2)
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(OUT, "v8_behavioral_readout.json"), "w"), indent=2)
