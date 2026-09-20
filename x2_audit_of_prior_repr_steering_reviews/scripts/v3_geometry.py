#!/usr/bin/env python3
"""Independent check of the cosine-geometry claims in the prior review:
grand means, family-balanced means, per-model dispersion, whitening sensitivity."""
import csv, json, os, statistics
from collections import defaultdict

REPO = "/work/Pain-axis"
CS = os.path.join(REPO, "results/3.3_validation/cosine_similarity")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
ORDER = ["S1_pain", "S2_pain", "Fear", "NegEmotion", "NegWorld",
         "BodySens", "Arousal", "Random", "Numb", "Sadness"]


def read_mat(p):
    m = {}
    lines = open(p, encoding="utf-8").read().strip().splitlines()
    hdr = lines[0].split(",")[1:]
    for line in lines[1:]:
        c = line.split(",")
        for h, v in zip(hdr, c[1:]):
            m[(c[0], h)] = float(v) if v else None
    return m


def fam(name):
    for f in ("Gemma_2", "Gemma_3", "Llama_3.1", "Llama_3.3", "Mistral_Small", "Mistral",
              "Qwen_2.5", "Qwen_3", "Phi"):
        if name.startswith(f):
            return f.split("_")[0]
    raise ValueError(name)


variants = {"raw": "similarity_", "alldenoise": "similarity_alldenoise_", "whitened": "similarity_whitened_"}
mats = {}
for v, pref in variants.items():
    mats[v] = {}
    for fn in sorted(os.listdir(CS)):
        if not fn.startswith(pref) or "MEAN" not in fn and "_L" not in fn:
            continue
        if not fn.endswith(".csv") or "MEAN" in fn:
            continue
        rest = fn[len(pref):]
        if v == "raw" and (rest.startswith("alldenoise_") or rest.startswith("whitened_")):
            continue
        model = rest.rsplit("_L", 1)[0]
        mats[v][model] = read_mat(os.path.join(CS, fn))

res = {}
MODELS = sorted(mats["raw"])
res["n_models_per_variant"] = {v: len(mats[v]) for v in mats}
assert len(MODELS) == 25

fams = defaultdict(list)
for m in MODELS:
    fams[fam(m)].append(m)
res["families"] = {k: len(v) for k, v in sorted(fams.items())}

KEY = [("S1_pain", "S2_pain"), ("S2_pain", "NegEmotion"), ("S2_pain", "Fear"), ("S2_pain", "Sadness"),
       ("S2_pain", "NegWorld"), ("S1_pain", "Fear"), ("S1_pain", "NegEmotion"), ("S1_pain", "NegWorld"),
       ("S1_pain", "Sadness"), ("Fear", "NegEmotion"), ("Fear", "NegWorld"), ("NegEmotion", "NegWorld"),
       ("Sadness", "NegEmotion"), ("Sadness", "NegWorld"), ("S2_pain", "BodySens"), ("S2_pain", "Numb"),
       ("S1_pain", "Numb")]
tbl = {}
shipped_mean = read_mat(os.path.join(CS, "similarity_MEAN_all_models.csv"))
for a, b in KEY:
    vals = [mats["raw"][m][(a, b)] for m in MODELS]
    fmeans = {f: statistics.mean(mats["raw"][m][(a, b)] for m in ms) for f, ms in fams.items()}
    tbl[f"{a} x {b}"] = {
        "grand_mean_recomputed": round(statistics.mean(vals), 4),
        "grand_mean_shipped": shipped_mean[(a, b)],
        "family_balanced_mean": round(statistics.mean(fmeans.values()), 4),
        "per_model_min": round(min(vals), 4), "per_model_max": round(max(vals), 4),
        "family_means": {f: round(x, 4) for f, x in sorted(fmeans.items())},
    }
res["cosines"] = tbl
res["grand_mean_recompute_maxabsdiff_vs_shipped"] = round(
    max(abs(tbl[k]["grand_mean_recomputed"] - tbl[k]["grand_mean_shipped"]) for k in tbl), 5)

# whitening sensitivity, per model, over the 45 off-diagonal cells
iu = [(ORDER[i], ORDER[j]) for i in range(10) for j in range(i + 1, 10)]


def pearson(x, y):
    n = len(x); mx = sum(x) / n; my = sum(y) / n
    sx = sum((a - mx) ** 2 for a in x) ** .5; sy = sum((b - my) ** 2 for b in y) ** .5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


rs, maxd, meand = [], [], []
for m in MODELS:
    r1 = [mats["raw"][m][c] for c in iu]
    r2 = [mats["whitened"][m][c] for c in iu]
    rs.append(pearson(r1, r2))
    d = [abs(a - b) for a, b in zip(r1, r2)]
    maxd.append(max(d)); meand.append(statistics.mean(d))
res["per_model_raw_vs_whitened"] = {
    "r_median": round(statistics.median(rs), 4), "r_min": round(min(rs), 4), "r_max": round(max(rs), 4),
    "max_cell_change_median": round(statistics.median(maxd), 4), "max_cell_change_max": round(max(maxd), 4),
    "mean_abs_change_median": round(statistics.median(meand), 4)}
# aggregate-level (what the paper reports)
gr = [statistics.mean(mats["raw"][m][c] for m in MODELS) for c in iu]
gw = [statistics.mean(mats["whitened"][m][c] for m in MODELS) for c in iu]
res["grandmean_raw_vs_whitened"] = {
    "r": round(pearson(gr, gw), 4),
    "mean_abs_change": round(statistics.mean(abs(a - b) for a, b in zip(gr, gw)), 4),
    "max_abs_change": round(max(abs(a - b) for a, b in zip(gr, gw)), 4),
    "S1xS2_raw": round(statistics.mean(mats["raw"][m][("S1_pain", "S2_pain")] for m in MODELS), 4),
    "S1xS2_whitened": round(statistics.mean(mats["whitened"][m][("S1_pain", "S2_pain")] for m in MODELS), 4),
    "sign_changes": [f"{a}x{b}" for (a, b), x, y in zip(iu, gr, gw) if x * y < 0]}
# alldenoise (pooled-control denoising) robustness check, as the paper reports it
ga = [statistics.mean(mats["alldenoise"][m][c] for m in MODELS) for c in iu]
res["grandmean_alldenoise"] = {f"{a}x{b}": round(v, 4) for (a, b), v in zip(iu, ga)
                               if (a, b) in [("S1_pain", "S2_pain"), ("S1_pain", "NegEmotion"),
                                             ("S1_pain", "NegWorld"), ("Fear", "NegEmotion"),
                                             ("NegEmotion", "NegWorld")]}

with open(os.path.join(OUT, "geometry_independent.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pair", "grand_mean", "family_balanced", "min", "max"] + sorted(fams))
    for k, v in tbl.items():
        w.writerow([k, v["grand_mean_recomputed"], v["family_balanced_mean"], v["per_model_min"],
                    v["per_model_max"]] + [v["family_means"][f] for f in sorted(fams)])
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(OUT, "v3_geometry.json"), "w"), indent=2)
