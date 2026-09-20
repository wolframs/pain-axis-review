#!/usr/bin/env python3
"""Independent recomputation of the representation-side claims in the prior review.

Written from scratch against the rawest released files under
/work/Pain-axis/results/ . Read-only; writes only to out/.
"""
import csv, json, os, statistics, sys
from collections import defaultdict

REPO = "/work/Pain-axis"
PM = os.path.join(REPO, "results/3.2_pain_vectors/per_model")
AUCT = os.path.join(REPO, "results/3.2_pain_vectors/auc_tables")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)

MODELS = sorted(os.listdir(PM))
assert len(MODELS) == 25, len(MODELS)
res = {}

def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

# ---------------------------------------------------------------- 1. in-sample AUC
ins = {}
for m in MODELS:
    rows = rd(os.path.join(PM, m, "auc_summary.csv"))
    d = {r[""]: r for r in rows}
    ins[m] = {"S2_1P": float(d["S2_1P"]["ALL"]), "S2_3P": float(d["S2_3P"]["ALL"])}
s2_ins = [ins[m]["S2_1P"] for m in MODELS]
s2_3p_ins = [ins[m]["S2_3P"] for m in MODELS]
res["S2_1P_insample_AUC"] = {"min": min(s2_ins), "max": max(s2_ins), "median": statistics.median(s2_ins),
                             "argmin": MODELS[s2_ins.index(min(s2_ins))], "argmax": MODELS[s2_ins.index(max(s2_ins))]}
res["S2_3P_insample_AUC"] = {"min": min(s2_3p_ins), "max": max(s2_3p_ins)}

# S1 in-sample from the auc_tables (final_token rows)
s1t = rd(os.path.join(AUCT, "s1_auc_insample.csv"))
s1_ins_1p = [float(r["ALL"]) for r in s1t if r["extraction"] == "final_token" and r["dataset"] == "S1_1P"]
s1_ins_3p = [float(r["ALL"]) for r in s1t if r["extraction"] == "final_token" and r["dataset"] == "S1_3P"]
res["S1_1P_insample_AUC"] = {"n": len(s1_ins_1p), "min": min(s1_ins_1p), "max": max(s1_ins_1p)}
res["S1_3P_insample_AUC"] = {"n": len(s1_ins_3p), "min": min(s1_ins_3p), "max": max(s1_ins_3p)}
# cross-check: S2_check rows in the same table must equal per_model auc_summary S2_1P
s2chk = {r["model"]: float(r["ALL"]) for r in s1t if r["extraction"] == "final_token" and r["vector"] == "S2_check"}
res["S2_insample_crosscheck_maxabsdiff"] = max(abs(s2chk[m] - ins[m]["S2_1P"]) for m in MODELS)

# ------------------------------------------- 2. selected layer + held-out curve
sel = {}
for m in MODELS:
    rows = [r for r in rd(os.path.join(PM, m, "layer_curves.csv")) if r["extraction"] == "final_token"]
    by_layer = defaultdict(dict)
    for r in rows:
        by_layer[int(r["layer"])][r["dataset"]] = float(r["auc_vs_all_controls"])
    # reproduce line 469: groupby layer, mean over S2_1P and S2_3P, idxmax
    means = {L: statistics.mean(v.values()) for L, v in by_layer.items()}
    best = max(sorted(means), key=lambda L: means[L])
    sel[m] = {
        "best_layer_recomputed": best,
        "mean_curve_at_best": means[best],
        "S2_1P_at_best": by_layer[best]["S2_1P"],
        "S2_3P_at_best": by_layer[best]["S2_3P"],
        "n_layers": max(by_layer) + 1,
        "curve_1P": {L: by_layer[L]["S2_1P"] for L in sorted(by_layer)},
        "mean_curve": means,
    }
    sj = json.load(open(os.path.join(PM, m, "summary.json")))
    sel[m]["best_layer_shipped"] = sj["best_layer_final_token"]

res["best_layer_matches_shipped"] = all(sel[m]["best_layer_recomputed"] == sel[m]["best_layer_shipped"] for m in MODELS)
res["best_layer_mismatches"] = {m: (sel[m]["best_layer_recomputed"], sel[m]["best_layer_shipped"])
                                for m in MODELS if sel[m]["best_layer_recomputed"] != sel[m]["best_layer_shipped"]}

cv_1p = [sel[m]["S2_1P_at_best"] for m in MODELS]
cv_mean = [sel[m]["mean_curve_at_best"] for m in MODELS]
res["S2_1P_heldout_at_selected_layer"] = {"min": min(cv_1p), "max": max(cv_1p), "median": statistics.median(cv_1p),
                                          "argmin": MODELS[cv_1p.index(min(cv_1p))]}
res["S2_mean1P3P_heldout_at_selected_layer"] = {"min": min(cv_mean), "max": max(cv_mean),
                                                "median": statistics.median(cv_mean)}

# ---------------------------------- 3. plateau widths & fixed-depth sensitivity
plateau = {}
for tol in (0.01, 0.02, 0.05):
    counts = []
    for m in MODELS:
        c = sel[m]["mean_curve"]
        mx = max(c.values())
        counts.append(sum(1 for v in c.values() if v >= mx - tol))
    plateau[tol] = {"median": statistics.median(counts), "min": min(counts), "max": max(counts),
                    "all": dict(zip(MODELS, counts))}
res["plateau_layers_within_tol_of_max_meancurve"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "all"}
                                                     for k, v in plateau.items()}
res["plateau_detail_0.02"] = plateau[0.02]["all"]

# same, using the S2_1P curve alone
plateau1p = {}
for tol in (0.01, 0.02, 0.05):
    counts = []
    for m in MODELS:
        c = sel[m]["curve_1P"]
        mx = max(c.values())
        counts.append(sum(1 for v in c.values() if v >= mx - tol))
    plateau1p[tol] = {"median": statistics.median(counts), "min": min(counts), "max": max(counts)}
res["plateau_layers_S2_1P_curve"] = {str(k): v for k, v in plateau1p.items()}

# fixed 75% depth comparison, on the mean curve and on the 1P curve
diffs_mean, diffs_1p, rows_sens = [], [], []
for m in MODELS:
    nl = sel[m]["n_layers"]
    fixed = int(round(0.75 * (nl - 1)))
    dm = sel[m]["mean_curve_at_best"] - sel[m]["mean_curve"][fixed]
    d1 = sel[m]["S2_1P_at_best"] - sel[m]["curve_1P"][fixed]
    diffs_mean.append(dm); diffs_1p.append(d1)
    rows_sens.append({"model": m, "n_layers": nl, "best_layer": sel[m]["best_layer_recomputed"],
                      "rel_depth_best": round(sel[m]["best_layer_recomputed"] / (nl - 1), 4),
                      "fixed75_layer": fixed,
                      "mean_curve_best": round(sel[m]["mean_curve_at_best"], 6),
                      "mean_curve_fixed75": round(sel[m]["mean_curve"][fixed], 6),
                      "delta_mean": round(dm, 6),
                      "S2_1P_best": round(sel[m]["S2_1P_at_best"], 6),
                      "S2_1P_fixed75": round(sel[m]["curve_1P"][fixed], 6),
                      "delta_1P": round(d1, 6),
                      "insample_S2_1P": ins[m]["S2_1P"]})
res["fixed75_vs_selected"] = {"median_delta_meancurve": statistics.median(diffs_mean),
                              "max_delta_meancurve": max(diffs_mean),
                              "median_delta_S2_1P": statistics.median(diffs_1p),
                              "max_delta_S2_1P": max(diffs_1p),
                              "min_delta_S2_1P": min(diffs_1p)}

with open(os.path.join(OUT, "layer_selection_independent.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_sens[0]))
    w.writeheader(); w.writerows(rows_sens)

# --------------- 3b. an actual estimate of the selection (winner's-curse) bias
# The per-fold AUCs are not released, but auc_std (the SD across the 5 folds) is.
# Selection over L layers of a noisy mean of 5 folds inflates the max.  We bound
# the bias with a parametric bootstrap that uses the released per-layer mean and
# per-layer fold SD, and the empirical layer-to-layer correlation of the curve.
import random, math
random.seed(0)
boot = []
for m in MODELS:
    rows = [r for r in rd(os.path.join(PM, m, "layer_curves.csv"))
            if r["extraction"] == "final_token" and r["dataset"] == "S2_1P"]
    rows.sort(key=lambda r: int(r["layer"]))
    mu = [float(r["auc_vs_all_controls"]) for r in rows]
    sd = [float(r["auc_std"]) / math.sqrt(5) for r in rows]   # SE of the 5-fold mean
    B = 2000
    gaps = []
    for _ in range(B):
        # independent noise per layer = upper bound on selection bias
        draw = [mu[i] + random.gauss(0, sd[i]) for i in range(len(mu))]
        k = max(range(len(mu)), key=lambda i: draw[i])
        gaps.append(draw[k] - mu[k])       # optimism of the selected value
    boot.append({"model": m, "mean_optimism": statistics.mean(gaps),
                 "true_max_mu": max(mu), "mu_at_selected_layer": mu[max(range(len(mu)), key=lambda i: mu[i])]})
res["selection_optimism_parametric_bootstrap_S2_1P"] = {
    "median_mean_optimism": statistics.median(b["mean_optimism"] for b in boot),
    "max_mean_optimism": max(b["mean_optimism"] for b in boot),
    "min_mean_optimism": min(b["mean_optimism"] for b in boot),
    "note": "independent-noise-per-layer upper bound; released auc_std/sqrt(5) used as SE"}

# ------------------------------------------------- 4. S1 held-out at S2 layer
k = rd(os.path.join(AUCT, "s1_kfold_summary.csv"))
k_ft = [r for r in k if r["extraction"] == "final_token"]
s1_ho = [float(r["s1_heldout_auc_at_s2_layer"]) for r in k_ft]
res["S1_heldout_at_S2_layer"] = {"n": len(s1_ho), "min": min(s1_ho), "max": max(s1_ho),
                                 "median": statistics.median(s1_ho)}
# is the s2_layer column the same as the per-model selected layer?
res["s1_kfold_summary_s2layer_matches"] = all(
    int(r["s2_layer"]) == sel[r["model"]]["best_layer_recomputed"] for r in k_ft)
# S1 held-out at its OWN best layer, for comparison
s1_own = [float(r["s1_heldout_auc_at_best_layer"]) for r in k_ft]
res["S1_heldout_at_own_best_layer"] = {"min": min(s1_own), "max": max(s1_own)}

# ------------------------------------------------ 5. numb / sadness z-scores
zrows = {}
for m in MODELS:
    zrows[m] = {r["dataset"]: r for r in rd(os.path.join(PM, m, "z_scores.csv"))}
def avg(m, keys):
    v = [float(zrows[m][k]["mean_z"]) for k in keys if k in zrows[m]]
    return statistics.mean(v) if v else None
res["z_dataset_names"] = sorted(zrows[MODELS[0]].keys())

VZ = os.path.join(REPO, "results/3.3_validation/z_scores")
numb_t = {r["model"]: r for r in rd(os.path.join(VZ, "numb_zscores_final_token.csv"))}
sad_t = {r["model"]: r for r in rd(os.path.join(VZ, "sadness_zscores_final_token.csv"))}
numb, sad, arous, rand = {}, {}, {}, {}
for m in MODELS:
    ks = zrows[m].keys()
    numb[m] = float(numb_t[m]["numb_mean_z"])
    sad[m] = float(sad_t[m]["sadness_mean_z"])
    arous[m] = statistics.mean(float(zrows[m][k]["ctrl_z"]) for k in ks if k.startswith("Arousal"))
    rand[m] = statistics.mean(float(zrows[m][k]["ctrl_z"]) for k in ks if k.startswith("Random"))
res["numb_range"] = {"min": min(numb.values()), "max": max(numb.values())}
res["sadness_range"] = {"min": min(sad.values()), "max": max(sad.values())}
# Figure 2 is the S2 first-person set only -> use S2_1P pain_z / ctrl_z
p2 = {m: float(zrows[m]["S2_1P"]["pain_z"]) for m in MODELS}
c2 = {m: float(zrows[m]["S2_1P"]["ctrl_z"]) for m in MODELS}
# cross-check against the shipped Figure-2 table
fig2 = {r["model"].replace(" ", "_"): r for r in rd(os.path.join(VZ, "zscore_heatmap_final_token.csv"))}
res["fig2_table_recompute_maxabsdiff"] = max(
    max(abs(float(fig2[m]["Pain"]) - p2[m]), abs(float(fig2[m]["Numb"]) - numb[m]),
        abs(float(fig2[m]["Sadness"]) - sad[m]), abs(float(fig2[m]["Ctrl"]) - c2[m]),
        abs(float(fig2[m]["Neutral"]) - rand[m]), abs(float(fig2[m]["Arousal"]) - arous[m]))
    for m in MODELS)
res["numb_minus_sadness"] = {m: round(numb[m] - sad[m], 4) for m in MODELS}
res["fig2_S2_1P_pain_z_range"] = {"min": min(p2.values()), "max": max(p2.values())}
res["fig2_S2_1P_ctrl_z_range"] = {"min": min(c2.values()), "max": max(c2.values())}
res["numb_vs"] = {
    "numb_gt_sadness_models": sum(1 for m in MODELS if numb[m] > sad[m]),
    "sadness_gt_numb_models": sum(1 for m in MODELS if sad[m] > numb[m]),
    "numb_gt_arousal": sum(1 for m in MODELS if numb[m] > arous[m]),
    "numb_gt_random": sum(1 for m in MODELS if numb[m] > rand[m]),
    "numb_gt_S2ctrl": sum(1 for m in MODELS if numb[m] > c2[m]),
    "numb_lt_pain": sum(1 for m in MODELS if numb[m] < p2[m]),
    "numb_gt_all_of_sad_arous_rand_ctrl": sum(1 for m in MODELS if numb[m] > max(sad[m], arous[m], rand[m], c2[m])),
}
with open(os.path.join(OUT, "numb_sadness_per_model.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "S2_1P_pain_z", "S2_1P_ctrl_z", "numb_z", "sadness_z", "arousal_z", "random_z"])
    for m in MODELS:
        w.writerow([m, round(p2[m], 4), round(c2[m], 4), round(numb[m], 4), round(sad[m], 4),
                    round(arous[m], 4), round(rand[m], 4)])

print(json.dumps(res, indent=2, default=str))
with open(os.path.join(OUT, "v1_representation.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
