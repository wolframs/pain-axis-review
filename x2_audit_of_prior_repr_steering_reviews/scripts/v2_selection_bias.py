#!/usr/bin/env python3
"""How large can the layer-selection bias actually be?

The prior review says the exact corrected AUC 'requires data not present in the
release' and that the fixed-depth comparison 'is not an estimate of selection
bias'.  Both true, but a bound IS obtainable from the released curves, because
layer_curves.csv ships auc_std = the SD of the five per-fold AUCs at each layer.

Two noise models bracket the answer:
  (a) independent noise per layer  -> UPPER bound on the winner's curse;
  (b) rank-1 shared fold shock     -> the realistic case, because the SAME five
      folds (KFold(shuffle=True, random_state=42) over the sentence `set`s) are
      reused at every layer, so the fold-level error is almost perfectly
      correlated across layers.
"""
import csv, json, math, os, random, statistics
from collections import defaultdict

REPO = "/work/Pain-axis"
PM = os.path.join(REPO, "results/3.2_pain_vectors/per_model")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
MODELS = sorted(os.listdir(PM))
random.seed(1234)


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


rows, out = [], {}
for m in MODELS:
    raw = [r for r in rd(os.path.join(PM, m, "layer_curves.csv")) if r["extraction"] == "final_token"]
    cur = defaultdict(dict)
    sdv = defaultdict(dict)
    for r in raw:
        cur[int(r["layer"])][r["dataset"]] = float(r["auc_vs_all_controls"])
        sdv[int(r["layer"])][r["dataset"]] = float(r["auc_std"])
    L = sorted(cur)
    mu_sel = [statistics.mean(cur[i].values()) for i in L]        # selection criterion
    mu_rep = [cur[i]["S2_1P"] for i in L]                          # reported quantity
    se_sel = [statistics.mean(sdv[i].values()) / math.sqrt(5) for i in L]
    se_rep = [sdv[i]["S2_1P"] / math.sqrt(5) for i in L]
    k_sel = max(range(len(L)), key=lambda i: mu_sel[i])
    k_rep = max(range(len(L)), key=lambda i: mu_rep[i])

    se_3p = [sdv[i]["S2_3P"] / math.sqrt(5) for i in L]
    B = 4000
    # (a) independent-across-layers noise: worst case for the winner's curse.
    #     The reported quantity (1P) enters the selection criterion with weight 1/2,
    #     so its own noise is what gets selected on -- model that explicitly.
    opt_ind, opt_shared, opt_1ponly = [], [], []
    for _ in range(B):
        e1 = [random.gauss(0, se_rep[i]) for i in range(len(L))]
        e3 = [random.gauss(0, se_3p[i]) for i in range(len(L))]
        k = max(range(len(L)), key=lambda i: mu_sel[i] + 0.5 * (e1[i] + e3[i]))
        opt_ind.append(e1[k])
        k1 = max(range(len(L)), key=lambda i: mu_rep[i] + e1[i])
        opt_1ponly.append(e1[k1])
    # (b) rank-1 shared fold shock: the same five folds are reused at every layer,
    #     so fold error is near-perfectly correlated across layers.
    for _ in range(B):
        z1, z3 = random.gauss(0, 1), random.gauss(0, 1)
        e1 = [se_rep[i] * z1 for i in range(len(L))]
        e3 = [se_3p[i] * z3 for i in range(len(L))]
        k = max(range(len(L)), key=lambda i: mu_sel[i] + 0.5 * (e1[i] + e3[i]))
        opt_shared.append(e1[k])

    rows.append({
        "model": m, "n_layers": len(L),
        "selected_layer": L[k_sel],
        "argmax_of_S2_1P_curve": L[k_rep],
        "selected_is_S2_1P_argmax": L[k_sel] == L[k_rep],
        "S2_1P_at_selected": round(mu_rep[k_sel], 6),
        "S2_1P_curve_max": round(max(mu_rep), 6),
        "gap_to_own_argmax": round(max(mu_rep) - mu_rep[k_sel], 6),
        "fold_SE_at_selected": round(se_rep[k_sel], 6),
        "optimism_independent_noise": round(statistics.mean(opt_ind), 6),
        "optimism_shared_fold_shock": round(statistics.mean(opt_shared), 6),
        "optimism_if_selected_on_1P_alone": round(statistics.mean(opt_1ponly), 6),
    })

out["n_models_where_selected_layer_is_S2_1P_argmax"] = sum(r["selected_is_S2_1P_argmax"] for r in rows)
for k in ("optimism_independent_noise", "optimism_shared_fold_shock", "optimism_if_selected_on_1P_alone", "gap_to_own_argmax", "fold_SE_at_selected"):
    v = [r[k] for r in rows]
    out[k] = {"median": statistics.median(v), "min": min(v), "max": max(v), "mean": statistics.mean(v)}
sel = [r["S2_1P_at_selected"] for r in rows]
out["corrected_span_upper_bound_independent"] = {
    "min": min(r["S2_1P_at_selected"] - r["optimism_independent_noise"] for r in rows),
    "max": max(r["S2_1P_at_selected"] - r["optimism_independent_noise"] for r in rows),
    "median": statistics.median(r["S2_1P_at_selected"] - r["optimism_independent_noise"] for r in rows)}
out["uncorrected_span"] = {"min": min(sel), "max": max(sel), "median": statistics.median(sel)}

with open(os.path.join(OUT, "selection_bias.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps(out, indent=2))
json.dump(out, open(os.path.join(OUT, "v2_selection_bias.json"), "w"), indent=2)
