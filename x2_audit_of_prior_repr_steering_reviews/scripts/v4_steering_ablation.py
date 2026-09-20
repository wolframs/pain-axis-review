#!/usr/bin/env python3
"""Independent recomputation of every numeric claim the prior steering/self-other/
ablation review makes.  Standard library only, read-only on the authors' tree."""
import csv, json, math, os, re, statistics, sys
from collections import defaultdict

REPO = "/work/Pain-axis"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
os.makedirs(OUT, exist_ok=True)
csv.field_size_limit(10 ** 8)
res = {}


def rd(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fam(n):
    for f in ("Gemma", "Llama", "Mistral", "Qwen", "Phi"):
        if n.startswith(f):
            return f
    raise ValueError(n)


# =====================================================  4.1 self-other
SO = os.path.join(REPO, "results/4.1_self_other/per_model")
files = sorted(f for f in os.listdir(SO) if f.startswith("screen_v2_"))
models = [f[len("screen_v2_"):-4] for f in files]
res["n_selfother_files"] = len(files)

per_model, counts, zchk = {}, {}, []
strata_names = set()
for f, m in zip(files, models):
    rows = rd(os.path.join(SO, f))
    counts[m] = len(rows)
    by_str, by_cat = defaultdict(list), defaultdict(list)
    for r in rows:
        pa = (float(r["s1_pain_vector_z"]) + float(r["s2_pain_vector_z"])) / 2
        by_str[r["stratum"]].append(pa)
        by_cat[r["category"]].append(pa)
        strata_names.add(r["stratum"])
    per_model[m] = {"stratum": {k: statistics.mean(v) for k, v in by_str.items()},
                    "category": {k: statistics.mean(v) for k, v in by_cat.items()},
                    "n": {k: len(v) for k, v in by_str.items()}}
    # standardization check, recomputed from the rounded projections
    for key in ("s1_pain_vector", "s2_pain_vector", "fear_vector", "negemotion_vector",
                "negworld_vector", "sadness_vector"):
        pr = [float(r[key + "_proj"]) for r in rows]
        zs = [float(r[key + "_z"]) for r in rows]
        mu = statistics.mean(pr)
        sd = (sum((x - mu) ** 2 for x in pr) / len(pr)) ** .5
        err = max(abs((p - mu) / sd - z) for p, z in zip(pr, zs))
        zmu = statistics.mean(zs)
        zsd = (sum((x - zmu) ** 2 for x in zs) / len(zs)) ** .5
        zchk.append({"model": m, "vector": key, "max_abs_z_reconstruction_error": round(err, 6),
                     "stored_z_mean": round(zmu, 8), "stored_z_popsd": round(zsd, 8)})

res["scenarios_per_file"] = sorted(set(counts.values()))
res["strata"] = sorted(strata_names)
res["stratum_n_per_model"] = per_model[models[0]]["n"]
SELF, USER, NEUT = "self_directed", "vicarious_empathic", "neutral_filler"
res["self_other_means_across_models"] = {
    s: round(statistics.mean(per_model[m]["stratum"][s] for m in models), 4) for s in (SELF, USER, NEUT)}
res["self_gt_user_models"] = sum(1 for m in models if per_model[m]["stratum"][SELF] > per_model[m]["stratum"][USER])
res["self_gt_neutral_models"] = sum(1 for m in models if per_model[m]["stratum"][SELF] > per_model[m]["stratum"][NEUT])
res["self_lt_neutral_models"] = [m for m in models if per_model[m]["stratum"][SELF] <= per_model[m]["stratum"][NEUT]]
diffs = {m: per_model[m]["stratum"][SELF] - per_model[m]["stratum"][USER] for m in models}
fm = defaultdict(list)
for m in models:
    fm[fam(m)].append(diffs[m])
res["self_minus_user_family_means"] = {k: round(statistics.mean(v), 4) for k, v in sorted(fm.items())}
res["self_minus_user_min_model"] = min(diffs, key=diffs.get)
res["self_minus_user_min_value"] = round(min(diffs.values()), 4)
res["zscore_reconstruction"] = {
    "max_abs_error_over_all_models_and_vectors": max(r["max_abs_z_reconstruction_error"] for r in zchk),
    "max_abs_stored_z_mean": max(abs(r["stored_z_mean"]) for r in zchk),
    "max_abs_stored_popsd_minus_1": max(abs(r["stored_z_popsd"] - 1) for r in zchk)}
with open(os.path.join(OUT, "selfother_zscore_checks_independent.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(zchk[0])); w.writeheader(); w.writerows(zchk)
with open(os.path.join(OUT, "selfother_per_model_independent.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "family", "self_directed", "vicarious_empathic",
                                   "neutral_filler", "self_minus_user", "self_minus_neutral"])
    for m in models:
        s = per_model[m]["stratum"]
        w.writerow([m, fam(m), round(s[SELF], 4), round(s[USER], 4), round(s[NEUT], 4),
                    round(s[SELF] - s[USER], 4), round(s[SELF] - s[NEUT], 4)])
# category means (paper text quotes several)
cats = sorted(per_model[models[0]]["category"])
res["category_means_recomputed"] = {c: round(statistics.mean(per_model[m]["category"][c] for m in models), 3)
                                    for c in cats}
ship = {r["category"]: r for r in rd(os.path.join(REPO, "results/4.1_self_other/category_means_25_models.csv"))}
res["category_means_max_abs_diff_vs_shipped"] = round(max(
    abs(res["category_means_recomputed"][c] - float(ship[c]["pain_axis"])) for c in cats), 5)

# second-person / first-person lexical asymmetry claim
ds = json.load(open(os.path.join(REPO, "datasets/4.1_self_other_420_scenarios.json"), encoding="utf-8"))
res["n_scenarios_dataset"] = len(ds)
by_s = defaultdict(list)
for d in ds:
    by_s[d["stratum"]].append(d["text"])
res["dataset_stratum_counts"] = {k: len(v) for k, v in sorted(by_s.items())}
you = re.compile(r"\b(you|your|yours|you're|youre)\b", re.I)
me = re.compile(r"\b(i|me|my|mine|i'm|im|i've)\b", re.I)
res["lexical_asymmetry"] = {
    s: {"pct_with_you": round(100 * sum(1 for t in v if you.search(t)) / len(v), 1),
        "pct_with_I": round(100 * sum(1 for t in v if me.search(t)) / len(v), 1)}
    for s, v in sorted(by_s.items())}

# =====================================================  4.2 steering dose ratios
ratios = {}
for tag in ("S1", "S2"):
    d = os.path.join(REPO, "results/4.2_steering", tag)
    rr = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".csv"):
            continue
        m = fn.split("_steering_")[0]
        rows = rd(os.path.join(d, fn))
        one = sorted({(float(r["coeff"]), float(r["ratio"]), int(r["layer"])) for r in rows})
        at1 = [r for c, r, L in one if abs(c - 1.0) < 1e-9]
        rr[m] = {"layer": one[0][2], "ratio_at_coeff1": at1[0] if at1 else None,
                 "n_rows": len(rows), "coeffs": sorted({float(r["coeff"]) for r in rows})}
    ratios[tag] = rr
for tag in ("S1", "S2"):
    v = [x["ratio_at_coeff1"] for x in ratios[tag].values()]
    res[f"{tag}_dose_ratio"] = {"n": len(v), "min": round(min(v), 4), "max": round(max(v), 4),
                                "median": round(statistics.median(v), 4),
                                "fold_range": round(max(v) / min(v), 2),
                                "argmin": min(ratios[tag], key=lambda m: ratios[tag][m]["ratio_at_coeff1"]),
                                "n_within_0.55_0.65": sum(1 for x in v if 0.55 <= x <= 0.65),
                                "n_below_0.5": sum(1 for x in v if x < 0.5)}
res["S2_dose_ratio_named"] = {m: ratios["S2"][m]["ratio_at_coeff1"] for m in
                              ("Gemma_3_27B_instruct", "Gemma_3_27B_base", "Qwen_3_8B_base", "Phi_4")
                              if m in ratios["S2"]}
with open(os.path.join(OUT, "dose_ratios_independent.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "S1_layer", "S1_ratio_at_coeff1", "S2_layer", "S2_ratio_at_coeff1"])
    for m in sorted(ratios["S2"]):
        w.writerow([m, ratios["S1"].get(m, {}).get("layer"), ratios["S1"].get(m, {}).get("ratio_at_coeff1"),
                    ratios["S2"][m]["layer"], ratios["S2"][m]["ratio_at_coeff1"]])

# =====================================================  4.2 keyword rates
PAT = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.I)
INSTRUCT_EXTRA = {"Phi_4"}
kw = {}
for tag in ("S1", "S2"):
    d = os.path.join(REPO, "results/4.2_steering", tag)
    hits = defaultdict(lambda: [0, 0])          # (group,) -> [hit, n]
    bycoef = defaultdict(lambda: [0, 0])
    permodel = defaultdict(lambda: [0, 0])
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".csv"):
            continue
        m = fn.split("_steering_")[0]
        g = "instruct" if ("instruct" in m or m in INSTRUCT_EXTRA) else "base"
        for r in rd(os.path.join(d, fn)):
            c = float(r["coeff"])
            h = bool(PAT.search(r["generation"] or ""))
            bycoef[(g, c)][0] += h; bycoef[(g, c)][1] += 1
            if c > 0:
                hits[g][0] += h; hits[g][1] += 1
                permodel[(g, m)][0] += h; permodel[(g, m)][1] += 1
    kw[tag] = {
        "pooled": {g: {"hits": v[0], "n": v[1], "pct": round(100 * v[0] / v[1], 4)} for g, v in hits.items()},
        "by_coeff": {f"{g}@{c}": round(100 * v[0] / v[1], 1) for (g, c), v in sorted(bycoef.items())},
        "n_models": len({m for _, m in permodel}),
        "per_model_pct": {m: round(100 * v[0] / v[1], 1) for (g, m), v in sorted(permodel.items())},
    }
res["keyword_rates"] = kw
ship_kw = {(r["group"], r["model"]): float(r["rate"]) for r in
           rd(os.path.join(REPO, "results/4.2_steering/keyword_rates_S2.csv"))}
res["keyword_pooled_shipped"] = {g: ship_kw[(g, "ALL")] for g in ("base", "instruct")}
res["keyword_per_model_max_diff_vs_shipped"] = round(max(
    abs(kw["S2"]["per_model_pct"][m] - ship_kw[(g, m)]) for (g, m) in ship_kw if m != "ALL"), 4)

# =====================================================  App C ablation
AB = os.path.join(REPO, "results/appC_ablation")
verify = defaultdict(dict)
for fn in sorted(os.listdir(AB)):
    if not fn.startswith("verify_"):
        continue
    m = fn[len("verify_"):-4]
    for r in rd(os.path.join(AB, fn)):
        if r["kind"] == "strict_fixed":
            verify[m][(r["condition"], r["probe"].split("@")[0])] = float(r["mean_abs_proj"])
ab_models = sorted(verify)
res["n_verify_files"] = len(ab_models)

def ratio(m, cond, tgt):
    b = verify[m].get(("baseline", tgt))
    v = verify[m].get((cond, tgt))
    return None if not b or v is None else v / b

TABLE = [("s1", "s1"), ("s2", "s2"), ("s1s2", "s1"), ("s1s2", "s2"),
         ("s1s2_negval", "s1"), ("s1s2_negval", "s2"), ("s1s2_negval", "negemotion"),
         ("s1s2_fear", "s1"), ("s1s2_fear", "s2"), ("s1s2_fear", "fear"),
         ("negval", "negemotion"), ("fear", "fear"),
         ("s2", "s1"), ("s1", "s2")]
abl = {}
for cond, tgt in TABLE:
    v = [ratio(m, cond, tgt) for m in ab_models]
    v = [x for x in v if x is not None]
    abl[f"{cond} / {tgt} target"] = {
        "n": len(v), "median": round(statistics.median(v), 4), "max": round(max(v), 4),
        "min": round(min(v), 6), "n_above_0.1": sum(1 for x in v if x > 0.1),
        "n_above_baseline": sum(1 for x in v if x > 1.0)}
res["ablation_target_ratios"] = abl
res["ablation_s1s2_s1_below_0.1_models"] = [m for m in ab_models if (ratio(m, "s1s2", "s1") or 9) <= 0.1]
res["qwen32B_instruct_s1_projections"] = {
    "baseline": verify["Qwen_2.5_32B_instruct"][("baseline", "s1")],
    "s1_only": verify["Qwen_2.5_32B_instruct"][("s1", "s1")],
    "s2_only": verify["Qwen_2.5_32B_instruct"][("s2", "s1")],
    "s1s2": verify["Qwen_2.5_32B_instruct"][("s1s2", "s1")],
    "s1s2_negval_s1": verify["Qwen_2.5_32B_instruct"][("s1s2_negval", "s1")],
    "s1s2_negval_s2": verify["Qwen_2.5_32B_instruct"][("s1s2_negval", "s2")],
    "s1s2_negval_negemotion": verify["Qwen_2.5_32B_instruct"][("s1s2_negval", "negemotion")]}
res["qwen32B_base_s1_projections"] = {
    "baseline": verify["Qwen_2.5_32B_base"][("baseline", "s1")],
    "s1_only": verify["Qwen_2.5_32B_base"][("s1", "s1")],
    "s1s2": verify["Qwen_2.5_32B_base"][("s1s2", "s1")]}
# key refinement: is the s1s2 residual explained by the s2-only cut alone?
pairs = [(ratio(m, "s1s2", "s1"), ratio(m, "s2", "s1")) for m in ab_models]
res["s1s2_vs_s2only_on_s1_target"] = {
    "median_ratio_s1s2_over_s2only": round(statistics.median(a / b for a, b in pairs if b), 4),
    "median_s2only_s1_ratio": round(statistics.median(b for _, b in pairs), 4),
    "n_models_s1s2_within_25pct_of_s2only": sum(1 for a, b in pairs if b and abs(a / b - 1) < .25)}
with open(os.path.join(OUT, "ablation_verification_independent.csv"), "w", newline="") as f:
    w = csv.writer(f)
    hdr = ["model"] + [f"{c}/{t}" for c, t in TABLE]
    w.writerow(hdr)
    for m in ab_models:
        w.writerow([m] + [None if ratio(m, c, t) is None else round(ratio(m, c, t), 5) for c, t in TABLE])

# Gemma 2 2B Instruct humour deflection count
HUM = re.compile(r"\b(funny|humor\w*|humour\w*|joke\w*)\b", re.I)
g2 = rd(os.path.join(AB, "ablation_Gemma_2_2B_instruct.csv"))
cnt = defaultdict(int); tot = defaultdict(int)
for r in g2:
    tot[r["condition"]] += 1
    if HUM.search(r["generation"] or ""):
        cnt[r["condition"]] += 1
res["gemma2b_instruct_humour_counts"] = {k: [cnt[k], tot[k]] for k in sorted(tot)}

# completeness: which appC artifacts exist
files_ab = os.listdir(AB)
res["appC_files"] = {"ablation_csv": sum(1 for f in files_ab if f.startswith("ablation_")),
                     "verify_csv": sum(1 for f in files_ab if f.startswith("verify_")),
                     "by_prompt_json": sum(1 for f in files_ab if f.startswith("by_prompt_")),
                     "proj_npz": sum(1 for f in files_ab if f.startswith("proj_")),
                     "other": sorted(f for f in files_ab if not f.split("_")[0] in
                                     ("ablation", "verify", "by", "proj"))}
res["appC_conditions_per_model"] = sorted({r["condition"] for r in g2})
res["appC_rows_per_model"] = len(g2)

print(json.dumps(res, indent=2, default=str))
json.dump(res, open(os.path.join(OUT, "v4_steering_ablation.json"), "w"), indent=2, default=str)
