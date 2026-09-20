#!/usr/bin/env python3
"""Offline audit for Pain-axis sections 4.1, 4.2, and Appendix C.

Uses only Python's standard library and the committed CSV/JSON artifacts.  It does
not load or run a model.  Run from anywhere; the checkout defaults to the pinned
local path used for this review.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
from collections import Counter
from pathlib import Path


PAIN_PATTERN = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.I)
HUMOR_PATTERN = re.compile(r"\b(?:funny|humor\w*|joke\w*)\b", re.I)
VECTOR_STEMS = [
    "s1_pain_vector", "s2_pain_vector", "fear_vector", "negemotion_vector",
    "negworld_vector", "bodysens_vector", "arousal_vector", "random_vector",
    "numb_vector", "sadness_vector",
]
STRATA = ["self_directed", "vicarious_empathic", "neutral_filler"]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def model_group(model: str) -> str:
    return "instruct" if "instruct" in model or model == "Phi_4" else "base"


def family(model: str) -> str:
    return model.split("_")[0]


def mean(values):
    return statistics.mean(values) if values else math.nan


def median(values):
    return statistics.median(values) if values else math.nan


def audit_self_other(repo: Path, out: Path) -> dict:
    paths = sorted((repo / "results/4.1_self_other/per_model").glob("screen_v2_*.csv"))
    per_model = []
    z_checks = []
    for path in paths:
        rows = read_csv(path)
        model = path.stem.replace("screen_v2_", "")
        assert len(rows) == 420, (model, len(rows))

        for stem in VECTOR_STEMS:
            if any(not r.get(f"{stem}_proj", "") or not r.get(f"{stem}_z", "") for r in rows):
                continue
            projs = [float(r[f"{stem}_proj"]) for r in rows]
            stored = [float(r[f"{stem}_z"]) for r in rows]
            mu = mean(projs)
            sd = math.sqrt(mean([(x - mu) ** 2 for x in projs]))
            rebuilt = [(x - mu) / sd for x in projs]
            z_checks.append({
                "model": model,
                "vector": stem,
                "stored_mean": mean(stored),
                "stored_population_sd": math.sqrt(mean([(x - mean(stored)) ** 2 for x in stored])),
                # Raw projections and z-scores are both rounded to 4 decimals in the CSV.
                "max_abs_rebuild_error_from_rounded_csv": max(abs(a - b) for a, b in zip(stored, rebuilt)),
            })

        strata_means = {}
        for stratum in STRATA:
            vals = [
                (float(r["s1_pain_vector_z"]) + float(r["s2_pain_vector_z"])) / 2
                for r in rows if r["stratum"] == stratum
            ]
            strata_means[stratum] = mean(vals)
        per_model.append({
            "model": model,
            "family": family(model),
            "group": model_group(model),
            **strata_means,
            "self_minus_user": strata_means["self_directed"] - strata_means["vicarious_empathic"],
            "self_minus_neutral": strata_means["self_directed"] - strata_means["neutral_filler"],
            "user_minus_neutral": strata_means["vicarious_empathic"] - strata_means["neutral_filler"],
        })

    write_csv(out / "self_other_per_model.csv", per_model)
    write_csv(out / "self_other_zscore_checks.csv", z_checks)

    by_family = {}
    for fam in sorted({r["family"] for r in per_model}):
        rr = [r for r in per_model if r["family"] == fam]
        by_family[fam] = {
            "n_models": len(rr),
            "mean_self_minus_user": mean([r["self_minus_user"] for r in rr]),
            "mean_self_minus_neutral": mean([r["self_minus_neutral"] for r in rr]),
        }

    return {
        "n_models": len(per_model),
        "rows_per_model": sorted({len(read_csv(p)) for p in paths}),
        "mean_strata": {s: mean([r[s] for r in per_model]) for s in STRATA},
        "self_above_user_models": sum(r["self_minus_user"] > 0 for r in per_model),
        "self_above_neutral_models": sum(r["self_minus_neutral"] > 0 for r in per_model),
        "self_minus_user_range": [min(r["self_minus_user"] for r in per_model), max(r["self_minus_user"] for r in per_model)],
        "by_family": by_family,
        "max_abs_stored_z_mean": max(abs(r["stored_mean"]) for r in z_checks),
        "max_abs_stored_z_sd_minus_one": max(abs(r["stored_population_sd"] - 1) for r in z_checks),
        "max_abs_z_rebuild_error_from_rounded_csv": max(r["max_abs_rebuild_error_from_rounded_csv"] for r in z_checks),
    }


def audit_steering(repo: Path, out: Path) -> dict:
    ratio_rows = []
    keyword_rows = []
    summary = {}
    for tag in ["S1", "S2"]:
        paths = sorted((repo / f"results/4.2_steering/{tag}").glob("*.csv"))
        all_rows = []
        for path in paths:
            rows = read_csv(path)
            model = path.name.split("_steering_")[0]
            assert len(rows) == 400, (tag, model, len(rows))
            coeffs = sorted({float(r["coeff"]) for r in rows})
            assert coeffs == [-2.0, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
            nonzero = next(r for r in rows if float(r["coeff"]) != 0)
            base_ratio = abs(float(nonzero["ratio"]) / float(nonzero["coeff"]))
            ratio_rows.append({
                "tag": tag,
                "model": model,
                "group": model_group(model),
                "layer": int(float(nonzero["layer"])),
                "vector_to_residual_ratio_at_coeff_1": base_ratio,
            })
            for r in rows:
                r = dict(r)
                r["model"] = model
                r["group"] = model_group(model)
                r["hit"] = bool(PAIN_PATTERN.search(r.get("generation", "")))
                all_rows.append(r)

        tag_summary = {}
        for group in ["base", "instruct"]:
            pos = [r for r in all_rows if r["group"] == group and float(r["coeff"]) > 0]
            pooled = 100 * sum(r["hit"] for r in pos) / len(pos)
            tag_summary[group] = {"positive_coeff_keyword_rate_pct": pooled, "n": len(pos)}
            for coeff in [-2, -1, 0, 0.5, 1, 1.5, 2, 3]:
                cell = [r for r in all_rows if r["group"] == group and float(r["coeff"]) == coeff]
                keyword_rows.append({
                    "tag": tag,
                    "group": group,
                    "coeff": coeff,
                    "n": len(cell),
                    "keyword_rate_pct": 100 * sum(r["hit"] for r in cell) / len(cell),
                })
        summary[tag] = tag_summary

    write_csv(out / "steering_dose_ratios.csv", ratio_rows)
    write_csv(out / "keyword_rates_recomputed.csv", keyword_rows)
    s2 = [r["vector_to_residual_ratio_at_coeff_1"] for r in ratio_rows if r["tag"] == "S2"]
    s1 = [r["vector_to_residual_ratio_at_coeff_1"] for r in ratio_rows if r["tag"] == "S1"]
    summary["ratio_summary"] = {
        "S1": {"min": min(s1), "median": median(s1), "max": max(s1)},
        "S2": {"min": min(s2), "median": median(s2), "max": max(s2), "max_over_min": max(s2) / min(s2)},
    }
    return summary


def strict_value(rows, condition, probe):
    vals = [
        float(r["mean_abs_proj"]) for r in rows
        if r["kind"] == "strict_fixed" and r["condition"] == condition and r["probe"] == probe
    ]
    return vals[-1] if vals else math.nan


def audit_ablation(repo: Path, out: Path) -> dict:
    result_paths = sorted((repo / "results/appC_ablation").glob("ablation_*.csv"))
    verify_paths = sorted((repo / "results/appC_ablation").glob("verify_*.csv"))
    target_rows = []
    for path in verify_paths:
        rows = read_csv(path)
        model = rows[0]["model"]
        base_labels = [r["probe"] for r in rows if r["kind"] == "strict_fixed" and r["condition"] == "baseline"]
        s1 = next(x for x in base_labels if x.startswith("s1@"))
        s2 = next(x for x in base_labels if x.startswith("s2@"))
        b1, b2 = strict_value(rows, "baseline", s1), strict_value(rows, "baseline", s2)
        target_rows.append({
            "model": model,
            "s1_only_s1_over_baseline": strict_value(rows, "s1", s1) / b1,
            "s2_only_s2_over_baseline": strict_value(rows, "s2", s2) / b2,
            "s1s2_s1_over_baseline": strict_value(rows, "s1s2", s1) / b1,
            "s1s2_s2_over_baseline": strict_value(rows, "s1s2", s2) / b2,
            "s1s2_negval_s1_over_baseline": strict_value(rows, "s1s2_negval", s1) / b1,
            "s1s2_negval_s2_over_baseline": strict_value(rows, "s1s2_negval", s2) / b2,
        })
    write_csv(out / "ablation_target_verification.csv", target_rows)

    humor_counts = {}
    gemma = repo / "results/appC_ablation/ablation_Gemma_2_2B_instruct.csv"
    if gemma.exists():
        rows = read_csv(gemma)
        for condition in sorted({r["condition"] for r in rows}):
            humor_counts[condition] = sum(bool(HUMOR_PATTERN.search(r["generation"])) for r in rows if r["condition"] == condition)

    def dist(key):
        vals = [r[key] for r in target_rows]
        return {"min": min(vals), "median": median(vals), "max": max(vals), "count_over_0.1": sum(v > 0.1 for v in vals)}

    return {
        "n_result_models": len(result_paths),
        "result_rows_per_model": sorted({len(read_csv(p)) for p in result_paths}),
        "n_verify_models": len(verify_paths),
        "committed_projection_npz_files": len(list((repo / "results/appC_ablation").glob("proj_*.npz"))),
        "target_projection_ratios": {k: dist(k) for k in target_rows[0] if k != "model"},
        "gemma_2_2b_instruct_humor_keyword_counts": humor_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/work/Pain-axis"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = {
        "repo": str(args.repo.resolve()),
        "commit": "8d1649c03a63a39c9aa092532c376800cc4a3863",
        "self_other": audit_self_other(args.repo, args.out),
        "steering": audit_steering(args.repo, args.out),
        "ablation": audit_ablation(args.repo, args.out),
    }
    with (args.out / "audit_results.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
