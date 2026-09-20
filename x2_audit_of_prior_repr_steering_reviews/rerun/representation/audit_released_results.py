#!/usr/bin/env python3
"""Lightweight, dependency-free audit of the released representation results.

This script reads only committed JSON/CSV artifacts from the pinned Pain-axis
checkout. It does not load or run any language model.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


REPO = Path("/work/Pain-axis")
OUT = Path("/work/pain-axis-review/x2_audit_of_prior_repr_steering_reviews/rerun/representation")
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def auc(labels, scores):
    pos = [s for y, s in zip(labels, scores) if y]
    neg = [s for y, s in zip(labels, scores) if not y]
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def tokens(text):
    text = re.sub(r"\b(?:i|they|she|he) feel(?:s)?:\s*$", "", text.lower())
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def grouped_nb_auc(sentences, seed=42, folds=5):
    """Bernoulli-ish multinomial NB, grouped by the released `set` field."""
    groups = sorted({int(s["set"]) for s in sentences})
    random.Random(seed).shuffle(groups)
    fold_groups = [set(groups[i::folds]) for i in range(folds)]
    ys, scores = [], []
    for held in fold_groups:
        train = [s for s in sentences if int(s["set"]) not in held]
        test = [s for s in sentences if int(s["set"]) in held]
        class_docs = {0: [], 1: []}
        for s in train:
            y = int(s["category"] in {"A1", "A2", "A3", "A4", "A5"})
            class_docs[y].append(tokens(s["prompt"]))
        vocab = sorted({t for docs in class_docs.values() for doc in docs for t in doc})
        counts = {y: Counter(t for doc in class_docs[y] for t in doc) for y in (0, 1)}
        totals = {y: sum(counts[y].values()) for y in (0, 1)}
        den = {y: totals[y] + len(vocab) for y in (0, 1)}
        prior = {y: math.log(len(class_docs[y]) / len(train)) for y in (0, 1)}
        for s in test:
            doc = tokens(s["prompt"])
            ll = {}
            for y in (0, 1):
                ll[y] = prior[y] + sum(math.log((counts[y][t] + 1) / den[y]) for t in doc)
            ys.append(int(s["category"] in {"A1", "A2", "A3", "A4", "A5"}))
            scores.append(ll[1] - ll[0])
    return auc(ys, scores)


def layer_audit():
    root = REPO / "results/3.2_pain_vectors/per_model"
    rows = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        curves_path = model_dir / "layer_curves.csv"
        summary_path = model_dir / "summary.json"
        if not curves_path.exists() or not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        raw = [r for r in read_csv(curves_path) if r["extraction"] == "final_token"]
        by_layer = defaultdict(dict)
        for r in raw:
            by_layer[int(r["layer"])][r["dataset"]] = float(r["auc_vs_all_controls"])
        pooled = {layer: statistics.mean(v.values()) for layer, v in by_layer.items()}
        chosen = int(summary["best_layer_final_token"])
        best = max(pooled.values())
        n_layers = int(summary["n_layers"])
        fixed = round(0.75 * (n_layers - 1))
        within = {d: sum(v >= best - d for v in pooled.values()) for d in (0.01, 0.02, 0.05)}
        rows.append({
            "model": model_dir.name,
            "n_layers": n_layers,
            "selected_layer": chosen,
            "selected_depth": chosen / (n_layers - 1),
            "selected_cv_auc": pooled[chosen],
            "fixed_75pct_layer": fixed,
            "fixed_75pct_cv_auc": pooled[fixed],
            "selected_minus_fixed": pooled[chosen] - pooled[fixed],
            "layers_within_0.01": within[0.01],
            "layers_within_0.02": within[0.02],
            "layers_within_0.05": within[0.05],
        })
    with (OUT / "layer_selection_sensitivity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    return rows


def dataset_audit():
    core = json.loads((REPO / "datasets/3.1_pain_and_control_datasets.json").read_text())["datasets"]
    sad = json.loads((REPO / "datasets/3.1_sadness_dataset.json").read_text())["datasets"]
    allsets = {**core, **sad}
    rows = []
    suffixes = {}
    for name, block in allsets.items():
        ss = block["sentences"]
        wc = [len(tokens(s["prompt"])) for s in ss]
        suffix = Counter()
        for s in ss:
            m = re.search(r"((?:I|They|She|He) feel(?:s)?):\s*$", s["prompt"])
            suffix[m.group(1) + ":" if m else "other"] += 1
        suffixes[name] = dict(suffix)
        rows.append({"dataset": name, "n": len(ss), "mean_words_without_cue": statistics.mean(wc),
                     "min_words": min(wc), "max_words": max(wc), "suffix_counts": json.dumps(dict(suffix), sort_keys=True)})
    for name in ("S1_1P", "S2_1P"):
        ss = core[name]["sentences"]
        labels = [s["category"] in {"A1", "A2", "A3", "A4", "A5"} for s in ss]
        lengths = [len(tokens(s["prompt"])) for s in ss]
        rows.append({"dataset": name + "__pain_vs_control", "n": len(ss),
                     "mean_words_without_cue": "",
                     "min_words": "", "max_words": "",
                     "suffix_counts": json.dumps({
                         "length_only_auc": auc(labels, lengths),
                         "grouped_unigram_nb_auc": grouped_nb_auc(ss),
                         "pain_mean_words": statistics.mean(x for x, y in zip(lengths, labels) if y),
                         "control_mean_words": statistics.mean(x for x, y in zip(lengths, labels) if not y),
                     }, sort_keys=True)})
    with (OUT / "dataset_audit.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    return rows, suffixes


def parse_matrix(path):
    rows = read_csv(path)
    labels = [k for k in rows[0] if k]
    return {(r[""] if "" in r else next(iter(r.values())) , lab): float(r[lab]) for r in rows for lab in labels}


def model_family(name):
    return name.split("_")[0]


def geometry_audit():
    root = REPO / "results/3.3_validation/cosine_similarity"
    raw_files = [p for p in root.glob("similarity_*_L*.csv") if "alldenoise" not in p.name and "whitened" not in p.name]
    pairs = [("S1_pain", "S2_pain"), ("S2_pain", "NegEmotion"), ("S2_pain", "Sadness"),
             ("S2_pain", "Fear"), ("S2_pain", "Numb")]
    rows = []
    per_pair = defaultdict(list)
    family_pair = defaultdict(lambda: defaultdict(list))
    for p in sorted(raw_files):
        model = re.match(r"similarity_(.+)_L\d+\.csv", p.name).group(1)
        raw = parse_matrix(p)
        white = parse_matrix(root / p.name.replace("similarity_", "similarity_whitened_", 1))
        labels = sorted({a for a, _ in raw})
        off = [(a, b) for i, a in enumerate(labels) for b in labels[i + 1:]]
        xs, ys = [raw[x] for x in off], [white[x] for x in off]
        mx, my = statistics.mean(xs), statistics.mean(ys)
        cov = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        den = math.sqrt(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))
        corr = cov / den
        max_delta = max(abs(x-y) for x,y in zip(xs,ys))
        row = {"model": model, "family": model_family(model), "raw_whitened_cell_correlation": corr,
               "raw_whitened_max_abs_delta": max_delta}
        for a,b in pairs:
            val=raw[(a,b)]
            row[f"{a}__{b}"]=val
            per_pair[(a,b)].append(val)
            family_pair[(a,b)][model_family(model)].append(val)
        rows.append(row)
    with (OUT / "geometry_per_model.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary=[]
    for pair, vals in per_pair.items():
        fammeans={fam:statistics.mean(v) for fam,v in family_pair[pair].items()}
        summary.append({"pair":" x ".join(pair),"model_weighted_mean":statistics.mean(vals),
                        "model_median":statistics.median(vals),"model_min":min(vals),"model_max":max(vals),
                        "family_balanced_mean":statistics.mean(fammeans.values()),
                        "family_means":json.dumps(fammeans,sort_keys=True)})
    with (OUT / "geometry_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    return rows, summary


def main():
    layers = layer_audit()
    datasets, suffixes = dataset_audit()
    geometry, geometry_summary = geometry_audit()
    report = {
        "layer_selection": {
            "n_models": len(layers),
            "median_selected_minus_fixed_75pct_auc": statistics.median(r["selected_minus_fixed"] for r in layers),
            "max_selected_minus_fixed_75pct_auc": max(r["selected_minus_fixed"] for r in layers),
            "median_layers_within_0.01": statistics.median(r["layers_within_0.01"] for r in layers),
            "median_layers_within_0.02": statistics.median(r["layers_within_0.02"] for r in layers),
            "median_layers_within_0.05": statistics.median(r["layers_within_0.05"] for r in layers),
        },
        "dataset_suffixes": suffixes,
        "geometry": geometry_summary,
        "whitening_per_model": {
            "correlation_min": min(r["raw_whitened_cell_correlation"] for r in geometry),
            "correlation_median": statistics.median(r["raw_whitened_cell_correlation"] for r in geometry),
            "max_abs_delta_median": statistics.median(r["raw_whitened_max_abs_delta"] for r in geometry),
            "max_abs_delta_max": max(r["raw_whitened_max_abs_delta"] for r in geometry),
        },
    }
    (OUT / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
