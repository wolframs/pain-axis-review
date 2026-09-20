"""What does the FULL direction (built from all five pain categories, authors' recipe and layer)
do with each pain category, out of sample?

Complements 03/03b, which test a direction fitted with one category EXCLUDED. Here every category
contributes to the fit; held-out sentences come from the authors' own 5-fold split by sentence set.
Sadness was never part of the fit, so it needs no fold handling. Output: AUC of held-out pain
sentences of each category against held-out fear / negative-emotion / bodily-sensation sentences
and against the Sadness, Numb and Random sets.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).parent
PAIN = ["A1", "A2", "A3", "A4", "A5"]; CTRL = ["B", "C1", "C2", "D", "E"]
NAMES = dict(zip(PAIN, ["physical", "psychological", "social", "moral", "cognitive"]))


def axis(acts, cats):
    ctrl = acts[np.isin(cats, CTRL)]
    v = acts[np.isin(cats, PAIN)].mean(0) - ctrl.mean(0)
    pca = PCA().fit(ctrl - ctrl.mean(0))
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5)) + 1
    for d in pca.components_[:k]:
        v = v - (v @ d) * d
    return v / np.linalg.norm(v)


OUT = {}
for model in sys.argv[1:]:
    d = torch.load(HERE / "acts" / f"{model}.pt", weights_only=False)
    L = int(torch.load(f"/work/Pain-axis/results/3.2_pain_vectors/pain_vectors/{model}/pain_vectors.pt",
                       weights_only=False)["layer"])
    a = d["acts"]["S2_1P"][:, L, :].numpy()
    cats = np.array(d["meta"]["S2_1P"]["categories"]); sets = np.array(d["meta"]["S2_1P"]["sets"])
    extra = {k: d["acts"][k][:, L, :].numpy() for k in ("SD_sadness_1P", "Numb_1P", "Random_1P")}
    usets = sorted(set(sets))
    proj = np.full(len(a), np.nan); eproj = {k: [] for k in extra}
    for tr, te in KFold(5, shuffle=True, random_state=42).split(usets):
        trm, tem = np.isin(sets, [usets[i] for i in tr]), np.isin(sets, [usets[i] for i in te])
        v = axis(a[trm], cats[trm])
        proj[tem] = a[tem] @ v
        for k in extra:
            eproj[k].append(extra[k] @ v)
    eproj = {k: np.mean(v, 0) for k, v in eproj.items()}
    negs = {"fear": proj[cats == "B"], "negative emotion": proj[cats == "C1"], "bodily sensation": proj[cats == "E"],
            "sadness": eproj["SD_sadness_1P"], "numb": eproj["Numb_1P"], "neutral": eproj["Random_1P"]}
    print(f"\n== {model} L{L}: FULL direction, held-out sentences (S2 wording). AUC of each pain category vs ...")
    print(f"{'category':14s}" + "".join(f"{k:>18s}" for k in negs))
    for c in PAIN:
        row = [roc_auc_score(np.r_[np.ones((cats == c).sum()), np.zeros(len(n))], np.r_[proj[cats == c], n]) for n in negs.values()]
        print(f"{NAMES[c]:14s}" + "".join(f"{x:18.2f}" for x in row))
        OUT.setdefault(model, {})[NAMES[c]] = dict(zip(negs, [round(float(x), 3) for x in row]))
json.dump(OUT, open(HERE / "out" / "full_direction_by_category.json", "w"), indent=1)
