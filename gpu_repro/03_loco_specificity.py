"""Leave-one-pain-category-out: build the authors' S2 pain axis from four pain categories
(their recipe, their layer), then score the held-out fifth category against sentence sets
of other negative states. Sadness was never part of the subtracted control pool, so
held-out-pain vs Sadness is the cleanest released test of 'pain, not just negative affect'."""
import sys, numpy as np, torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
HERE = Path(__file__).parent
PAIN = ["A1", "A2", "A3", "A4", "A5"]; CTRL = ["B", "C1", "C2", "D", "E"]
NAMES = {"A1": "physical", "A2": "psychological", "A3": "social", "A4": "moral", "A5": "cognitive"}

def axis(acts, cats, pain_cats):
    ctrl = acts[np.isin(cats, CTRL)]
    v = acts[np.isin(cats, pain_cats)].mean(0) - ctrl.mean(0)
    pca = PCA().fit(ctrl - ctrl.mean(0))
    k = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.5)) + 1
    for d in pca.components_[:k]:
        v = v - (v @ d) * d
    return v / np.linalg.norm(v)

for model in sys.argv[1:]:
    d = torch.load(HERE / "acts" / f"{model}.pt", weights_only=False)
    L = int(torch.load(f"/work/Pain-axis/results/3.2_pain_vectors/pain_vectors/{model}/pain_vectors.pt", weights_only=False)["layer"])
    A = {k: v[:, L, :].numpy() for k, v in d["acts"].items()}; M = d["meta"]
    # train on S2_1P, test held-out category from S1_1P (different wording) and S2_1P itself is excluded from training for that category
    s2, c2 = A["S2_1P"], np.array(M["S2_1P"]["categories"])
    s1, c1 = A["S1_1P"], np.array(M["S1_1P"]["categories"])
    negs = {"Fear(S1)": s1[c1 == "B"], "NegEmotion(S1)": s1[c1 == "C1"], "NegWorld(S1)": s1[c1 == "C2"],
            "Sadness": A["SD_sadness_1P"], "Numb": A["Numb_1P"], "Random": A["Random_1P"]}
    print(f"\n== {model} L{L}: AUC of held-out pain category (S1 wording; axis from the other four S2 categories) vs ...")
    print(f"{'held-out':14s}" + "".join(f"{k:>16s}" for k in negs))
    for held in PAIN:
        v = axis(s2, c2, [p for p in PAIN if p != held])
        pos = s1[c1 == held] @ v
        row = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(n))], np.r_[pos, n @ v]) for n in negs.values()]
        print(f"{NAMES[held]:14s}" + "".join(f"{x:16.2f}" for x in row))
