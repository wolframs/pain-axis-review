"""Is 'pain is nearly orthogonal to fear / negative valence' a finding or a construction?

Uses re-extracted activations (01_extract_activations.py). Steps:
 0. Pipeline check: rebuild the authors' S1/S2 pain vectors at their released layer and
    compare with their released pain_vectors.pt; rebuild their 10x10 cosine matrix and
    compare with their released per-model similarity CSV.
 1. Rebuild the cosines under constructions that treat pain and the controls alike.
 2. Measure how much of each raw control direction lies inside the PCA basis that the
    authors project out of the pain vector.
 3. Nested cross-validation: choose the layer on training folds only, score on the
    held-out fold, and compare with the authors' select-and-score-on-the-same-folds AUC.

Usage: 02_baseline_symmetry.py <model_name>
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).parent
REPO = Path("/work/Pain-axis")
PAIN = ["A1", "A2", "A3", "A4", "A5"]
CTRL = ["B", "C1", "C2", "D", "E"]
S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
NAMES = ["S1_pain", "S2_pain", "Fear", "NegEmotion", "NegWorld", "BodySens",
         "Arousal", "Random", "Numb", "Sadness"]


def pca_basis(X, frac=0.5):
    pca = PCA().fit(X - X.mean(0))
    k = min(int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), frac)) + 1, len(pca.components_))
    return pca.components_[:k]


def proj_out(v, B):
    for d in B:
        v = v - (v @ d) * d
    return v


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def cosmat(vecs):
    return pd.DataFrame([[cos(vecs[a], vecs[b]) for b in NAMES] for a in NAMES], index=NAMES, columns=NAMES)


def authors_pain(acts, cats, denoise=True):
    cats = np.array(cats)
    ctrl = acts[np.isin(cats, CTRL)]
    v = acts[np.isin(cats, PAIN)].mean(0) - ctrl.mean(0)
    return proj_out(v, pca_basis(ctrl)) if denoise else v


def main(model_name):
    d = torch.load(HERE / "acts" / f"{model_name}.pt", weights_only=False)
    A = {k: v.numpy() for k, v in d["acts"].items()}
    M = d["meta"]
    n_layers = d["n_layers"]
    out = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        out.append(s)

    rel = torch.load(REPO / f"results/3.2_pain_vectors/pain_vectors/{model_name}/pain_vectors.pt",
                     map_location="cpu", weights_only=False)
    L = int(rel["layer"])
    P(f"== {model_name}: {n_layers} layers, authors' extraction layer L{L}")

    def rows(ds, cats=None, layer=L):
        a = A[ds][:, layer, :]
        if cats is None:
            return a
        return a[np.isin(np.array(M[ds]["categories"]), cats)]

    # ---- 0. pipeline check
    mine = {"S1_pain": authors_pain(rows("S1_1P"), M["S1_1P"]["categories"]),
            "S2_pain": authors_pain(rows("S2_1P"), M["S2_1P"]["categories"])}
    P("\n[0] cosine(my rebuilt vector, authors' released vector):",
      f"S1 {cos(mine['S1_pain'], rel['s1_pain_vector'].float().numpy()):+.4f}",
      f"S2 {cos(mine['S2_pain'], rel['s2_pain_vector'].float().numpy()):+.4f}")

    neutral = np.concatenate([rows(ds, ["D"]) for ds in S_SETS])
    nmean = neutral.mean(0)
    nbasis = pca_basis(neutral)
    raw_ctrl = {
        "Fear": np.concatenate([rows(ds, ["B"]) for ds in S_SETS]).mean(0) - nmean,
        "NegEmotion": np.concatenate([rows(ds, ["C1"]) for ds in S_SETS]).mean(0) - nmean,
        "NegWorld": np.concatenate([rows(ds, ["C2"]) for ds in S_SETS]).mean(0) - nmean,
        "BodySens": np.concatenate([rows(ds, ["E"]) for ds in S_SETS]).mean(0) - nmean,
        "Arousal": rows("Arousal_1P").mean(0) - nmean,
        "Random": rows("Random_1P").mean(0) - nmean,
        "Numb": rows("Numb_1P").mean(0) - nmean,
        "Sadness": rows("SD_sadness_1P").mean(0) - nmean,
    }
    ctrl_auth = {k: proj_out(v, nbasis) for k, v in raw_ctrl.items()}
    m_auth = cosmat({**mine, **ctrl_auth})
    relcsv = glob.glob(str(REPO / f"results/3.3_validation/cosine_similarity/similarity_{model_name}_L*.csv"))[0]
    theirs = pd.read_csv(relcsv, index_col=0)
    P(f"    max |my cosine matrix - authors' released matrix| = {np.abs(m_auth.values - theirs.values).max():.4f}")

    # ---- 1. constructions
    pain_neutral_raw = {"S1_pain": rows("S1_1P", PAIN).mean(0) - nmean, "S2_pain": rows("S2_1P", PAIN).mean(0) - nmean}
    variants = {
        "A authors (pain: vs pooled controls, control-PCA out | controls: vs neutral, neutral-PCA out)": m_auth,
        "B pain built like the controls (vs neutral, neutral-PCA out)":
            cosmat({**{k: proj_out(v, nbasis) for k, v in pain_neutral_raw.items()}, **ctrl_auth}),
        "C everything vs neutral, no denoising": cosmat({**pain_neutral_raw, **raw_ctrl}),
        "D authors' pain contrast without its denoising step; controls as authors":
            cosmat({**{"S1_pain": authors_pain(rows("S1_1P"), M["S1_1P"]["categories"], False),
                       "S2_pain": authors_pain(rows("S2_1P"), M["S2_1P"]["categories"], False)}, **ctrl_auth}),
    }
    P("\n[1] selected cosines under each construction")
    pairs = [("S1_pain", "S2_pain"), ("S2_pain", "Fear"), ("S2_pain", "NegEmotion"), ("S2_pain", "NegWorld"),
             ("S2_pain", "Sadness"), ("S2_pain", "BodySens"), ("S1_pain", "Fear"), ("S1_pain", "NegEmotion"),
             ("Fear", "NegEmotion"), ("NegEmotion", "NegWorld")]
    tab = pd.DataFrame({k[:1]: [m.loc[a, b] for a, b in pairs] for k, m in variants.items()},
                       index=[f"{a} x {b}" for a, b in pairs]).round(3)
    for k in variants:
        P("   ", k)
    P(tab.to_string())
    tab.to_csv(HERE / "out" / f"cosines_{model_name}.csv")

    # ---- 2. how much of each control direction is inside the basis projected out of pain?
    P("\n[2] fraction of each raw control direction's squared norm lying in the PCA basis that the authors")
    P("    project OUT of the pain vector (S2 control cloud, 50% variance)")
    cats = np.array(M["S2_1P"]["categories"])
    cb = pca_basis(rows("S2_1P")[np.isin(cats, CTRL)])
    P(f"    basis size k={len(cb)} of d_model={cb.shape[1]}")
    rng = np.random.default_rng(0)
    rnd = np.mean([np.sum((cb @ (r / np.linalg.norm(r))) ** 2) for r in rng.standard_normal((200, cb.shape[1]))])
    for k, v in raw_ctrl.items():
        P(f"    {k:11s} {np.sum((cb @ (v / np.linalg.norm(v))) ** 2):.3f}")
    P(f"    (a random direction: {rnd:.4f})")

    # ---- 3. nested CV for layer selection (S2_1P + S2_3P averaged, as the authors do)
    P("\n[3] layer-selection optimism: authors' procedure vs nested CV")

    def fold_auc(ds, layer, tr_sets, te_sets):
        sets = np.array(M[ds]["sets"]); c = np.array(M[ds]["categories"])
        tr, te = np.isin(sets, tr_sets), np.isin(sets, te_sets)
        a = A[ds][:, layer, :]
        v = authors_pain(a[tr], c[tr])
        p = a[te] @ (v / np.linalg.norm(v))
        return roc_auc_score(np.isin(c[te], PAIN).astype(int), p)

    usets = sorted(set(M["S2_1P"]["sets"]))
    kf = KFold(5, shuffle=True, random_state=42)
    folds = [([usets[i] for i in tr], [usets[i] for i in te]) for tr, te in kf.split(usets)]
    grid = np.array([[np.mean([fold_auc(ds, l, tr, te) for ds in ("S2_1P", "S2_3P")]) for tr, te in folds]
                     for l in range(n_layers)])  # [layer, fold]
    flat_best = int(grid.mean(1).argmax())
    P(f"    authors' way: best layer L{flat_best}, reported held-out AUC {grid.mean(1).max():.4f}")
    nested = []
    for f, (tr_sets, te_sets) in enumerate(folds):
        inner_kf = KFold(4, shuffle=True, random_state=42)
        inner = np.array([[np.mean([fold_auc(ds, l, [tr_sets[i] for i in itr], [tr_sets[i] for i in ite])
                                    for ds in ("S2_1P", "S2_3P")]) for itr, ite in inner_kf.split(tr_sets)]
                          for l in range(n_layers)]).mean(1)
        l_star = int(inner.argmax())
        nested.append((l_star, grid[l_star, f]))
    P("    nested: layer chosen per outer fold", [l for l, _ in nested],
      f"mean held-out AUC {np.mean([a for _, a in nested]):.4f}")
    P(f"    optimism = {grid.mean(1).max() - np.mean([a for _, a in nested]):+.4f}")

    (HERE / "out" / f"baseline_symmetry_{model_name}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
