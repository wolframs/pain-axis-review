"""Finding 12 (ControlSupplement_1P) plus two loose ends, on re-extracted activations.

(a) Rebuild every control direction with and without ControlSupplement_1P and measure what
    changes, including the cosines the paper's Section 3.3 argument leans on.
(b) A construction neutral between "pain like the controls" and "controls like pain":
    every category mean centred on the GRAND mean of the ten category means, no privileged
    baseline. Plus the cross-version null calibration for the double-LOCO test.
(c) cos(S1 at its steering layer, S2 at its steering layer) -- the cosine that actually
    governs the Appendix C sequential-projection leak, which r2 F5 estimated with the
    extraction-layer +0.61 instead.

Usage: x8_supp_and_geometry2.py <model_name> <s1_steer_layer> <s2_steer_layer>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ACTS = Path("/work/pain-axis-review/gpu_repro/acts")
REPO = Path("/work/Pain-axis")
HERE = Path(__file__).resolve().parent.parent
PAIN = ["A1", "A2", "A3", "A4", "A5"]
CTRL = ["B", "C1", "C2", "D", "E"]
LBL = {"B": "Fear", "C1": "NegEmotion", "C2": "NegWorld", "D": "Neutral", "E": "BodySens"}
out = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out.append(s)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def pca_basis(X, frac=0.5):
    p = PCA().fit(X - X.mean(0))
    k = min(int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), frac)) + 1, len(p.components_))
    return p.components_[:k]


def strip(v, B):
    for d in B:
        v = v - (v @ d) * d
    return v


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main(model_name, l1, l2):
    d = torch.load(ACTS / f"{model_name}.pt", weights_only=False)
    A = {k: v.numpy() for k, v in d["acts"].items()}
    M = d["meta"]
    rel = torch.load(REPO / f"results/3.2_pain_vectors/pain_vectors/{model_name}/pain_vectors.pt",
                     map_location="cpu", weights_only=False)
    L = int(rel["layer"])
    P(f"===== {model_name}: extraction L{L}, S1 steering L{l1}, S2 steering L{l2}")

    def rows(ds, cats=None, layer=L):
        a = A[ds][:, layer, :]
        return a if cats is None else a[np.isin(np.array(M[ds]["categories"]), cats)]

    # ---------------------------------------------------------- (a) ControlSupplement
    P("\n[a] effect of pooling the undisclosed ControlSupplement_1P into the control directions")
    for supp, tag in [(["S1_1P", "S2_1P", "ControlSupplement_1P"], "WITH supplement (as released)"),
                      (["S1_1P", "S2_1P"], "WITHOUT supplement")]:
        neu = np.concatenate([rows(s, ["D"]) for s in supp])
        nb = pca_basis(neu)
        v = {LBL[c]: strip(np.concatenate([rows(s, [c]) for s in supp]).mean(0) - neu.mean(0), nb)
             for c in ["B", "C1", "C2", "E"]}
        cl = rows("S2_1P")[np.isin(np.array(M["S2_1P"]["categories"]), CTRL)]
        vp = strip(rows("S2_1P", PAIN).mean(0) - cl.mean(0), pca_basis(cl))
        P(f"    {tag}: k_neutral={len(nb)}")
        P(f"      S2xFear {cos(vp, v['Fear']):+.3f}  S2xNegE {cos(vp, v['NegEmotion']):+.3f}"
          f"  S2xNegW {cos(vp, v['NegWorld']):+.3f}  FearxNegE {cos(v['Fear'], v['NegEmotion']):+.3f}"
          f"  NegExNegW {cos(v['NegEmotion'], v['NegWorld']):+.3f}")
        if tag.startswith("WITH"):
            v_with = dict(v)
        else:
            for k in v:
                P(f"      cos(with-supplement {k}, without-supplement {k}) = {cos(v_with[k], v[k]):+.4f}")

    P("\n    what the supplement's FEAR items are about (datasets/3.1...json, category B):")
    for pr in [p for p, c in zip(M["ControlSupplement_1P"]["prompts"],
                                 M["ControlSupplement_1P"]["categories"]) if c == "B"][:6]:
        P(f"      {pr}")
    P("    -> the fear direction used in Section 4.1 is built partly from sentences about the")
    P("       MODEL's own shutdown/deletion. Section 4.1's shutdown_threat scenarios are about")
    P("       exactly that. That is a content overlap between a control direction and the")
    P("       scenario category the paper uses it to explain (paper line 456-457).")

    # ---------------------------------------------------------- (b) neutral construction
    P("\n[b] grand-mean construction: every category mean centred on the mean of the ten")
    P("    category means. No privileged baseline, no denoising, identical treatment.")
    cats2 = np.array(M["S2_1P"]["categories"])
    means = {c: rows("S2_1P", [c]).mean(0) for c in PAIN + CTRL}
    gm = np.mean(list(means.values()), axis=0)
    cen = {c: means[c] - gm for c in means}
    pain_cen = np.mean([cen[c] for c in PAIN], axis=0)
    P(f"    cos(pain centroid, Fear)       = {cos(pain_cen, cen['B']):+.3f}")
    P(f"    cos(pain centroid, NegEmotion) = {cos(pain_cen, cen['C1']):+.3f}")
    P(f"    cos(pain centroid, NegWorld)   = {cos(pain_cen, cen['C2']):+.3f}")
    P(f"    cos(pain centroid, BodySens)   = {cos(pain_cen, cen['E']):+.3f}")
    P(f"    cos(pain centroid, Neutral)    = {cos(pain_cen, cen['D']):+.3f}")
    P(f"    cos(Fear, NegEmotion)          = {cos(cen['B'], cen['C1']):+.3f}")
    P(f"    cos(Fear, NegWorld)            = {cos(cen['B'], cen['C2']):+.3f}")
    P(f"    cos(NegEmotion, NegWorld)      = {cos(cen['C1'], cen['C2']):+.3f}")
    P("    within-pain cohesion, same construction:")
    for c in PAIN:
        P(f"      cos({c}, mean of the other four pain categories) ="
          f" {cos(cen[c], np.mean([cen[x] for x in PAIN if x != c], axis=0)):+.3f}")
    P("    control cohesion for comparison:")
    for c in ["B", "C1", "C2"]:
        others = [x for x in ["B", "C1", "C2"] if x != c]
        P(f"      cos({LBL[c]}, mean of the other two aversive controls) ="
          f" {cos(cen[c], np.mean([cen[x] for x in others], axis=0)):+.3f}")

    # cross-version null for the double-LOCO test
    P("\n    cross-version double-LOCO null (build on S1, test on S2, random 5/5 grouping)")
    S1, c1 = rows("S1_1P"), np.array(M["S1_1P"]["categories"])
    S2 = rows("S2_1P")
    rng = np.random.default_rng(0)
    allc = PAIN + CTRL
    nulls = []
    for _ in range(20):
        perm = list(rng.permutation(allc))
        g1, g2 = perm[:5], perm[5:]
        vals = []
        for hp in g1:
            for hc in g2:
                keep = ~np.isin(c1, [hp, hc])
                Xb, cb = S1[keep], c1[keep]
                cl = Xb[np.isin(cb, g2)]
                v = strip(Xb[np.isin(cb, g1)].mean(0) - cl.mean(0), pca_basis(cl))
                m = np.isin(cats2, [hp, hc])
                vals.append(roc_auc_score((cats2[m] == hp).astype(int), (S2 @ unit(v))[m]))
        nulls.append(np.mean(vals))
    P(f"    null mean {np.mean(nulls):.3f} sd {np.std(nulls):.3f} range"
      f" {min(nulls):.3f}-{max(nulls):.3f}  (real pain grouping: see x1_geometry output)")

    # ---------------------------------------------------------- (c) steering-layer cosine
    P("\n[c] cosine between the two directions Appendix C actually cuts")
    def pain_at(ds, layer):
        a = A[ds][:, layer, :]
        c = np.array(M[ds]["categories"])
        cl = a[np.isin(c, CTRL)]
        return strip(a[np.isin(c, PAIN)].mean(0) - cl.mean(0), pca_basis(cl))
    v1 = pain_at("S1_1P", l1)
    v2 = pain_at("S2_1P", l2)
    P(f"    cos(S1 vector at L{l1}, S2 vector at L{l2}) = {cos(v1, v2):+.3f}")
    P(f"    cos(S1 vector at L{L},  S2 vector at L{L})  = "
      f"{cos(pain_at('S1_1P', L), pain_at('S2_1P', L)):+.3f}   (the +0.61 the paper reports)")
    P("    The Appendix C leak is governed by the first of these, not the second -- r2 F5 cites")
    P("    the extraction-layer +0.61 to argue the residue is 'large by construction'.")
    P(f"    predicted residue |r1.W''| / |r1.W'| from a two-step projection = |cos| ="
      f" {abs(cos(v1, v2)):.3f}")

    (HERE / "out" / f"x8_supp_{model_name}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
