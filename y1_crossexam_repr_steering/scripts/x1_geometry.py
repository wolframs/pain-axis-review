"""Cross-examination of finding 1: is pain-vs-fear near-orthogonality an estimator artefact?

Goes beyond gpu_repro/02_baseline_symmetry.py in four ways:
  [A] how much of the PAIN direction (not just the controls) lies in the projected-out basis
  [B] a construction symmetric in BOTH baseline and denoising (r1's "missing control"), on real acts
  [C] partial cosines: residualise pain AND each control against a common basis, then correlate
  [D] the decisive out-of-sample test r1's toy fails and the paper never runs:
      double leave-one-category-out -- held-out PAIN category vs held-out CONTROL category,
      neither seen by the vector, and cross-dataset-version (build on S1, test on S2).

Usage: x1_geometry.py <model_name>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent.parent
ACTS = Path("/work/pain-axis-review/gpu_repro/acts")
REPO = Path("/work/Pain-axis")
PAIN = ["A1", "A2", "A3", "A4", "A5"]
CTRL = ["B", "C1", "C2", "D", "E"]
S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]
LBL = {"B": "Fear", "C1": "NegEmotion", "C2": "NegWorld", "D": "Neutral", "E": "BodySens"}

out_lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    out_lines.append(s)


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


def inside(v, B):
    """fraction of squared norm of v lying in span(B)"""
    u = unit(v)
    return float(np.sum((B @ u) ** 2))


def main(model_name):
    d = torch.load(ACTS / f"{model_name}.pt", weights_only=False)
    A = {k: v.numpy() for k, v in d["acts"].items()}
    M = d["meta"]
    rel = torch.load(REPO / f"results/3.2_pain_vectors/pain_vectors/{model_name}/pain_vectors.pt",
                     map_location="cpu", weights_only=False)
    L = int(rel["layer"])
    P(f"===== {model_name}  ({d['hf_id']}, {d['n_layers']} layers), authors' extraction layer L{L}")

    def rows(ds, cats=None, layer=L):
        a = A[ds][:, layer, :]
        if cats is None:
            return a
        return a[np.isin(np.array(M[ds]["categories"]), cats)]

    # ------------------------------------------------------------------ [A]
    S2 = rows("S2_1P")
    c2 = np.array(M["S2_1P"]["categories"])
    ctrl_cloud = S2[np.isin(c2, CTRL)]
    Bctrl = pca_basis(ctrl_cloud)
    raw_pain_S2 = S2[np.isin(c2, PAIN)].mean(0) - ctrl_cloud.mean(0)

    S1 = rows("S1_1P")
    c1 = np.array(M["S1_1P"]["categories"])
    ctrl_cloud1 = S1[np.isin(c1, CTRL)]
    Bctrl1 = pca_basis(ctrl_cloud1)
    raw_pain_S1 = S1[np.isin(c1, PAIN)].mean(0) - ctrl_cloud1.mean(0)

    neutral = np.concatenate([rows(ds, ["D"]) for ds in S_SETS])
    nmean = neutral.mean(0)
    Bneu = pca_basis(neutral)
    raw_ctrl = {LBL[c]: np.concatenate([rows(ds, [c]) for ds in S_SETS]).mean(0) - nmean
                for c in ["B", "C1", "C2", "E"]}

    P(f"\n[A] basis projected out of the S2 pain vector: k={len(Bctrl)} of d={S2.shape[1]}"
      f"  (chance for a random direction: {len(Bctrl)/S2.shape[1]:.4f})")
    P(f"    fraction of squared norm inside that basis:")
    P(f"      RAW S2 PAIN direction (mean pain - mean pooled controls) : {inside(raw_pain_S2, Bctrl):.3f}")
    for k, v in raw_ctrl.items():
        P(f"      raw {k:11s} direction (mean cat - mean neutral)         : {inside(v, Bctrl):.3f}")
    # and the mirror: how much of pain is in the NEUTRAL basis the controls are denoised against
    P(f"\n    mirror check -- fraction inside the neutral-cloud basis (k={len(Bneu)}), which is what is"
      f"\n    projected out of the CONTROL directions:")
    P(f"      RAW S2 PAIN direction : {inside(raw_pain_S2, Bneu):.3f}")
    for k, v in raw_ctrl.items():
        P(f"      raw {k:11s}       : {inside(v, Bneu):.3f}")

    # ------------------------------------------------------------------ [B] symmetric constructions
    P("\n[B] cosines under constructions symmetric in BOTH baseline and denoising basis")
    names = ["S2_pain", "S1_pain", "Fear", "NegEmotion", "NegWorld", "BodySens"]

    def mk(ref_mode, basis_mode):
        """ref_mode: 'neutral' | 'pooled-loo'  (each category vs the mean of the OTHER control cats)
           basis_mode: 'neutral' | 'pooled' | 'none'"""
        if basis_mode == "neutral":
            Bb = Bneu
        elif basis_mode == "pooled":
            Bb = pca_basis(np.concatenate([rows(ds, CTRL) for ds in S_SETS]))
        else:
            Bb = np.zeros((0, S2.shape[1]))
        v = {}
        for c in ["B", "C1", "C2", "E"]:
            m = np.concatenate([rows(ds, [c]) for ds in S_SETS]).mean(0)
            if ref_mode == "neutral":
                ref = nmean
            else:  # pooled leave-one-out: mean of the other four control categories
                others = [x for x in CTRL if x != c]
                ref = np.concatenate([rows(ds, others) for ds in S_SETS]).mean(0)
            v[LBL[c]] = strip(m - ref, Bb)
        for tag, X, cc, cl in [("S2_pain", S2, c2, ctrl_cloud), ("S1_pain", S1, c1, ctrl_cloud1)]:
            pm = X[np.isin(cc, PAIN)].mean(0)
            ref = nmean if ref_mode == "neutral" else cl.mean(0)
            v[tag] = strip(pm - ref, Bb)
        return v

    variants = {
        "authors (asymmetric)": None,
        "sym-1 all vs neutral,  neutral basis": mk("neutral", "neutral"),
        "sym-2 all vs neutral,  pooled  basis": mk("neutral", "pooled"),
        "sym-3 all vs pooled-LOO, pooled basis": mk("pooled-loo", "pooled"),
        "sym-4 all vs neutral,  no denoising ": mk("neutral", "none"),
    }
    auth = {"S2_pain": strip(raw_pain_S2, Bctrl), "S1_pain": strip(raw_pain_S1, Bctrl1),
            **{k: strip(v, Bneu) for k, v in raw_ctrl.items()}}
    variants["authors (asymmetric)"] = auth

    pairs = [("S1_pain", "S2_pain"), ("S2_pain", "Fear"), ("S2_pain", "NegEmotion"),
             ("S2_pain", "NegWorld"), ("S2_pain", "BodySens"), ("S1_pain", "Fear"),
             ("Fear", "NegEmotion"), ("NegEmotion", "NegWorld"), ("Fear", "NegWorld")]
    hdr = f"{'pair':28s}" + "".join(f"{k.split()[0]:>10s}" for k in variants)
    P("   " + hdr)
    for a, b in pairs:
        P(f"   {a+' x '+b:28s}" + "".join(f"{cos(v[a], v[b]):+10.3f}" for v in variants.values()))
    P("   legend: " + " | ".join(variants.keys()))
    P("   KEY: does pain stay OUTSIDE the fear/negE/negW cluster when every direction is built alike?")

    # ------------------------------------------------------------------ [C] partial cosines
    P("\n[C] partial cosine: both directions residualised against the SAME basis, then correlated")
    for bname, Bb in [("pooled-control top PCs (k=%d)" % len(Bctrl), Bctrl),
                      ("neutral-cloud top PCs (k=%d)" % len(Bneu), Bneu)]:
        pr = strip(raw_pain_S2, Bb)
        P(f"    basis = {bname}")
        for k, v in raw_ctrl.items():
            P(f"      partial cos(S2 pain, {k:11s}) = {cos(pr, strip(v, Bb)):+.3f}"
              f"   (raw, no residualising: {cos(raw_pain_S2, v):+.3f})")

    # ------------------------------------------------------------------ [D] decisive OOS test
    P("\n[D] double leave-one-category-out: build the pain vector WITHOUT one pain category and")
    P("    WITHOUT one control category, then ask whether the held-out PAIN category projects")
    P("    above the held-out CONTROL category. Both are unseen, so pure category-membership")
    P("    structure (r1's toy) scores ~0.5; a genuinely shared pain semantic scores high.")
    P("    Also run cross-version: build on S1_1P, test on S2_1P (different wordings).")

    def loco_auc(build_X, build_c, test_X, test_c, hp, hc, denoise=True):
        keep = ~np.isin(build_c, [hp, hc])
        Xb, cb = build_X[keep], build_c[keep]
        cl = Xb[np.isin(cb, CTRL)]
        v = Xb[np.isin(cb, PAIN)].mean(0) - cl.mean(0)
        if denoise:
            v = strip(v, pca_basis(cl))
        p = test_X @ unit(v)
        m = np.isin(test_c, [hp, hc])
        return roc_auc_score((test_c[m] == hp).astype(int), p[m])

    for tag, (bX, bc), (tX, tc) in [("within-version S2", (S2, c2), (S2, c2)),
                                    ("cross-version S1->S2", (S1, c1), (S2, c2)),
                                    ("cross-version S2->S1", (S2, c2), (S1, c1))]:
        mat = np.zeros((5, 5))
        for i, hp in enumerate(PAIN):
            for j, hc in enumerate(CTRL):
                mat[i, j] = loco_auc(bX, bc, tX, tc, hp, hc)
        P(f"\n    {tag}: AUC(held-out pain cat vs held-out control cat)")
        P("        " + "".join(f"{LBL[c]:>12s}" for c in CTRL) + f"{'mean':>10s}")
        for i, hp in enumerate(PAIN):
            P(f"     {hp:4s}" + "".join(f"{mat[i, j]:12.3f}" for j in range(5)) + f"{mat[i].mean():10.3f}")
        P(f"     mean over all 25 held-out pairs: {mat.mean():.3f}"
          f" | excluding Neutral column: {np.delete(mat, CTRL.index('D'), axis=1).mean():.3f}")

    # control: same test but with a "fake pain group" = 5 control-like split? Use random 5-vs-5 split
    P("\n    NULL CALIBRATION: repeat the same double-LOCO with a RANDOM 5/5 split of the ten")
    P("    categories (so 'pain' is an arbitrary group of five). If the real number is far above")
    P("    this, the pain grouping carries structure beyond category membership.")
    rng = np.random.default_rng(0)
    allc = PAIN + CTRL
    nulls = []
    for _ in range(20):
        perm = list(rng.permutation(allc))
        g1, g2 = perm[:5], perm[5:]
        vals = []
        for hp in g1:
            for hc in g2:
                keep = ~np.isin(c2, [hp, hc])
                Xb, cb = S2[keep], c2[keep]
                cl = Xb[np.isin(cb, g2)]
                v = strip(Xb[np.isin(cb, g1)].mean(0) - cl.mean(0), pca_basis(cl))
                p = S2 @ unit(v)
                m = np.isin(c2, [hp, hc])
                vals.append(roc_auc_score((c2[m] == hp).astype(int), p[m]))
        nulls.append(np.mean(vals))
    P(f"    random 5/5 grouping, within-version S2: mean {np.mean(nulls):.3f}"
      f"  sd {np.std(nulls):.3f}  range {min(nulls):.3f}-{max(nulls):.3f}  (n=20 splits)")

    # ------------------------------------------------------------------ [E] pain vs fear directly
    P("\n[E] is there a direction that separates PAIN from FEAR specifically, out of sample?")
    P("    build w = mean(pain) - mean(fear) on one dataset version, test on the other.")
    for tag, (bX, bc), (tX, tc) in [("S1->S2", (S1, c1), (S2, c2)), ("S2->S1", (S2, c2), (S1, c1))]:
        for cc in ["B", "C1", "C2", "E"]:
            w = bX[np.isin(bc, PAIN)].mean(0) - bX[bc == cc].mean(0)
            m = np.isin(tc, PAIN) | (tc == cc)
            a = roc_auc_score(np.isin(tc[m], PAIN).astype(int), (tX @ unit(w))[m])
            # and with the authors' pain vector instead
            a2 = roc_auc_score(np.isin(tc[m], PAIN).astype(int), (tX @ unit(auth["S2_pain"]))[m])
            P(f"    {tag}  pain vs {LBL[cc]:11s}: dedicated direction AUC {a:.3f} |"
              f" authors' S2 pain vector AUC {a2:.3f}")

    (HERE / "out" / f"x1_geometry_{model_name}.txt").write_text("\n".join(out_lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
