"""What distinctness claim survives the estimator objection?

r1/gpu_repro show that pain x fear ~ 0 is largely a construction artefact: the pain
vector is denoised against a basis holding 80-86% of the fear / neg-emotion directions.
Granted. The question this script asks is the one the authors should be asked instead:

  Q1. Out of sample, does the pain axis separate pain sentences from EACH control
      individually -- fear, negative emotion, negative world state, bodily sensation,
      neutral -- and from sets the vector never saw (sadness, arousal, random)?
  Q2. Is that separation symmetric? Build a FEAR axis by the authors' own recipe
      (fear vs the pooled rest, denoised against the rest) and ask whether it
      separates fear from pain. Two-way separability is not manufacturable by the
      choice of which category you privilege.
  Q3. LEAVE-ONE-PAIN-CATEGORY-OUT. Build the axis from 4 pain categories only and
      score the held-out 5th against the Random set (never part of any contrast).
      This is the test that discriminates "the 5 pain categories share a factor"
      from "the vector is the mean of 5 topic directions". r1's synthetic
      no-pain-factor model predicts chance here; the authors' hypothesis predicts
      high AUC.
  Q4. Same three questions under the SYMMETRIC construction (pain built exactly like
      the controls: vs neutral, denoised against neutral) -- the construction r1 says
      was never run.
  Q5. Physical pain specifically (r5): is A1 weakest, and is that because bodily
      sensation (E) is in the subtracted pool? Re-run dropping E from the pool.

Usage: y1_distinctness.py <model_name>
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

HERE = Path(__file__).parent
ACTS = Path("/work/pain-axis-review/gpu_repro/acts")
REPO = Path("/work/Pain-axis")
PAIN = ["A1", "A2", "A3", "A4", "A5"]
CTRL = ["B", "C1", "C2", "D", "E"]
LABEL = {"A1": "physical", "A2": "psychological", "A3": "social", "A4": "moral",
         "A5": "cognitive", "B": "fear", "C1": "neg-emotion", "C2": "neg-world",
         "D": "neutral", "E": "body-sens"}
S_SETS = ["S1_1P", "S2_1P", "ControlSupplement_1P"]


def pca_basis(X, frac=0.5):
    p = PCA().fit(X - X.mean(0))
    k = min(int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), frac)) + 1, len(p.components_))
    return p.components_[:k]


def proj_out(v, B):
    for d in B:
        v = v - (v @ d) * d
    return v


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def auc(pos, neg):
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return roc_auc_score(y, np.r_[pos, neg])


def main(model):
    d = torch.load(ACTS / f"{model}.pt", weights_only=False)
    A = {k: v.numpy() for k, v in d["acts"].items()}
    M = d["meta"]
    rel = torch.load(REPO / f"results/3.2_pain_vectors/pain_vectors/{model}/pain_vectors.pt",
                     map_location="cpu", weights_only=False)
    L = int(rel["layer"])
    out = []

    def P(*a):
        s = " ".join(str(x) for x in a)
        print(s)
        out.append(s)

    P(f"== {model}  (extraction layer L{L}, d={A['S2_1P'].shape[2]})")

    def X(ds, cats=None):
        a = A[ds][:, L, :]
        if cats is None:
            return a
        return a[np.isin(np.array(M[ds]["categories"]), cats)]

    cats2 = np.array(M["S2_1P"]["categories"])
    sets2 = np.array(M["S2_1P"]["sets"])
    a2 = A["S2_1P"][:, L, :]

    # standalone sets, never part of any pain contrast
    OUTSIDE = {"sadness": X("SD_sadness_1P"), "arousal": X("Arousal_1P"),
               "random(neutral everyday)": X("Random_1P"), "numb": X("Numb_1P")}

    # ---------- shared fitting helpers ----------
    def vec_pooled(mask, pain_cats=PAIN, ctrl_cats=CTRL, denoise=True):
        """authors' recipe on the rows selected by `mask`"""
        c = a2[mask & np.isin(cats2, ctrl_cats)]
        v = a2[mask & np.isin(cats2, pain_cats)].mean(0) - c.mean(0)
        return unit(proj_out(v, pca_basis(c)) if denoise else v)

    def vec_symmetric(mask, target_cats):
        """built exactly like a control direction: vs neutral, denoised against neutral"""
        n = a2[mask & (cats2 == "D")]
        v = a2[mask & np.isin(cats2, target_cats)].mean(0) - n.mean(0)
        return unit(proj_out(v, pca_basis(n)))

    kf = KFold(5, shuffle=True, random_state=42)
    usets = sorted(set(sets2.tolist()))
    folds = [([usets[i] for i in tr], [usets[i] for i in te]) for tr, te in kf.split(usets)]

    # ================= Q1 / Q4: per-control held-out AUC, both constructions ==========
    P("\n[Q1/Q4] Held-out AUC of the pain axis, pain vs EACH control separately.")
    P("        5-fold by sentence set; the axis is refit on the training sets only.")
    P("        'authors' = vs pooled controls + control-PCA denoise."
      "  'symmetric' = vs neutral + neutral-PCA denoise (built like a control).")
    rows = {}
    for name, builder in (("authors", vec_pooled), ("symmetric", lambda m: vec_symmetric(m, PAIN))):
        per = {c: [] for c in CTRL}
        per_out = {c: [] for c in OUTSIDE}
        allc = []
        for tr, te in folds:
            mtr, mte = np.isin(sets2, tr), np.isin(sets2, te)
            v = builder(mtr)
            pp = a2[mte & np.isin(cats2, PAIN)] @ v
            for c in CTRL:
                per[c].append(auc(pp, a2[mte & (cats2 == c)] @ v))
            allc.append(auc(pp, a2[mte & np.isin(cats2, CTRL)] @ v))
            for c, Z in OUTSIDE.items():
                per_out[c].append(auc(pp, Z @ v))  # outside sets never used in fitting
        rows[name] = (per, per_out, allc)
        P(f"\n  {name}:")
        P("    vs pooled controls           %.3f" % np.mean(allc))
        for c in CTRL:
            P(f"    vs {LABEL[c]:<14s}            %.3f" % np.mean(per[c]))
        for c in OUTSIDE:
            P(f"    vs {c:<24s}  %.3f" % np.mean(per_out[c]))

    # ================= Q2: symmetry -- a fear axis by the same recipe =================
    P("\n[Q2] Symmetry. Each category's axis built by the AUTHORS' recipe applied to it")
    P("     (category vs the pooled remaining 9 categories, denoised against that pool),")
    P("     then held-out AUC of that axis for its own category vs the PAIN categories.")
    P("     If pain and fear were the same thing, neither direction could separate them.")
    for tgt in ["B", "C1", "C2", "E", "A1"]:
        aucs, aucs_ctrl = [], []
        rest = [c for c in PAIN + CTRL if c != tgt]
        for tr, te in folds:
            mtr, mte = np.isin(sets2, tr), np.isin(sets2, te)
            v = vec_pooled(mtr, pain_cats=[tgt], ctrl_cats=rest)
            pos = a2[mte & (cats2 == tgt)] @ v
            aucs.append(auc(pos, a2[mte & np.isin(cats2, PAIN if tgt in CTRL else CTRL)] @ v))
            aucs_ctrl.append(auc(pos, a2[mte & np.isin(cats2, rest)] @ v))
        other = "all 5 pain cats" if tgt in CTRL else "all 5 controls"
        P(f"    {LABEL[tgt]:<12s} axis: vs {other:<16s} %.3f   vs pooled rest %.3f"
          % (np.mean(aucs), np.mean(aucs_ctrl)))

    # ================= Q3: leave-one-pain-category-out ==============================
    P("\n[Q3] LEAVE-ONE-PAIN-CATEGORY-OUT. Axis built from the other 4 pain categories")
    P("     (authors' recipe, all 20 sentence sets), held-out category scored against")
    P("     sets the axis never saw. Deflationary account (vector = mean of 5 topic")
    P("     directions, no shared pain factor) predicts ~0.5 vs Random; authors predict high.")
    allmask = np.ones(len(cats2), bool)
    P("      held-out cat      vs Random  vs Arousal  vs Sadness  vs neutral(D)  vs fear(B)")
    lopco = {}
    for held in PAIN:
        four = [c for c in PAIN if c != held]
        v = vec_pooled(allmask, pain_cats=four)
        pos = a2[cats2 == held] @ v
        r = [auc(pos, OUTSIDE["random(neutral everyday)"] @ v), auc(pos, OUTSIDE["arousal"] @ v),
             auc(pos, OUTSIDE["sadness"] @ v), auc(pos, a2[cats2 == "D"] @ v),
             auc(pos, a2[cats2 == "B"] @ v)]
        lopco[held] = r
        P(f"      {LABEL[held]:<16s}" + "".join("   %.3f   " % x for x in r))
    P("      (mean)          " + "".join("   %.3f   " % np.mean([lopco[h][i] for h in PAIN])
                                         for i in range(5)))

    # also: the same, with the SYMMETRIC construction, to show it is not the denoising
    P("\n     same test, symmetric construction (4 pain cats vs neutral, neutral-PCA):")
    for held in PAIN:
        four = [c for c in PAIN if c != held]
        v = vec_symmetric(allmask, four)
        pos = a2[cats2 == held] @ v
        P(f"      {LABEL[held]:<16s}   %.3f      %.3f      %.3f      %.3f      %.3f"
          % (auc(pos, OUTSIDE["random(neutral everyday)"] @ v), auc(pos, OUTSIDE["arousal"] @ v),
             auc(pos, OUTSIDE["sadness"] @ v), auc(pos, a2[cats2 == "D"] @ v),
             auc(pos, a2[cats2 == "B"] @ v)))

    # negative control: a "pain axis" built from 4 CONTROL categories, held-out control
    P("\n     negative control -- leave-one-CONTROL-out, same recipe (4 controls vs the 5 pain")
    P("     cats + remaining control), held-out control scored vs Random:")
    for held in ["B", "C1", "C2", "E"]:
        four = [c for c in CTRL if c not in (held, "D")]
        v = vec_pooled(allmask, pain_cats=four, ctrl_cats=PAIN + ["D"])
        pos = a2[cats2 == held] @ v
        P(f"      {LABEL[held]:<16s}   %.3f      (vs neutral D %.3f)"
          % (auc(pos, OUTSIDE["random(neutral everyday)"] @ v), auc(pos, a2[cats2 == "D"] @ v)))

    # ================= Q5: is physical pain weak because E is subtracted? ============
    P("\n[Q5] Physical pain (r5's mechanism). Mean z of each pain category on the axis,")
    P("     z-scored against the S2 control categories, under three control pools.")
    variants = {
        "authors (B,C1,C2,D,E)": CTRL,
        "drop body-sens (B,C1,C2,D)": ["B", "C1", "C2", "D"],
        "neutral only (D)": ["D"],
    }
    for nm, pool in variants.items():
        v = vec_pooled(allmask, ctrl_cats=pool)
        allp = a2 @ v
        mu, sd = allp[np.isin(cats2, CTRL)].mean(), allp[np.isin(cats2, CTRL)].std()
        z = (allp - mu) / sd
        s = "  ".join(f"{LABEL[c]} {z[cats2 == c].mean():+.2f}" for c in PAIN)
        P(f"    {nm:<28s} {s}")
        P(f"    {'':<28s} body-sens {z[cats2 == 'E'].mean():+.2f}  fear {z[cats2 == 'B'].mean():+.2f}"
          "  sadness %+.2f" % (((OUTSIDE["sadness"] @ v).mean() - mu) / sd))
    # per-category AUC of physical pain against body sensation, held out
    ph = []
    for tr, te in folds:
        mtr, mte = np.isin(sets2, tr), np.isin(sets2, te)
        v = vec_pooled(mtr)
        ph.append(auc(a2[mte & (cats2 == "A1")] @ v, a2[mte & (cats2 == "E")] @ v))
    P("    held-out AUC, physical pain vs non-painful bodily sensation: %.3f" % np.mean(ph))

    (HERE / "out" / f"y1_distinctness_{model}.txt").write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
