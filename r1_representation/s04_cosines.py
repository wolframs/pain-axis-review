"""Verify every cosine-similarity number in paper lines 346-378 from the released
per-model matrices, and report the per-model spread the 25-model average hides."""
import numpy as np, pandas as pd, re
from pathlib import Path
C = Path("/work/Pain-axis/results/3.3_validation/cosine_similarity")
ORDER = ["S1_pain","S2_pain","Fear","NegEmotion","NegWorld","BodySens","Arousal","Random","Numb","Sadness"]

def load(prefix):
    fs=[p for p in sorted(C.glob(f"similarity_{prefix}*_L*.csv")) if "MEAN" not in p.name
        and (prefix or not p.name.startswith(("similarity_alldenoise_","similarity_whitened_")))]
    mats={}
    for p in fs:
        m=re.match(rf"similarity_{prefix}(.+)_L(\d+)\.csv", p.name)
        df=pd.read_csv(p,index_col=0).reindex(index=ORDER,columns=ORDER)
        mats[m.group(1)]=df.values
    return fs, mats

pairs_paper = {
 ("S1_pain","S2_pain"):0.61, ("Fear","NegEmotion"):0.68, ("Fear","NegWorld"):0.59,
 ("NegEmotion","NegWorld"):0.73, ("Sadness","NegEmotion"):0.50, ("Sadness","NegWorld"):0.41,
 ("S1_pain","Fear"):0.09, ("S1_pain","NegEmotion"):0.06, ("S1_pain","NegWorld"):-0.07,
 ("S2_pain","Fear"):0.12, ("S2_pain","NegEmotion"):0.21, ("S2_pain","NegWorld"):0.03,
 ("Sadness","S2_pain"):0.38, ("Sadness","S1_pain"):0.26}

for variant,prefix in [("raw",""),("alldenoise","alldenoise_"),("whitened","whitened_")]:
    fs,mats = load(prefix)
    A=np.stack(list(mats.values()))
    mean=A.mean(0)
    print(f"\n########## {variant}: {len(fs)} per-model files")
    print(pd.DataFrame(mean,index=ORDER,columns=ORDER).round(3).to_string())
    if variant=="raw":
        print("\n--- paper line 346-353 values vs recomputed mean (and per-model spread) ---")
        for (a,b),v in pairs_paper.items():
            i,j=ORDER.index(a),ORDER.index(b)
            vals=A[:,i,j]
            print(f"  {a:11s} x {b:11s}  paper {v:+.2f}   mean {vals.mean():+.3f}   "
                  f"min {vals.min():+.3f} max {vals.max():+.3f} sd {vals.std():.3f}  "
                  f"{'OK' if abs(vals.mean()-v)<0.006 else '<-- MISMATCH'}")
        # per-model heterogeneity: how many models have S2xNegEmotion above S2xSadness etc.
        i_s1,i_s2=0,1
        print("\n  per-model: S1xS2 below 0.4 in:", [m for m,M in mats.items() if M[0,1]<0.4])
        for b in ["Fear","NegEmotion","NegWorld","Sadness"]:
            j=ORDER.index(b); v=A[:,i_s2,j]
            n=(v>0.3).sum()
            print(f"  S2 x {b:11s}: models with cos > 0.30: {n}/25  (>0.5: {(v>0.5).sum()})")
    # also the released MEAN file, as a check that my recompute matches theirs
    mf = C/f"similarity_{prefix}MEAN_all_models.csv"
    if mf.exists():
        their=pd.read_csv(mf,index_col=0).reindex(index=ORDER,columns=ORDER).values
        print(f"  max |my mean - released MEAN file| = {np.nanmax(np.abs(their-mean)):.5f}")

# robustness check 2: raw vs whitened over the 45 off-diagonal cells (paper lines 374-378)
_,raw=load(""); _,wh=load("whitened_")
R=np.stack(list(raw.values())).mean(0); W=np.stack(list(wh.values())).mean(0)
iu=np.triu_indices(10,1)
r=np.corrcoef(R[iu],W[iu])[0,1]
print("\n=== robustness check 2 (paper lines 374-378) ===")
print(f"  r(raw, whitened) over 45 off-diagonal cells = {r:.4f}   (paper: 0.992)")
print(f"  mean |delta| = {np.mean(np.abs(R[iu]-W[iu])):.4f}  (paper 0.020);  max |delta| = {np.max(np.abs(R[iu]-W[iu])):.4f} (paper 0.058)")
print(f"  S1xS2 raw {R[0,1]:+.3f} -> whitened {W[0,1]:+.3f}  (paper +0.61 -> +0.55)")
sign=[(ORDER[i],ORDER[j],R[i,j],W[i,j]) for i,j in zip(*iu) if np.sign(R[i,j])!=np.sign(W[i,j])]
print("  sign changes:", [(a,b,round(x,3),round(y,3)) for a,b,x,y in sign], "(paper: only S2 x BodySens, +0.04 -> -0.02)")

# robustness check 1 (paper lines 365-373)
_,ad=load("alldenoise_"); D=np.stack(list(ad.values())).mean(0)
print("\n=== robustness check 1, pooled-control denoising (paper lines 365-373) ===")
for (a,b),claim in [(("S1_pain","S2_pain"),"stays +0.61"),(("S1_pain","NegEmotion"),"+0.06 -> +0.13"),
                    (("S1_pain","NegWorld"),"-0.07 -> +0.02"),(("NegEmotion","NegWorld"),"+0.73 -> +0.40"),
                    (("Fear","NegEmotion"),"+0.68 -> +0.42")]:
    i,j=ORDER.index(a),ORDER.index(b)
    print(f"  {a:11s} x {b:11s}: raw {R[i,j]:+.3f} -> alldenoise {D[i,j]:+.3f}   paper: {claim}")
print("\n  full pain-row change under alldenoise:")
for b in ["Fear","NegEmotion","NegWorld","BodySens","Arousal","Random","Numb","Sadness"]:
    j=ORDER.index(b)
    print(f"    S1 x {b:11s} {R[0,j]:+.3f} -> {D[0,j]:+.3f} | S2 x {b:11s} {R[1,j]:+.3f} -> {D[1,j]:+.3f}")
