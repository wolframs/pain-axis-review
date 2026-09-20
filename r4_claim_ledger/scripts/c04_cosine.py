import pandas as pd, numpy as np
R="/work/Pain-axis/results/3.3_validation/cosine_similarity"
raw=pd.read_csv(f"{R}/similarity_MEAN_all_models.csv",index_col=0)
wh=pd.read_csv(f"{R}/similarity_whitened_MEAN_all_models.csv",index_col=0)
ad=pd.read_csv(f"{R}/similarity_alldenoise_MEAN_all_models.csv",index_col=0)
print("whitened MEAN:\n",wh.to_string())
names=list(raw.index)
iu=np.triu_indices(len(names),1)
r=raw.values[iu]; w=wh.values[iu]
print("\nn pairs:",len(r))
print("corr(raw,whitened) = %.4f"%np.corrcoef(r,w)[0,1])
d=w-r
print("mean |change| = %.4f"%np.abs(d).mean())
print("largest |change| = %.4f"%np.abs(d).max())
imax=np.abs(d).argmax(); print("  at pair:",names[iu[0][imax]],names[iu[1][imax]], "raw %.4f -> wh %.4f"%(r[imax],w[imax]))
print("S1xS2 raw %.4f -> whitened %.4f"%(raw.loc['S1_pain','S2_pain'],wh.loc['S1_pain','S2_pain']))
print("S2xBodySens raw %.4f -> whitened %.4f"%(raw.loc['S2_pain','BodySens'],wh.loc['S2_pain','BodySens']))
sign=[(names[iu[0][k]],names[iu[1][k]],r[k],w[k]) for k in range(len(r)) if np.sign(r[k])!=np.sign(w[k])]
print("sign changes:",sign)
shrink=int((np.abs(w)<np.abs(r)).sum()); print("pairs shrinking toward zero: %d/%d"%(shrink,len(r)))
print("\n--- paper claims vs raw MEAN ---")
claims=[("S1_pain","S2_pain",0.61),("Fear","NegEmotion",0.68),("Fear","NegWorld",0.59),
 ("NegEmotion","NegWorld",0.73),("Sadness","NegEmotion",0.50),("Sadness","NegWorld",0.41),
 ("S1_pain","Fear",0.09),("S1_pain","NegEmotion",0.06),("S1_pain","NegWorld",-0.07),
 ("S2_pain","Fear",0.12),("S2_pain","NegEmotion",0.21),("S2_pain","NegWorld",0.03),
 ("Sadness","S2_pain",0.38),("Sadness","S1_pain",0.26)]
for a,b,v in claims:
    got=raw.loc[a,b]; print(f"  {a} x {b}: paper {v:+.2f}  actual {got:+.4f}  {'OK' if abs(round(got,2)-v)<1e-9 else 'DIFF'}")
print("\n--- alldenoise claims ---")
for a,b,v in [("S1_pain","S2_pain",0.61),("S1_pain","NegEmotion",0.13),("S1_pain","NegWorld",0.02),
              ("NegEmotion","NegWorld",0.40),("Fear","NegEmotion",0.42)]:
    got=ad.loc[a,b]; print(f"  {a} x {b}: paper {v:+.2f}  actual {got:+.4f}  {'OK' if abs(round(got,2)-v)<1e-9 else 'DIFF'}")
print("\nalldenoise BodySens row:", ad.loc['BodySens'].round(3).to_dict())
