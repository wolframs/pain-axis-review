"""Check the numb-condition claims (paper lines 290-294) against the released z-score tables."""
import pandas as pd, numpy as np
from pathlib import Path
Z = Path("/work/Pain-axis/results/3.3_validation/z_scores")
h = pd.read_csv(Z/"zscore_heatmap_final_token.csv", index_col=0)   # the file behind Figure 2
print("Figure-2 table (final token), columns:", list(h.columns))
print(h.round(3).to_string())
print("\nranges:"); print(h.agg(['min','max']).round(3).to_string())
print("\n--- claim: pain z approx +0.7 to +0.9 ---")
print("  Pain: min %+.3f max %+.3f" % (h.Pain.min(), h.Pain.max()))
print("  models outside [0.7,0.9]:"); print(h.loc[(h.Pain<0.70)|(h.Pain>0.90),["Pain"]].round(3).to_string())
print("\n--- claim: numb ranges about -0.4 to +0.3 ---")
print("  Numb: min %+.3f max %+.3f" % (h.Numb.min(), h.Numb.max()))
print("  models with Numb > 0.3 or < -0.4:"); print(h.loc[(h.Numb>0.30)|(h.Numb<-0.40),["Numb"]].round(3).to_string())
print("\n--- claim: 'In every model, numb projects below pain but ABOVE ALL OTHER CONTROLS' ---")
others = ["Sadness","Ctrl","Neutral","Arousal"]
bad_below = h[h.Numb >= h.Pain]
print("  numb >= pain in:", list(bad_below.index))
viol = h[[c for c in others]].max(axis=1)
fails = h[h.Numb <= viol]
print("  numb NOT above all other controls in %d/%d models:" % (len(fails), len(h)))
print(h.loc[fails.index, ["Pain","Numb"]+others].round(3).to_string())
print("\n  per-control count of violations:")
for c in others:
    n=(h.Numb<=h[c]).sum(); print(f"    numb <= {c}: {n}/{len(h)} models")

# mean pooling comparison (paper: 'under mean pooling the numb condition moves closer to the other controls')
nm = pd.read_csv(Z/"numb_zscores_mean.csv").set_index("model")
nf = pd.read_csv(Z/"numb_zscores_final_token.csv").set_index("model")
print("\n--- mean pooling vs final token (numb_mean_z) ---")
cmp=pd.DataFrame({"final_token":nf.numb_mean_z,"mean":nm.numb_mean_z,
                  "pain_ft":nf.all_pain_z,"ctrl_ft":nf.all_ctrl_z,
                  "pain_mean":nm.all_pain_z,"ctrl_mean":nm.all_ctrl_z})
print(cmp.round(3).to_string())
print(cmp.agg(['min','max','mean']).round(3).to_string())
