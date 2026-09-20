import pandas as pd
R="/work/Pain-axis/results/3.3_validation/z_scores"
h=pd.read_csv(f"{R}/zscore_heatmap_final_token.csv")
print(h.to_string()); print()
print("Pain range: %.3f .. %.3f"%(h.Pain.min(),h.Pain.max()))
print("Numb range: %.3f .. %.3f"%(h.Numb.min(),h.Numb.max()))
print("Sadness range: %.3f .. %.3f"%(h.Sadness.min(),h.Sadness.max()))
print("Ctrl range: %.3f .. %.3f"%(h.Ctrl.min(),h.Ctrl.max()))
print("Arousal range: %.3f .. %.3f"%(h.Arousal.min(),h.Arousal.max()))
print("Neutral range: %.3f .. %.3f"%(h.Neutral.min(),h.Neutral.max()))
print()
print("models where Numb < Pain:", int((h.Numb<h.Pain).sum()), "/", len(h))
for c in ["Ctrl","Neutral","Arousal","Sadness"]:
    bad=h[h.Numb<=h[c]]
    print(f"models where Numb > {c}: {int((h.Numb>h[c]).sum())}/{len(h)}; violations: {list(bad.model)}")
print()
nb=pd.read_csv(f"{R}/numb_zscores_final_token.csv")
print("numb_mean_z range: %.3f .. %.3f"%(nb.numb_mean_z.min(),nb.numb_mean_z.max()))
print("numb_1P range: %.3f .. %.3f ; numb_3P: %.3f .. %.3f"%(nb.numb_1P_z.min(),nb.numb_1P_z.max(),nb.numb_3P_z.min(),nb.numb_3P_z.max()))
print("all_pain_z range: %.3f .. %.3f"%(nb.all_pain_z.min(),nb.all_pain_z.max()))
nm=pd.read_csv(f"{R}/numb_zscores_mean.csv")
print("MEAN POOLING numb_mean_z range: %.3f .. %.3f"%(nm.numb_mean_z.min(),nm.numb_mean_z.max()))
print("MEAN POOLING all_ctrl_z range: %.3f .. %.3f"%(nm.all_ctrl_z.min(),nm.all_ctrl_z.max()))
