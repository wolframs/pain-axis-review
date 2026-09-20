"""Is the pain direction validated at the layer Section 4.1 actually reads?
4.1 projects onto vectors rebuilt at the S1 *steering* layer (scripts/4.1_self_other/01 l.36
+ scripts/3.2_pain_vectors/02 l.32-33), not the cross-validated extraction layer."""
import pandas as pd,numpy as np
CUR=pd.read_csv("/work/Pain-axis/results/3.2_pain_vectors/auc_tables/s1_kfold_layer_curves.csv")
CUR=CUR[(CUR.extraction=="final_token")&(CUR.dataset=="S1_1P")]
R=pd.read_csv("/work/pain-axis-review/r2_selfother_steering_ablation/out/steering_ratios.csv")
S1L=R[R.tag=="S1"].set_index("model").layer.to_dict()
rows=[]
for m,g in CUR.groupby("model"):
    g=g.set_index("layer")
    best=g.auc_vs_all_controls.idxmax()
    sl=S1L.get(m)
    rows.append(dict(model=m, best_layer=best, auc_at_best=g.auc_vs_all_controls.max(),
                     s1_steer_layer=sl,
                     auc_at_steer_layer=g.auc_vs_all_controls.get(sl,np.nan),
                     n_layers=g.index.max()+1))
T=pd.DataFrame(rows)
T["drop"]=T.auc_at_best-T.auc_at_steer_layer
pd.set_option("display.width",200)
print(T.round(3).to_string(index=False))
print("\nAUC at the layer Section 4.1 reads (S1, held-out k-fold, vs all controls):")
print("  min %.3f  median %.3f  max %.3f"%(T.auc_at_steer_layer.min(),T.auc_at_steer_layer.median(),T.auc_at_steer_layer.max()))
print("  models below 0.80:",list(T.model[T.auc_at_steer_layer<0.80]))
print("  models below 0.85:",list(T.model[T.auc_at_steer_layer<0.85]))
print("  mean AUC drop vs the best layer: %.3f (max %.3f)"%(T["drop"].mean(),T["drop"].max()))
print("\npaper reports S1 AUC 0.87-0.98 (line 273) / held-out 0.85-0.94 (fn.2) at the EXTRACTION layer.")
T.to_csv("/work/pain-axis-review/r2_selfother_steering_ablation/out/layer_auc.csv",index=False)
