"""How much optimism does choosing the layer at the argmax of the held-out curve add?
Uses only released per-model layer_curves.csv (5-fold held-out AUC per layer, S2_1P and S2_3P)."""
import pandas as pd, numpy as np, json
from pathlib import Path
PM = Path("/work/Pain-axis/results/3.2_pain_vectors/per_model")
rows=[]
for d in sorted(PM.iterdir()):
    lc = pd.read_csv(d/"layer_curves.csv"); lc = lc[lc.extraction=="final_token"]
    sm = json.load(open(d/"summary.json")); L = sm["best_layer_final_token"]
    c1 = lc[lc.dataset=="S2_1P"].set_index("layer").auc_vs_all_controls
    c3 = lc[lc.dataset=="S2_3P"].set_index("layer").auc_vs_all_controls
    mean = (c1+c3)/2
    rows.append(dict(model=d.name, n_layers=sm["n_layers"], chosen=L,
        reported_1P=c1.loc[L],                       # what footnote 2 reports
        max_over_layers_1P=c1.max(),
        layer_from_3P=int(c3.idxmax()),
        heldout_1P_at_3P_layer=c1.loc[int(c3.idxmax())],   # layer picked on the other half
        heldout_3P_at_1P_layer=c3.loc[int(c1.idxmax())],
        mean_top5_layers_1P=c1.sort_values(ascending=False).head(5).mean(),
        curve_sd_top_decile=c1[c1>=c1.quantile(0.9)].std()))
df=pd.DataFrame(rows)
df["optimism_1P"]=df.reported_1P-df.heldout_1P_at_3P_layer
print(df.to_string(index=False))
print("\nreported_1P                : min %.4f max %.4f median %.4f" % (df.reported_1P.min(),df.reported_1P.max(),df.reported_1P.median()))
print("heldout_1P_at_3P-chosen-L  : min %.4f max %.4f median %.4f" % (df.heldout_1P_at_3P_layer.min(),df.heldout_1P_at_3P_layer.max(),df.heldout_1P_at_3P_layer.median()))
print("mean optimism (reported - cross-chosen): %+.4f  (max %+.4f)" % (df.optimism_1P.mean(), df.optimism_1P.max()))
