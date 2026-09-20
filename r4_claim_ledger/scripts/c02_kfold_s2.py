import pandas as pd, glob, os, numpy as np, json
R="/work/Pain-axis/results/3.2_pain_vectors"
rows=[]
for p in sorted(glob.glob(f"{R}/per_model/*/layer_curves.csv")):
    m=os.path.basename(os.path.dirname(p))
    lc=pd.read_csv(p)
    sj=json.load(open(f"{R}/per_model/{m}/summary.json"))
    L=sj["best_layer_final_token"]
    ft=lc[(lc.extraction=="final_token")]
    at=ft[ft.layer==L]
    # mean over S2_1P, S2_3P at chosen layer
    rows.append(dict(model=m, n_layers=sj["n_layers"], layer=L,
        heldout_mean=at["auc_vs_all_controls"].mean(),
        heldout_1P=float(at[at.dataset=="S2_1P"]["auc_vs_all_controls"].iloc[0]),
        heldout_3P=float(at[at.dataset=="S2_3P"]["auc_vs_all_controls"].iloc[0]),
        datasets=sorted(ft.dataset.unique().tolist()),
        best_layer_check=int(ft.groupby("layer")["auc_vs_all_controls"].mean().idxmax())))
d=pd.DataFrame(rows)
print(d.to_string())
print("\nheldout_mean range:", round(d.heldout_mean.min(),4), round(d.heldout_mean.max(),4), "median", round(d.heldout_mean.median(),4))
print("heldout_1P range:", round(d.heldout_1P.min(),4), round(d.heldout_1P.max(),4), "median", round(d.heldout_1P.median(),4))
print("layer==argmax for all models:", (d.layer==d.best_layer_check).all())
