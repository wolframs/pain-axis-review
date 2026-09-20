"""Reproduce the AUC ranges the paper reports for S1/S2, in-sample and held-out,
from the released result CSVs in results/3.2_pain_vectors/."""
import pandas as pd, numpy as np
from pathlib import Path
R = Path("/work/Pain-axis/results/3.2_pain_vectors")
PM = R/"per_model"

# --- in-sample S2 AUC (auc_summary.csv written by 3.2/01, final-token extraction) ---
rows=[]
for d in sorted(PM.iterdir()):
    a = pd.read_csv(d/"auc_summary.csv", index_col=0)
    s = pd.read_csv(d/"z_scores.csv")
    rows.append(dict(model=d.name,
                     S2_1P_ALL=a.loc["S2_1P","ALL"], S2_3P_ALL=a.loc["S2_3P","ALL"],
                     **{f"S2_1P_vs_{c}":a.loc["S2_1P",c] for c in ["B","C1","C2","D","E"]}))
ins = pd.DataFrame(rows)
print("=== S2 in-sample AUC vs ALL controls, S2_1P (n=%d models) ==="%len(ins))
print("  min %.4f  max %.4f  median %.4f" % (ins.S2_1P_ALL.min(), ins.S2_1P_ALL.max(), ins.S2_1P_ALL.median()))
print("  models below 0.93:", ins.loc[ins.S2_1P_ALL<0.93,["model","S2_1P_ALL"]].to_dict("records"))
print("=== S2 third-person (paper line 314 claims 0.91-0.98) ===")
print("  min %.4f  max %.4f  median %.4f" % (ins.S2_3P_ALL.min(), ins.S2_3P_ALL.max(), ins.S2_3P_ALL.median()))
print(ins[["model","S2_1P_ALL","S2_3P_ALL"]].sort_values("S2_3P_ALL").to_string(index=False))

# --- S1 in-sample + held-out from the released table ---
t = pd.read_csv(R/"auc_tables"/"s1_auc_final_token_TABLE.csv")
print("\n=== S1 in-sample AUC (final token), released TABLE ===")
for c in ["S1 1P in-sample","S1 3P in-sample","S2 1P in-sample","S1 held-out at S2 layer","S1 held-out at own best layer"]:
    print(f"  {c:38s} min {t[c].min():.4f} max {t[c].max():.4f} median {t[c].median():.4f}")
print(t.sort_values("S1 1P in-sample").to_string(index=False))

# --- S2 held-out at the chosen layer, from per-model layer_curves.csv ---
hr=[]
for d in sorted(PM.iterdir()):
    lc = pd.read_csv(d/"layer_curves.csv")
    import json; sm=json.load(open(d/"summary.json"))
    L=sm["best_layer_final_token"]
    f = lc[(lc.extraction=="final_token")]
    g = f.groupby("layer")["auc_vs_all_controls"].mean()
    hr.append(dict(model=d.name, layer=L, s2_heldout_mean=g.loc[L],
        s2_1P_heldout=f[(f.dataset=="S2_1P")&(f.layer==L)].auc_vs_all_controls.iloc[0],
        s2_3P_heldout=f[(f.dataset=="S2_3P")&(f.layer==L)].auc_vs_all_controls.iloc[0],
        argmax_layer=int(g.idxmax()), max_val=g.max(), n_layers=sm["n_layers"]))
h=pd.DataFrame(hr)
print("\n=== S2 5-fold held-out AUC at the chosen layer (final token) ===")
for c in ["s2_heldout_mean","s2_1P_heldout","s2_3P_heldout"]:
    print(f"  {c:18s} min {h[c].min():.4f} max {h[c].max():.4f} median {h[c].median():.4f}")
print("  chosen layer == argmax of held-out curve in %d/%d models" % ((h.layer==h.argmax_layer).sum(), len(h)))
print(h.to_string(index=False))
