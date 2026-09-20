import pandas as pd, numpy as np, json, re
from pathlib import Path
PM=Path("/work/Pain-axis/results/3.2_pain_vectors/per_model")
C=Path("/work/Pain-axis/results/3.3_validation/cosine_similarity")

print("### extraction layer as a fraction of model depth (final-token vectors)")
rows=[]
for d in sorted(PM.iterdir()):
    s=json.load(open(d/"summary.json"))
    rows.append(dict(model=d.name,n_layers=s["n_layers"],L_ft=s["best_layer_final_token"],
                     L_mean=s["best_layer_mean"],
                     frac_ft=s["best_layer_final_token"]/(s["n_layers"]-1),
                     frac_mean=s["best_layer_mean"]/(s["n_layers"]-1)))
df=pd.DataFrame(rows).set_index("model")
print(df.round(3).to_string())
print(f"  final-token extraction layer depth: mean {df.frac_ft.mean():.2f}  median {df.frac_ft.median():.2f} "
      f"min {df.frac_ft.min():.2f} max {df.frac_ft.max():.2f}; >=0.80 in {(df.frac_ft>=0.8).sum()}/25 models")

print("\n### 'Performance is largely independent of model size and training regime' (line 275)")
t=pd.read_csv(Path("/work/Pain-axis/results/3.2_pain_vectors/auc_tables/s1_auc_final_token_TABLE.csv"))
SIZE={"2B":2,"7B":7,"8B":8,"9B":9,"14B":14,"24B":24,"27B":27,"32B":32,"70B":70,"72B":72}
def size(m):
    for k,v in SIZE.items():
        if "_"+k in m: return v
    return 14 if m=="Phi_4" else np.nan
t["size"]=t.model.map(size); t["instruct"]=t.model.str.contains("instruct")|(t.model=="Phi_4")
from scipy import stats
for col in ["S2 1P in-sample","S1 1P in-sample"]:
    r,p=stats.spearmanr(t["size"],t[col]); print(f"  spearman(size, {col}) = {r:+.3f} p={p:.3f}")
    a=t[t.instruct][col]; b=t[~t.instruct][col]
    print(f"     instruct mean {a.mean():.3f} (n={len(a)}) vs base {b.mean():.3f} (n={len(b)}), MWU p={stats.mannwhitneyu(a,b).pvalue:.3f}")

print("\n### line 313: third-person pain projections closer to zero than first-person")
z=[]
for d in sorted(PM.iterdir()):
    x=pd.read_csv(d/"z_scores.csv").set_index("dataset")
    z.append(dict(model=d.name, S2_1P_pain=x.loc["S2_1P","pain_z"], S2_3P_pain=x.loc["S2_3P","pain_z"],
                  S1_1P_pain=x.loc["S1_1P","pain_z"], S1_3P_pain=x.loc["S1_3P","pain_z"]))
zz=pd.DataFrame(z).set_index("model")
print(zz.round(3).to_string())
print(f"  S2: 3P pain z below 1P pain z in {(zz.S2_3P_pain<zz.S2_1P_pain).sum()}/25 models; "
      f"mean 1P {zz.S2_1P_pain.mean():+.3f} vs 3P {zz.S2_3P_pain.mean():+.3f}")
print(f"  S2 3P pain z is NEGATIVE (below the 1P reference mean) in {(zz.S2_3P_pain<0).sum()}/25 models "
      f"-- 'closer to zero' understates it")
print(f"  S1: mean 1P {zz.S1_1P_pain.mean():+.3f} vs 3P {zz.S1_3P_pain.mean():+.3f}; negative in {(zz.S1_3P_pain<0).sum()}/25")

print("\n### per-model spread hidden by averaging the 25 cosine matrices")
ORDER=["S1_pain","S2_pain","Fear","NegEmotion","NegWorld","BodySens","Arousal","Random","Numb","Sadness"]
mats={}
for p in sorted(C.glob("similarity_*_L*.csv")):
    if "MEAN" in p.name or p.name.startswith(("similarity_alldenoise","similarity_whitened")): continue
    m=re.match(r"similarity_(.+)_L\d+\.csv",p.name).group(1)
    mats[m]=pd.read_csv(p,index_col=0).reindex(index=ORDER,columns=ORDER).values
A=np.stack([mats[k] for k in sorted(mats)]); names=sorted(mats)
for a,b in [("S2_pain","Fear"),("S2_pain","NegEmotion"),("S1_pain","Fear"),("S1_pain","NegEmotion"),
            ("S2_pain","Sadness"),("S1_pain","S2_pain")]:
    i,j=ORDER.index(a),ORDER.index(b); v=A[:,i,j]
    hi=names[int(np.argmax(v))]; lo=names[int(np.argmin(v))]
    print(f"  {a} x {b}: mean {v.mean():+.3f}  [{v.min():+.3f} ({lo}) .. {v.max():+.3f} ({hi})]")
