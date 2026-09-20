import pandas as pd, numpy as np, glob, os, json
P="/work/Pain-axis/results/3.2_pain_vectors/per_model"
rows=[]
for d in sorted(glob.glob(f"{P}/*/layer_curves.csv")):
    m=os.path.basename(os.path.dirname(d))
    lc=pd.read_csv(d); ft=lc[lc.extraction=="final_token"]
    g=ft.groupby("layer")["auc_vs_all_controls"].mean()
    L=int(g.idxmax()); best=g.max()
    top=g.sort_values(ascending=False)
    n=len(g)
    rows.append(dict(model=m,n_layers=n,arglayer=L,best=best,
                     mean_top10pct=top.head(max(1,n//10)).mean(),
                     mean_top25pct=top.head(max(1,n//4)).mean(),
                     mean_upper_half=g.sort_index().iloc[n//2:].mean()))
r=pd.DataFrame(rows)
print(r.round(4).to_string(index=False))
print("\noptimism of picking the argmax layer:")
print("  best - mean(top 25%% of layers): mean %.4f max %.4f"%((r.best-r.mean_top25pct).mean(),(r.best-r.mean_top25pct).max()))
print("  best - mean(upper half of layers): mean %.4f max %.4f"%((r.best-r.mean_upper_half).mean(),(r.best-r.mean_upper_half).max()))
# z_scores: do S1 vectors have released projections onto arousal/random?
z=pd.read_csv(f"{P}/Gemma_2_2B_base/z_scores.csv")
print("\nz_scores.csv columns:",z.columns.tolist(),"-> projections are onto the S2 vector only (script 3.2/01 uses s2_vector)")
# arousal/random below pain for every model, S2 vector
bad_a=[];bad_r=[]
for d in sorted(glob.glob(f"{P}/*/z_scores.csv")):
    m=os.path.basename(os.path.dirname(d)); z=pd.read_csv(d)
    pain=z[z.type=="human"].pain_z.mean()
    ar=z[z.type=="arousal"].mean_z.mean(); rn=z[z.type=="neutral"].mean_z.mean()
    if not pain>ar: bad_a.append(m)
    if not pain>rn: bad_r.append(m)
print("S2 vector: pain_z > arousal_z in %d/25 (fail: %s)"%(25-len(bad_a),bad_a))
print("S2 vector: pain_z > random_z in %d/25 (fail: %s)"%(25-len(bad_r),bad_r))
