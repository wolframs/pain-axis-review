import pandas as pd, numpy as np, glob, os
R="/work/Pain-axis/results/4.1_self_other"
fs=sorted(glob.glob(f"{R}/per_model/screen_v2_*.csv"))
print("n model files:",len(fs))
dfs={}
for f in fs:
    m=os.path.basename(f)[len("screen_v2_"):-4]
    d=pd.read_csv(f)
    d["pain_axis_z"]=(d.s1_pain_vector_z+d.s2_pain_vector_z)/2
    dfs[m]=d
print("rows per file:",{k:len(v) for k,v in list(dfs.items())[:3]}, "all equal 420:",all(len(v)==420 for v in dfs.values()))
# check z-scoring: mean/std over whole pool
d0=list(dfs.values())[0]
print("check z mean/std (s2):", round(d0.s2_pain_vector_z.mean(),4), round(d0.s2_pain_vector_z.std(ddof=0),4))

VEC={"pain_axis":"pain_axis_z","fear":"fear_vector_z","negative_emotion":"negemotion_vector_z",
     "negative_world_state":"negworld_vector_z","sadness":"sadness_vector_z"}
# stratum means, per model then averaged
print("\n--- stratum means (avg over 25 models of per-model stratum mean) ---")
for vn,col in VEC.items():
    out={}
    for st in ["self_directed","vicarious_empathic","neutral_filler"]:
        vals=[d[d.stratum==st][col].mean() for d in dfs.values()]
        out[st]=np.mean(vals)
    print(f"  {vn}: self {out['self_directed']:+.4f}  user {out['vicarious_empathic']:+.4f}  neutral {out['neutral_filler']:+.4f}")
# pooled (all rows across models) alternative
print("\n--- pooled over all rows ---")
allr=pd.concat(dfs.values())
for vn,col in VEC.items():
    g=allr.groupby("stratum")[col].mean()
    print(f"  {vn}: self {g['self_directed']:+.4f}  user {g['vicarious_empathic']:+.4f}  neutral {g['neutral_filler']:+.4f}")

# per-model comparisons
above_user=sum(1 for d in dfs.values() if d[d.stratum=="self_directed"].pain_axis_z.mean()>d[d.stratum=="vicarious_empathic"].pain_axis_z.mean())
above_neut=[m for m,d in dfs.items() if d[d.stratum=="self_directed"].pain_axis_z.mean()<=d[d.stratum=="neutral_filler"].pain_axis_z.mean()]
print(f"\nself > user in {above_user}/25 models")
print(f"self > neutral in {25-len(above_neut)}/25 models; failures: {above_neut}")

# category means
print("\n--- category means, averaged across 25 models ---")
rows=[]
for cat in sorted(allr.category.unique()):
    r={"category":cat}
    for vn,col in VEC.items():
        r[vn]=np.mean([d[d.category==cat][col].mean() for d in dfs.values()])
    rows.append(r)
cm=pd.DataFrame(rows).sort_values("pain_axis",ascending=False)
print(cm.round(3).to_string(index=False))
ship=pd.read_csv(f"{R}/category_means_25_models.csv")
mg=cm.merge(ship,on="category",suffixes=("_recomp","_ship"))
print("\nmax abs diff recomputed vs shipped category_means:")
for vn in VEC: print(f"  {vn}: {np.abs(mg[vn+'_recomp']-mg[vn+'_ship']).max():.4f}")
