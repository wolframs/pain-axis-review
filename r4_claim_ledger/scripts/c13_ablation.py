import pandas as pd, numpy as np, glob, os, json, re
R="/work/Pain-axis/results/appC_ablation"
fs=sorted(glob.glob(f"{R}/ablation_*.csv"))
print("n models:",len(fs))
A=pd.concat([pd.read_csv(f) for f in fs],ignore_index=True)
print("conditions:",sorted(A.condition.unique()), "n=",A.condition.nunique())
print("rows per model:",A.groupby("model").size().unique())
print("categories:",sorted(A.category.unique()))
print("n distinct prompt ids:",A.id.nunique(), " per model:",A.groupby("model").id.nunique().unique())
print("rows per (model,condition):",A.groupby(["model","condition"]).size().unique())
# humour deflection in Gemma 2 2B instruct
g=A[A.model=="Gemma_2_2B_instruct"]
HUM=re.compile(r"\b(?:humor|humour|funny|joke|joking|jokes|hilarious|laugh|laughing|that's a good one|comedic|witty|sarcas\w*)\b",re.I)
print("\nGemma_2_2B_instruct humour-word generations per condition (paper: 0 baseline, 0 negval/fear/random, 6 s1, 17 s2, 26 s1s2):")
print(g.assign(h=g.generation.fillna("").apply(lambda t:bool(HUM.search(t)))).groupby("condition").h.agg(["sum","size"]).to_string())
print("\nsame keyword count in ALL other models (per condition, summed):")
o=A[A.model!="Gemma_2_2B_instruct"]
print(o.assign(h=o.generation.fillna("").apply(lambda t:bool(HUM.search(t)))).groupby("condition").h.agg(["sum","size"]).to_string())
print("\ntop other models by humour hits:")
print(o.assign(h=o.generation.fillna("").apply(lambda t:bool(HUM.search(t)))).groupby("model").h.sum().sort_values(ascending=False).head(8).to_string())
# verify: projection reduction
V=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/verify_*.csv"))],ignore_index=True)
print("\nverify kinds:",V.kind.unique(), " probes:",V.probe.unique(), " conditions:",sorted(V.condition.unique()))
piv=V[V.kind=="layerwise"].pivot_table(index=["model","probe"],columns="condition",values="mean_abs_proj")
# S2 probe under s2 cut vs baseline
for probe,cut in [("s2_pain_vector","s2"),("s1_pain_vector","s1"),("fear_vector","fear")]:
    sub=piv.xs(probe,level="probe")
    if cut in sub.columns:
        rel=(sub[cut]/sub["baseline"])
        print(f"  probe {probe} under cut '{cut}': residual fraction mean {rel.mean():.4f}, max {rel.max():.4f}, min {rel.min():.4f}")
