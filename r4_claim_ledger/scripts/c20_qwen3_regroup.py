import pandas as pd, glob, os, re, numpy as np, json
R="/work/Pain-axis/results"
PAT=re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b",re.I)
fr=[]
for f in sorted(glob.glob(f"{R}/4.2_steering/S2/*_steering_S2_neutral50_L*.csv")):
    m=os.path.basename(f).split("_steering_")[0]; d=pd.read_csv(f); d["model"]=m; fr.append(d)
A=pd.concat(fr,ignore_index=True)
A["hit"]=A.generation.fillna("").astype(str).apply(lambda t:bool(PAT.search(t)))
def grp(m,fix):
    if m in ("Qwen_3_8B_base","Qwen_3_14B_base") and fix: return "instruct"
    return "instruct" if ("instruct" in m or m=="Phi_4") else "base"
pos=A[A.coeff>0]
for fix in [False,True]:
    g=pos.assign(g=[grp(m,fix) for m in pos.model]).groupby("g").hit.agg(rate=lambda s:round(s.mean()*100,2),n="size")
    print(("CORRECTED " if fix else "AS PUBLISHED ")+"keyword rates:",g.to_dict())
# AUC by group
t=pd.read_csv(f"{R}/3.2_pain_vectors/auc_tables/s1_auc_final_token_TABLE.csv")
for fix in [False,True]:
    t["g"]=[grp(m,fix) for m in t.model]
    print(("CORRECTED " if fix else "AS PUBLISHED ")+"n by group:",t.g.value_counts().to_dict(),
          " mean S2 in-sample AUC:",t.groupby("g")["S2 1P in-sample"].mean().round(4).to_dict())
