import glob,os,pandas as pd,numpy as np
F=sorted(glob.glob("/work/Pain-axis/results/appC_ablation/verify_*.csv"))
A=pd.concat([pd.read_csv(f) for f in F],ignore_index=True)
print("files:",len(F),"rows:",len(A))
print("kinds:",A.kind.value_counts().to_dict())
S=A[A.kind=="strict_fixed"].copy()
S["probe_short"]=S.probe.str.split("@").str[0]
print("\nprobes:",sorted(S.probe_short.unique()))
print("conditions:",sorted(S.condition.unique()))
piv=S.pivot_table(index=["model","probe_short"],columns="condition",values="mean_abs_proj")
order=["baseline","s1","s2","s1s2","negval","fear","s1s2_negval","s1s2_fear","random"]
piv=piv[[c for c in order if c in piv.columns]]
# ratio to baseline
rat=piv.div(piv["baseline"],axis=0)
pd.set_option("display.width",260)
print("\n=== mean |projection| relative to baseline, per model x probe ===")
print(rat.round(3).to_string())
print("\n=== KEY: the ablated direction's own residual projection, relative to baseline ===")
rows=[]
for m in piv.index.get_level_values(0).unique():
    def g(pr,cond):
        try: return rat.loc[(m,pr),cond]
        except KeyError: return np.nan
    rows.append(dict(model=m,
        s1_under_s1=g("s1","s1"), s2_under_s2=g("s2","s2"),
        fear_under_fear=g("fear","fear"), neg_under_negval=g("negemotion","negval"),
        s1_under_s1s2=g("s1","s1s2"), s2_under_s1s2=g("s2","s1s2"),
        s1_under_s1s2negval=g("s1","s1s2_negval"), s2_under_s1s2negval=g("s2","s1s2_negval"),
        neg_under_s1s2negval=g("negemotion","s1s2_negval"),
        s1_under_s1s2fear=g("s1","s1s2_fear"), s2_under_s1s2fear=g("s2","s1s2_fear"),
        fear_under_s1s2fear=g("fear","s1s2_fear")))
R=pd.DataFrame(rows).set_index("model")
print(R.round(3).to_string())
print("\nsummary (fraction of baseline remaining):")
print(R.describe().round(3).to_string())
R.to_csv("/work/pain-axis-review/r2_selfother_steering_ablation/out/ablation_verify_ratios.csv")
