import glob,os,numpy as np,pandas as pd
from scipy import stats
PM="/work/Pain-axis/results/4.1_self_other/per_model"
fr=[]
for f in sorted(glob.glob(os.path.join(PM,"screen_v2_*.csv"))):
    d=pd.read_csv(f); d["model"]=os.path.basename(f)[10:-4]; fr.append(d)
A=pd.concat(fr,ignore_index=True)
A["pain"]=(A.s1_pain_vector_z+A.s2_pain_vector_z)/2
A["grp"]=np.where(A.model.str.contains("base"),"base","instruct")
pm=A.groupby(["grp","model","stratum"])[["pain","fear_vector_z","negemotion_vector_z"]].mean().reset_index()
for g in ["base","instruct"]:
    P=pm[pm.grp==g].pivot(index="model",columns="stratum",values="fear_vector_z")
    Q=pm[pm.grp==g].pivot(index="model",columns="stratum",values="negemotion_vector_z")
    R=pm[pm.grp==g].pivot(index="model",columns="stratum",values="pain")
    print(f"\n== {g} (n={len(P)}) ==")
    for nm,X in [("pain",R),("fear",P),("negemo",Q)]:
        d=X.vicarious_empathic-X.self_directed
        t,p=stats.ttest_rel(X.vicarious_empathic,X.self_directed)
        print(f"  {nm:7s} self={X.self_directed.mean():+.3f} user={X.vicarious_empathic.mean():+.3f} "
              f"neutral={X.neutral_filler.mean():+.3f} | user-self={d.mean():+.3f} t={t:+.2f} p={p:.3g} "
              f"user>self in {int((d>0).sum())}/{len(d)}")
