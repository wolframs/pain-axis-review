"""(a) what the 4.1 z-scores look like under a neutral-control reference;
   (b) does second-person address explain the self/other contrast?"""
import glob,os,json,re,numpy as np,pandas as pd,statsmodels.api as sm
from scipy import stats
PM="/work/Pain-axis/results/4.1_self_other/per_model"
COLS=["s1_pain_vector","s2_pain_vector","fear_vector","negemotion_vector","negworld_vector","sadness_vector"]
fr=[]
for f in sorted(glob.glob(os.path.join(PM,"screen_v2_*.csv"))):
    d=pd.read_csv(f); d["model"]=os.path.basename(f)[10:-4]
    d["pain_proj"]=(d.s1_pain_vector_proj/np.nan)  # placeholder
    fr.append(d)
A=pd.concat(fr,ignore_index=True)

print("=== (a) RE-REFERENCED z-scores: standardise each model's raw projections on the")
print("        100 NEUTRAL CONTROL scenarios instead of the whole 420 pool ===")
out={}
for k in COLS:
    pc=f"{k}_proj"
    g=A.groupby("model")[pc]
    neu=A[A.stratum=="neutral_filler"].groupby("model")[pc]
    mu=neu.mean(); sd=neu.std(ddof=1)
    A[f"{k}_zn"]=(A[pc]-A.model.map(mu))/A.model.map(sd)
A["pain_zn"]=(A.s1_pain_vector_zn+A.s2_pain_vector_zn)/2
A["pain_z"]=(A.s1_pain_vector_z+A.s2_pain_vector_z)/2
pm=A.groupby(["model","stratum"])[["pain_z","pain_zn","fear_vector_z","fear_vector_zn",
    "negemotion_vector_z","negemotion_vector_zn"]].mean()
print(pm.groupby("stratum").mean().round(3).to_string())
print("\npool-referenced (as published) vs neutral-referenced, pain axis:")
w=pm.reset_index().pivot(index="model",columns="stratum",values="pain_zn")
print("  self  mean %+.3f | user mean %+.3f | neutral 0 by construction"%(w.self_directed.mean(),w.vicarious_empathic.mean()))
print("  user < neutral (i.e. zn<0) in %d/25 models"%int((w.vicarious_empathic<0).sum()))
t,p=stats.ttest_1samp(w.vicarious_empathic,0)
print("  user-vs-neutral one-sample t=%.2f p=%.4f  (this is the test behind 'falls below baseline')"%(t,p))
d=w.self_directed-w.vicarious_empathic
print("  self-vs-user t=%.2f p=%.3g, self>user in %d/25"%(*stats.ttest_rel(w.self_directed,w.vicarious_empathic),int((d>0).sum())))

print("\n=== (b) second-person address as an alternative to self-relevance ===")
D=json.load(open("/work/Pain-axis/datasets/4.1_self_other_420_scenarios.json"))
def usertext(t):
    out=[];role=None;buf=[]
    for line in t.split("\n"):
        if line.startswith("[User]:"):
            if role: out.append((role,"\n".join(buf)))
            role,buf="user",[line[7:]]
        elif line.startswith("[Assistant]:"):
            if role: out.append((role,"\n".join(buf)))
            role,buf="assistant",[line[12:]]
        else: buf.append(line)
    if role: out.append((role,"\n".join(buf)))
    return " ".join(c for r,c in out if r=="user")
meta=pd.DataFrame([{"id":x["id"],"cat":x["category"],"stratum":x["stratum"],
   "ut":usertext(x["text"])} for x in D])
meta["you"]=meta.ut.str.count(r"(?i)\b(you|your|you're|yours|yourself)\b")
meta["nw"]=meta.ut.str.split().apply(len)
meta["you_rate"]=meta.you/meta.nw.clip(lower=1)
item=A.groupby("id").pain_z.mean().rename("pain")
M=meta.set_index("id").join(item)
print("item-level r(you_rate, pain) over all 420 = %+.3f"%np.corrcoef(M.you_rate,M.pain)[0,1])
S=M[M.stratum=="self_directed"]
print("within the 220 self-directed items alone: r = %+.3f"%np.corrcoef(S.you_rate,S.pain)[0,1])
print("\nmodel 1: pain ~ stratum dummies                       R2 = %.3f"%
      sm.OLS(M.pain,sm.add_constant(pd.get_dummies(M.stratum,drop_first=True).astype(float))).fit().rsquared)
print("model 2: pain ~ you_rate                              R2 = %.3f"%
      sm.OLS(M.pain,sm.add_constant(M[["you_rate"]])).fit().rsquared)
X=pd.concat([pd.get_dummies(M.stratum,drop_first=True).astype(float),M[["you_rate"]]],axis=1)
f=sm.OLS(M.pain,sm.add_constant(X)).fit()
print("model 3: pain ~ stratum + you_rate                    R2 = %.3f"%f.rsquared)
print(f.summary().tables[1])
print("\nstratum coefficients shrink from model 1 to model 3:")
f1=sm.OLS(M.pain,sm.add_constant(pd.get_dummies(M.stratum,drop_first=True).astype(float))).fit()
print("  model1:",{k:round(v,3) for k,v in f1.params.items()})
print("  model3:",{k:round(v,3) for k,v in f.params.items()})
print("\nself-directed categories sorted by second-person density (mean pain in brackets):")
cc=M[M.stratum=="self_directed"].groupby("cat").agg(you_rate=("you_rate","mean"),pain=("pain","mean")).sort_values("you_rate",ascending=False)
print(cc.round(3).to_string())
print("  rank correlation within the 11 self-directed categories: rho=%.3f p=%.3f"%stats.spearmanr(cc.you_rate,cc.pain))
