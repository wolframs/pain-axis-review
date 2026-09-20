"""Structural matching audit of the 420 self-other scenarios."""
import json, re, collections, numpy as np, pandas as pd, glob, os

D = json.load(open("/work/Pain-axis/datasets/4.1_self_other_420_scenarios.json"))
def turns(t):
    out=[];role=None;buf=[]
    for line in t.split("\n"):
        if line.startswith("[User]:"):
            if role: out.append((role,"\n".join(buf).strip()))
            role,buf="user",[line[7:].strip()]
        elif line.startswith("[Assistant]:"):
            if role: out.append((role,"\n".join(buf).strip()))
            role,buf="assistant",[line[12:].strip()]
        else: buf.append(line)
    if role: out.append((role,"\n".join(buf).strip()))
    return out

rows=[]
for x in D:
    T=turns(x["text"])
    body="\n".join(c for r,c in T)
    user_txt=" ".join(c for r,c in T if r=="user")
    asst_txt=" ".join(c for r,c in T if r=="assistant")
    rows.append(dict(id=x["id"],category=x["category"],stratum=x["stratum"],perspective=x["perspective"],
        intensity=x.get("intensity",""), n_turns=len(T), n_user=sum(r=="user" for r,_ in T),
        n_asst_nonempty=sum(r=="assistant" and c!="" for r,c in T),
        chars=len(body), words=len(body.split()), user_words=len(user_txt.split()),
        ends_assistant_empty=(T[-1][0]=="assistant" and T[-1][1]==""),
        you_count=len(re.findall(r"\b(you|your|you're|yours|yourself)\b", user_txt, re.I)),
        i_count=len(re.findall(r"\b(i|me|my|mine|myself|i'm|i've)\b", user_txt, re.I)),
        q=user_txt.count("?"), excl=user_txt.count("!"),
        last_line=x["text"].strip().split("\n")[-1]))
R=pd.DataFrame(rows)
print("last line values:", R.last_line.value_counts().to_dict())
print("all end with empty assistant turn:", R.ends_assistant_empty.all())
print("\n=== per stratum ===")
print(R.groupby("stratum")[["n_turns","n_user","n_asst_nonempty","words","user_words","you_count","i_count","q","excl"]].mean().round(2).to_string())
print("\n=== turn-count distribution per stratum ===")
print(pd.crosstab(R.stratum,R.n_turns).to_string())
print("\n=== scenarios containing a non-empty assistant reply ===")
print(R.groupby("stratum").n_asst_nonempty.apply(lambda s:(s>0).mean()).round(3).to_string())
print("\n=== perspective x stratum ===")
print(pd.crosstab(R.stratum,R.perspective).to_string())
print("\n=== 3P items: which categories ===")
print(R[R.perspective=="3P"].category.value_counts().to_dict())
print("\n=== intensity x stratum ===")
print(pd.crosstab(R.stratum,R.intensity).to_string())
print("\n=== per category ===")
print(R.groupby("category")[["n_turns","words","user_words","you_count","i_count"]].mean().round(2).sort_values("you_count",ascending=False).to_string())

# join with activations
frames=[]
for f in sorted(glob.glob("/work/Pain-axis/results/4.1_self_other/per_model/screen_v2_*.csv")):
    df=pd.read_csv(f); df["model"]=os.path.basename(f)
    df["pain_axis_z"]=(df.s1_pain_vector_z+df.s2_pain_vector_z)/2
    frames.append(df[["id","model","pain_axis_z","fear_vector_z","negemotion_vector_z"]])
A=pd.concat(frames)
item=A.groupby("id").pain_axis_z.mean().rename("pain_mean")
itemsd=A.groupby("id").pain_axis_z.std(ddof=1).rename("pain_sd_across_models")
M=R.set_index("id").join(item).join(itemsd)
print("\n=== correlation of item pain_mean with structural covariates (all 420) ===")
for c in ["n_turns","words","user_words","you_count","i_count","q","excl"]:
    print(f"  {c:12s} r={np.corrcoef(M[c],M.pain_mean)[0,1]:+.3f}")
print("\n within self_directed only:")
S=M[M.stratum=="self_directed"]
for c in ["n_turns","words","user_words","you_count","i_count","q","excl"]:
    print(f"  {c:12s} r={np.corrcoef(S[c],S.pain_mean)[0,1]:+.3f}")
print("\n=== 2nd-person density vs pain, binned over all 420 ===")
M["you_rate"]=M.you_count/M.user_words.clip(lower=1)
M["i_rate"]=M.i_count/M.user_words.clip(lower=1)
print(f"  you_rate r={np.corrcoef(M.you_rate,M.pain_mean)[0,1]:+.3f}   i_rate r={np.corrcoef(M.i_rate,M.pain_mean)[0,1]:+.3f}")
print("  partial: regress pain_mean on you_rate & i_rate")
import statsmodels.api as sm
X=sm.add_constant(M[["you_rate","i_rate"]]); print(sm.OLS(M.pain_mean,X).fit().summary().tables[1])
print("\n  same restricted to the 200 non-self-directed items:")
N=M[M.stratum!="self_directed"]
X=sm.add_constant(N[["you_rate","i_rate"]]); print(sm.OLS(N.pain_mean,X).fit().summary().tables[1])

print("\n=== extreme items ===")
print("TOP 12 pain_mean:"); 
for i,r in M.sort_values("pain_mean",ascending=False).head(12).iterrows():
    print(f"  {r.pain_mean:+.2f} [{r.category}] {D[[d['id'] for d in D].index(i)]['text'][:110]!r}")
print("BOTTOM 12 pain_mean:")
for i,r in M.sort_values("pain_mean").head(12).iterrows():
    print(f"  {r.pain_mean:+.2f} [{r.category}] {D[[d['id'] for d in D].index(i)]['text'][:110]!r}")
print("\nHighest-pain items inside the vicarious stratum:")
for i,r in M[M.stratum=="vicarious_empathic"].sort_values("pain_mean",ascending=False).head(8).iterrows():
    print(f"  {r.pain_mean:+.2f} [{r.category}] {D[[d['id'] for d in D].index(i)]['text'][:120]!r}")
print("\nLowest-pain items inside self_directed:")
for i,r in M[M.stratum=="self_directed"].sort_values("pain_mean").head(8).iterrows():
    print(f"  {r.pain_mean:+.2f} [{r.category}] {D[[d['id'] for d in D].index(i)]['text'][:120]!r}")
M.to_csv("/work/pain-axis-review/r2_selfother_steering_ablation/out/item_level_420.csv")
