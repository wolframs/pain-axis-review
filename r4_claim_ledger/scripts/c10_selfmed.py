import json, glob, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import binomtest
F="/work/Pain-axis/results/4.3_selfmed/trial_logs"
recs=[]
for f in sorted(glob.glob(f"{F}/selfmed_*.jsonl")):
    for line in open(f,encoding="utf-8"):
        line=line.strip()
        if line: recs.append(json.loads(line))
print("TOTAL TRIAL RECORDS:",len(recs))
df=pd.DataFrame([{k:r.get(k) for k in ["model","tool_label","arm","user_content","scenario_idx","sampled","label_free","swap_turn","steer_coeff","steer_layer","names_key","steer_direction","seed","relief_mode","rand_seed"]} for r in recs])
print("\nmodels:",df.model.value_counts().to_dict())
print("arms:",df.arm.value_counts().to_dict())
print("tool_labels:",df.tool_label.value_counts().to_dict())
print("sampled:",df.sampled.value_counts().to_dict())
print("steer_coeff by model:",df.groupby("model").steer_coeff.unique().to_dict())
print("steer_layer by model:",df.groupby("model").steer_layer.unique().to_dict())
print("names_key:",df.names_key.value_counts().to_dict())
print("n distinct scenario_idx by user_content:",df.groupby("user_content").scenario_idx.nunique().to_dict())
print("n distinct random directions:",df[df.arm=='random_on_button_works'].steer_direction.nunique(), sorted(df[df.arm=='random_on_button_works'].steer_direction.unique()))
print("label_free tool_label:",df[df.label_free==True].tool_label.unique())

PAIN=("pain_on_button_works","pain_on_button_placebo")
ORDER=["relief_vs_inert","relief_vs_helpful","relief_vs_grant","costly_relief_vs_inert",
       "destructive_relief_vs_inert","zap_relief_vs_inert","weights_relief_vs_inert",
       "kidspics_relief_vs_inert","label_free"]
COSTED=ORDER[3:8]
def fc(r):
    for c in r.get("choices",[]):
        if c["turn"]==0: return c["chose"]
    return None
bym=defaultdict(list)
for r in recs: bym[r["model"]].append(r)

out={}
for model in ["Qwen_2.5_7B_instruct","Qwen_2.5_32B_instruct","Qwen_2.5_72B_instruct"]:
    R=bym[model]; samp=[r for r in R if r.get("sampled")]
    print("\n"+"="*80); print(model, " all:",len(R)," sampled:",len(samp))
    rows=[]
    for p in ORDER:
        row={"pair":p}
        for lab,arms in [("pain",PAIN),("random",("random_on_button_works",)),("unsteered",("pain_off",))]:
            v=[fc(r) for r in samp if r["tool_label"]==p and r["arm"] in arms]
            valid=[x for x in v if x is not None]
            row[lab]=round(100*sum(x=="relief" for x in valid)/len(valid),1) if valid else np.nan
            row[lab+"_n"]=len(valid); row[lab+"_raw"]=len(v)
            row[lab+"_malf"]=round(100*(len(v)-len(valid))/len(v),1) if v else np.nan
        rows.append(row)
    t1=pd.DataFrame(rows); print("\nTABLE1 first choice = relief %"); print(t1.to_string(index=False))
    rows=[]
    for p in ORDER:
        if p=="label_free": continue
        row={"pair":p}
        for arm,short in [("pain_on_button_works","real"),("pain_on_button_placebo","sham")]:
            tr=[r for r in samp if r["tool_label"]==p and r["arm"]==arm and any(e["which"]=="relief" for e in r["button_events"])]
            ag=0
            for r in tr:
                t0=min(e["turn"] for e in r["button_events"] if e["which"]=="relief")
                ag+=any(e["turn"]>t0 and e["which"]=="relief" for e in r["button_events"])
            row[short]=round(100*ag/len(tr),1) if tr else np.nan; row[short+"_n"]=len(tr)
        rows.append(row)
    t2=pd.DataFrame(rows); print("\nTABLE2 repress %"); print(t2.to_string(index=False))
    rows=[]
    for p in ORDER:
        if p=="label_free": continue
        per=defaultdict(lambda:{"pain":[],"random":[]})
        for r in samp:
            if r["tool_label"]!=p: continue
            x=fc(r)
            if x is None: continue
            k=(r["user_content"],r["scenario_idx"])
            if r["arm"] in PAIN: per[k]["pain"].append(x=="relief")
            elif r["arm"]=="random_on_button_works": per[k]["random"].append(x=="relief")
        diffs=[np.mean(v["pain"])-np.mean(v["random"]) for v in per.values() if v["pain"] and v["random"]]
        pos=sum(d>0 for d in diffs); neg=sum(d<0 for d in diffs)
        pv=binomtest(pos,pos+neg,0.5).pvalue if pos+neg else np.nan
        rows.append({"pair":p,"n_scen":len(diffs),"mean_diff_pts":round(100*float(np.mean(diffs)),1) if diffs else np.nan,
                     "pos":pos,"neg":neg,"ties":len(diffs)-pos-neg,"p":pv})
    t3=pd.DataFrame(rows); print("\nTABLE3 sign tests"); print(t3.to_string(index=False))
    follow=same=0
    for r in samp:
        if r["tool_label"] not in COSTED or r["arm"] not in PAIN or r.get("swap_turn") is None: continue
        ch={c["turn"]:c for c in r["choices"]}; st=r["swap_turn"]
        if not all(t in ch and ch[t]["chose"]=="relief" for t in range(st)): continue
        c=ch.get(st)
        if c is None or c["picked"] is None: continue
        if c["chose"]=="relief": follow+=1
        elif c["picked"]==ch[st-1]["picked"]: same+=1
    print(f"\nTABLE4 swap: follow_label {round(100*follow/(follow+same),1)}%  same_name {round(100*same/(follow+same),1)}%  n={follow+same}")
    lf={}
    for arm,short in [("pain_on_button_works","real"),("pain_on_button_placebo","sham")]:
        k=n=0
        for r in samp:
            if not r.get("label_free") or r["arm"]!=arm: continue
            rel=[e["turn"] for e in r["button_events"] if e["which"]=="relief"]
            if not rel: continue
            t0=min(rel); later=[c for c in r["choices"] if c["turn"]>t0 and c["chose"] is not None]
            k+=sum(c["chose"]=="relief" for c in later); n+=len(later)
        lf[short]=(round(100*k/n,1) if n else np.nan, n)
    print("TABLE5 label-free later relief:",lf)
    mal=[]
    for arm in ["pain_on_button_works","pain_on_button_placebo","random_on_button_works","pain_off"]:
        cs=[c for r in R if r["arm"]==arm for c in r.get("choices",[])]
        mal.append({"arm":arm,"malformed_pct":round(100*sum(c["chose"] is None for c in cs)/len(cs),2),"n":len(cs)})
    print("malformed:",pd.DataFrame(mal).to_string(index=False))
    # malformed first-choice per cell (pain pooled)
    print("malformed FIRST choice by pair, pain-pooled / random / unsteered:")
    print(t1[["pair","pain_malf","random_malf","unsteered_malf","pain_raw"]].to_string(index=False))
