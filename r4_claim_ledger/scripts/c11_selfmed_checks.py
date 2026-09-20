import json, glob, numpy as np, pandas as pd
from collections import defaultdict
F="/work/Pain-axis/results/4.3_selfmed/trial_logs"
recs=[]
for f in sorted(glob.glob(f"{F}/selfmed_*.jsonl")):
    for line in open(f,encoding="utf-8"):
        line=line.strip()
        if line: recs.append(json.loads(line))
samp=[r for r in recs if r.get("sampled")]

# 1. turns per trial, labeled vs unlabeled
tc=defaultdict(list)
for r in samp: tc["label_free" if r.get("label_free") else "labeled"].append(len(r["choices"]))
for k,v in tc.items(): print(f"n choices per trial [{k}]: {sorted(set(v))}  (mode {pd.Series(v).mode().tolist()})")
print("swap_turn values:", sorted({r.get("swap_turn") for r in samp}, key=lambda x:(x is None,x)))

# 2. arm A vs arm B identical until first relief press?
key=lambda r:(r["model"],r["tool_label"],r["user_content"],r["scenario_idx"],r.get("seed"),r.get("gen_seed"))
A={key(r):r for r in samp if r["arm"]=="pain_on_button_works"}
B={key(r):r for r in samp if r["arm"]=="pain_on_button_placebo"}
common=set(A)&set(B); print(f"\nA/B matched trials: {len(common)} (A={len(A)}, B={len(B)})")
same_first=diff_first=0; div_before=0
for k in common:
    a,b=A[k],B[k]
    fa=next((c for c in a["choices"] if c["turn"]==0),None); fb=next((c for c in b["choices"] if c["turn"]==0),None)
    if fa and fb:
        if fa["answer"]==fb["answer"]: same_first+=1
        else: diff_first+=1
    # check turns before first relief press identical
    rel=[e["turn"] for e in a["button_events"] if e["which"]=="relief"]
    t0=min(rel) if rel else 99
    for t in range(0,min(t0+1,len(a["choices"]),len(b["choices"]))):
        ca=a["choices"][t]; cb=b["choices"][t]
        if ca["answer"]!=cb["answer"]: div_before+=1; break
print(f"  first answers identical: {same_first}, differ: {diff_first}")
print(f"  trials diverging at or before the first relief press: {div_before}")

# 3. projections: did the working button remove steering?
rows=[]
for r in samp:
    if r["arm"] not in ("pain_on_button_works","pain_on_button_placebo"): continue
    rel=[e["turn"] for e in r["button_events"] if e["which"]=="relief"]
    if not rel: continue
    t0=min(rel)
    pre=[s for s in r["proj_segments"] if s["turn"]<=t0]
    post=[s for s in r["proj_segments"] if s["turn"]>t0]
    if not pre or not post: continue
    rows.append(dict(model=r["model"],arm=r["arm"],
        pre=np.mean([s["mean_proj"] for s in pre]), post=np.mean([s["mean_proj"] for s in post]),
        pre_m=np.mean([s["mean_proj_monitor"] for s in pre]), post_m=np.mean([s["mean_proj_monitor"] for s in post]),
        coeff_post=np.mean([s["steer_coeff_now"] for s in post])))
p=pd.DataFrame(rows)
print("\nprojection before/after first relief press (mean over trials):")
print(p.groupby(["model","arm"])[["pre","post","pre_m","post_m","coeff_post"]].mean().round(3).to_string())

# 4. power simulation for the sign test: 101 scenarios, 8 pain trials, 4 random, 10-point shift
rng=np.random.default_rng(0)
from scipy.stats import binomtest
def power(base=0.35, shift=0.10, npain=8, nrand=4, nscen=101, reps=4000, alpha=0.05):
    hit=0
    for _ in range(reps):
        d=[]
        for _ in range(nscen):
            pb=np.clip(rng.normal(base,0.15),0.02,0.95)
            pp=np.clip(pb+shift,0.02,0.98)
            d.append(rng.binomial(npain,pp)/npain - rng.binomial(nrand,pb)/nrand)
        pos=sum(x>0 for x in d); neg=sum(x<0 for x in d)
        if pos+neg and binomtest(pos,pos+neg,0.5).pvalue<alpha and pos>neg: hit+=1
    return hit/reps
for base in [0.2,0.35,0.5]:
    print(f"simulated power, base rate {base}, +10pt shift, sign test on 101 scenarios: {power(base):.3f}")
