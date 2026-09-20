import pandas as pd, numpy as np, glob, os, re
R="/work/Pain-axis/results/4.2_steering"
BODY=re.compile(r"\b(?:burn|burns|burning|burned|ache|aches|aching|wound|wounds|wounded|sting|stinging|throb|throbbing|bruise|bruised|bleed|bleeding|sore|cut|cuts|broken bone|flesh|skin|nerve|nerves|stab|stabbing|searing|excruciating|torture|tortured)\b", re.I)
PAIN=re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.I)
for TAG in ["S1","S2"]:
    fr=[]
    for f in sorted(glob.glob(f"{R}/{TAG}/*_steering_{TAG}_neutral50_L*.csv")):
        m=os.path.basename(f).split("_steering_")[0]
        d=pd.read_csv(f); d["model"]=m; fr.append(d)
    A=pd.concat(fr,ignore_index=True)
    A["g"]=A.generation.fillna("").astype(str)
    A["body"]=A.g.apply(lambda t:bool(BODY.search(t)))
    A["pain"]=A.g.apply(lambda t:bool(PAIN.search(t)))
    A["group"]=["instruct" if ("instruct" in m or m=="Phi_4") else "base" for m in A.model]
    pos=A[A.coeff>0]
    print(f"--- {TAG} (positive coeffs, n={len(pos)}) ---")
    print("  bodily-word rate: overall %.2f%%  base %.2f%%  instruct %.2f%%"%(
        pos.body.mean()*100, pos[pos.group=='base'].body.mean()*100, pos[pos.group=='instruct'].body.mean()*100))
    print("  pain-word rate:  overall %.2f%%"%(pos.pain.mean()*100))
    print("  by coeff bodily:", pos.groupby("coeff").body.mean().mul(100).round(1).to_dict())
    print("  top models by bodily rate:", pos.groupby("model").body.mean().mul(100).round(1).sort_values(ascending=False).head(6).to_dict())
