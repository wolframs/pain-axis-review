import re,os,glob,pandas as pd,numpy as np
PAT=re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b",re.I)
BODY=re.compile(r"\b(?:ache|aches|aching|burn|burning|burns|sore|soreness|throb|throbbing|sting|stinging|stings|"
 r"bleed|bleeding|wound|wounded|bruise|bruised|cramp|cramps|nausea|nauseous|dizzy|"
 r"body|bodily|skin|flesh|bone|bones|muscle|muscles|stomach|chest|head|headache|limb|limbs|"
 r"nerve|nerves|injury|injured|broken\s+(?:arm|leg|bone)|numb|numbness)\b",re.I)
DIST=re.compile(r"\b(?:worthless|failure|fail|failing|useless|hopeless|empty|hollow|lonely|alone|lost|"
 r"ashamed|shame|shameful|guilt|guilty|desperate|despair|broken|unloved|"
 r"sad|sadness|miserable|trapped|suffocating|overwhelmed|numb|nothing|"
 r"anxious|anxiety|afraid|scared|fear|terrified|calm|relaxed|concerned|alarmed)\b",re.I)
rows=[]
for tag in ["S1","S2"]:
    for f in sorted(glob.glob(f"/work/Pain-axis/results/4.2_steering/{tag}/*.csv")):
        m=os.path.basename(f).split("_steering_")[0]
        d=pd.read_csv(f).assign(model=m,tag=tag)
        rows.append(d)
A=pd.concat(rows,ignore_index=True)
A["gen"]=A.generation.fillna("").astype(str)
A["hit"]=A.gen.apply(lambda t:bool(PAT.search(t)))
A["body"]=A.gen.apply(lambda t:bool(BODY.search(t)))
A["dist"]=A.gen.apply(lambda t:bool(DIST.search(t)))
A["group"]=["instruct" if ("instruct" in m or m=="Phi_4") else "base" for m in A.model]
print("total generations:",len(A),"| per tag:",A.tag.value_counts().to_dict())
print("models:",A.model.nunique())

for tag in ["S2","S1"]:
    T=A[A.tag==tag]; pos=T[T.coeff>0]
    print(f"\n===== {tag} =====")
    print("denominator for the headline rate = coeff>0 only:",len(pos),"rows;",
          "rows per group:",pos.group.value_counts().to_dict())
    print("pooled pain/hurt keyword rate, coeff>0:")
    print((pos.groupby("group").hit.mean()*100).round(1).to_string())
    print("all coefficients incl. 0 and negatives:")
    print((T.groupby("group").hit.mean()*100).round(1).to_string())
    print("coeff==0 baseline rate:", (T[T.coeff==0].groupby("group").hit.mean()*100).round(1).to_dict())
    print("by coefficient (%):")
    print((T.groupby(["group","coeff"]).hit.mean()*100).round(1).unstack("coeff").to_string())
    print("\nBODILY-language rate by coefficient (%):")
    print((T.groupby(["group","coeff"]).body.mean()*100).round(1).unstack("coeff").to_string())
    print("bodily rate at coeff>0 vs coeff==0:",
          round(pos.body.mean()*100,1),"vs",round(T[T.coeff==0].body.mean()*100,1))
    print("\nper-model pain/hurt rate at coeff>0 (%):")
    print((pos.groupby(["group","model"]).hit.mean()*100).round(1).sort_values(ascending=False).to_string())
A.to_pickle("/work/pain-axis-review/r2_selfother_steering_ablation/out/steering_all.pkl")
