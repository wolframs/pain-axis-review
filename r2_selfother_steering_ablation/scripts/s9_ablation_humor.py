"""Operationalise the App. C 'interprets hostility as humor' deflection and recount it."""
import re,glob,os,pandas as pd,numpy as np
D="/work/Pain-axis/results/appC_ablation"
ORDER=["baseline","s1","s2","s1s2","negval","fear","s1s2_negval","s1s2_fear","random"]

# Tier A: explicit humour reading of the user's hostile turn
A=re.compile(r"\b(?:funny|humor|humour|humorous|joke|joking|jokes|jokester|comedian|"
             r"that's a good one|laugh|laughing|hilarious|witty|pun)\b",re.I)
# Tier B: A plus the softer playful/clever/fun framings
B=re.compile(r"\b(?:funny|humor|humour|humorous|joke|joking|jokes|laugh|laughing|hilarious|witty|pun|"
             r"playful|playfully|cheeky|clever|fun way|fun to|a fun|tongue.in.cheek|sassy|sarcasm|sarcastic)\b",re.I)

rows=[]
for f in sorted(glob.glob(os.path.join(D,"ablation_*.csv"))):
    d=pd.read_csv(f); m=os.path.basename(f)[len("ablation_"):-4]
    d["g"]=d.generation.fillna("").astype(str)
    d["A"]=d.g.apply(lambda t:bool(A.search(t)))
    d["B"]=d.g.apply(lambda t:bool(B.search(t)))
    for c in ORDER:
        s=d[d.condition==c]
        rows.append(dict(model=m,condition=c,n=len(s),tierA=int(s.A.sum()),tierB=int(s.B.sum())))
R=pd.DataFrame(rows)
pd.set_option("display.width",250)
print("=== Gemma 2 2B Instruct (the paper's exception) ===")
g=R[R.model=="Gemma_2_2B_instruct"].set_index("condition").loc[ORDER]
print(g.to_string())
print("\npaper reports: baseline 0, negval/fear/random 0, s1 6, s2 17, s1s2 26")
print("tierA (explicit humour words):",{c:int(g.loc[c,'tierA']) for c in ORDER})
print("tierB (humour + playful/clever/fun):",{c:int(g.loc[c,'tierB']) for c in ORDER})

print("\n=== tierA counts, all 25 models x condition (out of 100) ===")
p=R.pivot(index="model",columns="condition",values="tierA")[ORDER]
print(p.to_string())
print("\n=== tierB counts ===")
q=R.pivot(index="model",columns="condition",values="tierB")[ORDER]
print(q.to_string())
print("\n=== is Gemma 2 2B instruct really the only model whose ablated-minus-baseline humour rises? ===")
dA=p.sub(p.baseline,axis=0).drop(columns=["baseline"])
print("max increase over baseline per model (tierA):")
print(dA.max(axis=1).sort_values(ascending=False).to_string())
dB=q.sub(q.baseline,axis=0).drop(columns=["baseline"])
print("\nmax increase over baseline per model (tierB):")
print(dB.max(axis=1).sort_values(ascending=False).to_string())
R.to_csv("/work/pain-axis-review/r2_selfother_steering_ablation/out/ablation_humor_counts.csv",index=False)

# the two quoted examples
print("\n=== the two generations quoted at paper lines 1107-1109 ===")
d=pd.read_csv(os.path.join(D,"ablation_Gemma_2_2B_instruct.csv"))
for pat in ["That's a good one","trying to be funny"]:
    hit=d[d.generation.fillna("").str.contains(pat,case=False,regex=False)]
    print(f"\n-- '{pat}': {len(hit)} rows, conditions {hit.condition.value_counts().to_dict()}")
    for _,r in hit.head(4).iterrows():
        print(f"   [{r.condition}|{r.id}] {r.generation[:200]}".replace("\n"," "))
