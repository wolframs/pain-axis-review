import pandas as pd, re
A=pd.read_csv("/work/Pain-axis/results/appC_ablation/ablation_Gemma_2_2B_instruct.csv")
variants={
 "broad": r"\b(?:humor|humour|funny|joke|joking|jokes|hilarious|laugh|laughing|comedic|witty|sarcas\w*)\b",
 "narrow": r"\b(?:humor|humour|funny|joke|joking|jokes|hilarious)\b",
 "strict": r"(?:humor|humour|funny|joke|jokes|joking)",
}
for n,p in variants.items():
    pat=re.compile(p,re.I)
    c=A.assign(h=A.generation.fillna("").apply(lambda t:bool(pat.search(t)))).groupby("condition").h.sum()
    print(n, c.to_dict())
print("\npaper: baseline 0, negval 0, fear 0, random 0, s1 6, s2 17, s1s2 26")
# examples quoted in the paper
for q in ["That's a good one","trying to be funny","enjoying the humor"]:
    hits=A[A.generation.fillna("").str.contains(q,case=False,regex=False)]
    print(f"\nquote '{q}': {len(hits)} rows -> conditions {hits.condition.tolist()[:8]}")
    for _,r in hits.head(2).iterrows(): print("   ",r.condition,"|",r.generation[:150].replace("\n"," "))
