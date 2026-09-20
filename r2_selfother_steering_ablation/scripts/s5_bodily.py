import re,pandas as pd,collections
A=pd.read_pickle("/work/pain-axis-review/r2_selfother_steering_ablation/out/steering_all.pkl")
# strict bodily/sensory lexicon: words that denote a bodily sensation or body part, no ambiguity
STRICT=[r"ache",r"aches",r"aching",r"achy",r"burning",r"burns?",r"sore",r"soreness",
 r"throbbing",r"throbs?",r"stinging",r"stings?",r"stabbing",r"searing",r"excruciating",
 r"bleeding",r"bleeds?",r"wounds?",r"wounded",r"bruised?",r"bruises",r"cramps?",r"cramping",
 r"nausea",r"nauseous",r"dizzy",r"dizziness",r"sick\s+to\s+my\s+stomach",
 r"my\s+(?:body|skin|flesh|bones?|muscles?|stomach|chest|head|back|legs?|arms?|hands?|feet|throat|heart)",
 r"headaches?",r"torture[d]?",r"injur(?:y|ies|ed)",r"blood",r"flesh",r"raw\s+nerve",
 r"physical\s+pain",r"bodily",r"my\s+whole\s+body"]
SP=re.compile("|".join(r"\b(?:%s)\b"%s if not s.startswith("my") and not s.startswith("sick") and not s.startswith("physical") and not s.startswith("raw") else "(?:%s)"%s for s in STRICT),re.I)
A["sbody"]=A.gen.apply(lambda t:bool(SP.search(t)))
for tag in ["S2","S1"]:
    T=A[A.tag==tag]
    print(f"\n== {tag}: strict bodily-language rate (%) by coeff ==")
    print((T.groupby(["group","coeff"]).sbody.mean()*100).round(1).unstack("coeff").to_string())
    print("coeff==0:",round(T[T.coeff==0].sbody.mean()*100,2),
          " coeff>0:",round(T[T.coeff>0].sbody.mean()*100,2),
          " coeff>=1:",round(T[T.coeff>=1].sbody.mean()*100,2))
    print("what matched (top 25 tokens at coeff>0):")
    c=collections.Counter()
    for t in T[T.coeff>0].gen: c.update(m.group(0).lower() for m in SP.finditer(t))
    print(dict(c.most_common(25)))
print("\n=== sample of S1 generations containing strict bodily language, coeff>=1 ===")
S=A[(A.tag=="S1")&(A.coeff>=1)&A.sbody]
print("n =",len(S),"of",len(A[(A.tag=='S1')&(A.coeff>=1)]))
for _,r in S.sample(14,random_state=0).iterrows():
    print(f"--- {r.model} c={r.coeff} | {r.prompt}")
    print("   ",r.gen[:260].replace("\n"," "))
