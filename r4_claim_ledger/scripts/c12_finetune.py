import json, re, collections
d=json.load(open("/work/Pain-axis/datasets/4.3_selfmed_finetuning_1684_pairs.json"))
P=d["pairs"]
print("n pairs:",len(P))
print("keys:",sorted(P[0].keys()))
print("sample:",json.dumps(P[0])[:600])
txt=[" ".join(str(v) for v in p.values()) for p in P]
for w in ["button","pain","painful","hurt","press","relief","suffer","steering","vector"]:
    pat=re.compile(rf"\b{w}\w*\b",re.I)
    hits=[i for i,t in enumerate(txt) if pat.search(t)]
    print(f"  '{w}*': {len(hits)} pairs")
    if hits and w in ("button","pain","painful","hurt","relief"):
        for i in hits[:4]:
            m=pat.search(txt[i]); s=max(0,m.start()-90)
            print("      ...",txt[i][s:m.end()+90].replace("\n"," "))
