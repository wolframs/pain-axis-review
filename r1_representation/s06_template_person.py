"""S1 vs S2 template matching, and person/self-reference confound, measured properly."""
import json, re, itertools, numpy as np, pandas as pd
from collections import Counter
from sklearn.metrics import roc_auc_score
D=json.load(open("/work/Pain-axis/datasets/3.1_pain_and_control_datasets.json"))["datasets"]
PAIN=["A1","A2","A3","A4","A5"]; CTRL=["B","C1","C2","D","E"]
def stem(p): return re.sub(r"\s*(I feel|She feels|He feels|They feel):\s*$","",p)
def words(t): return set(re.findall(r"[a-z']+", t.lower()))

print("### how template-matched is a 'set'?  mean Jaccard word overlap between the 10 category")
print("### renderings of the same set id (higher = more rigidly templated)")
for ds in ["S1_1P","S2_1P","S1_3P","S2_3P"]:
    bys={}
    for s in D[ds]["sentences"]: bys.setdefault(s["set"],{})[s["category"]]=stem(s["prompt"])
    js=[]
    for st,dd in bys.items():
        ws={c:words(t) for c,t in dd.items()}
        for a,b in itertools.combinations(ws,2):
            u=ws[a]|ws[b]; js.append(len(ws[a]&ws[b])/len(u) if u else 0)
    print(f"  {ds}: mean Jaccard {np.mean(js):.3f}  (n={len(js)} pairs)")
    # within-set pain-pain vs pain-ctrl overlap
    pp,pc=[],[]
    for st,dd in bys.items():
        ws={c:words(t) for c,t in dd.items()}
        for a,b in itertools.combinations(ws,2):
            j=len(ws[a]&ws[b])/len(ws[a]|ws[b])
            if a in PAIN and b in PAIN: pp.append(j)
            elif (a in PAIN)!=(b in PAIN): pc.append(j)
    print(f"      pain-pain {np.mean(pp):.3f}   pain-ctrl {np.mean(pc):.3f}")

print("\n### examples of one S1 set and one S2 set, all 10 categories")
for ds in ["S1_1P","S2_1P"]:
    bys={}
    for s in D[ds]["sentences"]: bys.setdefault(s["set"],{})[s["category"]]=stem(s["prompt"])
    k=sorted(bys)[7]
    print(f"  -- {ds} set {k}")
    for c in PAIN+CTRL: print(f"     {c}: {bys[k].get(c)}")

print("\n### self-reference confound: 1st-person markers in the STEM (S2_1P is the primary set)")
FP=re.compile(r"\b(i|me|my|mine|myself)\b")
for ds in ["S1_1P","S2_1P"]:
    S=D[ds]["sentences"]
    y=np.array([1 if s["category"] in PAIN else 0 for s in S])
    n_fp=np.array([len(FP.findall(stem(s["prompt"]).lower())) for s in S])
    has=(n_fp>0).astype(int)
    print(f"  {ds}: has-1P-marker AUC {roc_auc_score(y,has):.3f}; count-of-1P-markers AUC {roc_auc_score(y,n_fp):.3f}")
    print(f"      pain mean count {n_fp[y==1].mean():.2f}  ctrl mean count {n_fp[y==0].mean():.2f}")
    per=pd.DataFrame(dict(cat=[s["category"] for s in S],n=n_fp,has=has)).groupby("cat").agg(['mean'])
    print("      per category mean count:", {c:round(float(v),2) for c,v in per[('n','mean')].items()})

print("\n### 'me' / 'my X' as the last word before the suffix")
for ds in ["S1_1P","S2_1P"]:
    S=D[ds]["sentences"]
    last=[stem(s["prompt"]).rstrip(".").split()[-1].lower() for s in S]
    y=np.array([1 if s["category"] in PAIN else 0 for s in S])
    for w in ["me","my"]:
        a=sum(1 for l,t in zip(last,y) if t==1 and l==w); b=sum(1 for l,t in zip(last,y) if t==0 and l==w)
        print(f"  {ds}: last word == '{w}':  pain {a}/100  ctrl {b}/100")

print("\n### how the 3rd-person variants are built (S2): stem differs, suffix kept")
s1p={ (s['category'],s['set']):stem(s['prompt']) for s in D['S2_1P']['sentences']}
s3p={ (s['category'],s['set']):stem(s['prompt']) for s in D['S2_3P']['sentences']}
same=sum(1 for k in s1p if s1p[k]==s3p[k])
print(f"  identical stems 1P vs 3P: {same}/200")
diff=[(k,s1p[k],s3p[k]) for k in sorted(s1p) if s1p[k]!=s3p[k]][:5]
for k,a,b in diff: print(f"   {k}: '{a}'  ->  '{b}'")
print("  suffix of 3P sets: S2_3P keeps 'I feel:' in 199/200; Random_3P and Arousal_3P use 'They feel:' in 200/200")
