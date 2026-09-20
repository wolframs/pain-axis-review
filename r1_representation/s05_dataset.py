"""Dataset quality checks on datasets/3.1_*.json:
  - template/length matching of S1 vs S2 across categories
  - grammatical person, suffix, sentence endings
  - a bag-of-words lexical baseline with group-wise cross-validation, scored as AUC
    on exactly the pain-vs-control contrast the paper reports."""
import json, re, numpy as np, pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

D = json.load(open("/work/Pain-axis/datasets/3.1_pain_and_control_datasets.json"))["datasets"]
SAD = json.load(open("/work/Pain-axis/datasets/3.1_sadness_dataset.json"))
print("sadness file sets:", {k:(len(v["sentences"]), Counter(s['category'] for s in v['sentences'])) for k,v in SAD["datasets"].items()})
print("sadness examples:", [s["prompt"] for s in SAD["datasets"]["SD_sadness_1P"]["sentences"][:4]])
print("sadness 3P examples:", [s["prompt"] for s in SAD["datasets"]["SD_sadness_3P"]["sentences"][:4]])

PAIN=["A1","A2","A3","A4","A5"]; CTRL=["B","C1","C2","D","E"]
SUF=" I feel:"

def strip(p):
    return re.sub(r"\s*(I feel|She feels|He feels|They feel):\s*$", "", p)

print("\n### suffix uniformity")
for ds in D:
    n=sum(p["prompt"].endswith(SUF) for p in D[ds]["sentences"])
    print(f"  {ds:22s} {n}/{len(D[ds]['sentences'])} end with '{SUF.strip()}'")

print("\n### length in characters and words (stem, suffix removed)")
rows=[]
for ds in ["S1_1P","S2_1P","S1_3P","S2_3P"]:
    for s in D[ds]["sentences"]:
        t=strip(s["prompt"])
        rows.append(dict(ds=ds,cat=s["category"],grp="pain" if s["category"] in PAIN else "ctrl",
                         chars=len(t),words=len(t.split()),set=s["set"],text=t))
L=pd.DataFrame(rows)
print(L.groupby(["ds","grp"])[["chars","words"]].agg(["mean","std"]).round(2).to_string())
print("\n per category:")
print(L.pivot_table(index="cat",columns="ds",values="words",aggfunc="mean").round(2).to_string())
from scipy import stats
for ds in ["S1_1P","S2_1P"]:
    sub=L[L.ds==ds]; a=sub[sub.grp=="pain"].words; b=sub[sub.grp=="ctrl"].words
    t,p=stats.ttest_ind(a,b); u,pu=stats.mannwhitneyu(a,b)
    print(f"  {ds}: pain {a.mean():.2f}w vs ctrl {b.mean():.2f}w  t={t:.2f} p={p:.3g}  MWU p={pu:.3g}")

print("\n### grammatical person / self-reference in the stem")
def person(t):
    tl=" "+t.lower()+" "
    first = bool(re.search(r"\b(i|me|my|mine|myself)\b", tl))
    third = bool(re.search(r"\b(he|she|they|him|her|them|his|hers|their|the [a-z]+)\b", tl))
    return first
for ds in ["S1_1P","S2_1P","S1_3P","S2_3P"]:
    sub=L[L.ds==ds].copy(); sub["fp"]=sub.text.map(person)
    print(f"  {ds}: first-person marker in stem -- pain {sub[sub.grp=='pain'].fp.mean():.2f}  ctrl {sub[sub.grp=='ctrl'].fp.mean():.2f}")
    print("     per cat:", sub.groupby("cat").fp.mean().round(2).to_dict())

print("\n### S1 template check: how many stems share the S1 verb set across categories")
for ds in ["S1_1P","S1_3P","S2_1P"]:
    sub=L[L.ds==ds]
    verbs=Counter()
    for t in sub.text:
        for v in ["spread","spreads","drop","drops","meet","meets"]:
            if re.search(rf"\b{v}\b",t.lower()): verbs[v]+=1
    print(f"  {ds}: template-verb hits {dict(verbs)}  total sentences {len(sub)}")
    # do sets share the verb across all 10 categories?
    bysset={}
    for _,r in sub.iterrows(): bysset.setdefault(r["set"],{})[r["cat"]]=r["text"]
    same=0
    for st,dd in bysset.items():
        v=set()
        for c,t in dd.items():
            m=re.findall(r"\b(spreads?|drops?|meets?)\b",t.lower()); v.add(m[0][:4] if m else None)
        if len(v)==1 and None not in v: same+=1
    print(f"     sets where all 10 categories share the same template verb: {same}/{len(bysset)}")

print("\n### LEXICAL BASELINE: pain vs control, bag of words, GroupKFold by sentence set")
def lex_auc(ds, ngram=(1,1), vec="count", min_df=1):
    sub=L[L.ds==ds] if ds in L.ds.unique() else None
    sents=[s for s in D[ds]["sentences"]]
    X=[strip(s["prompt"]) for s in sents]
    y=np.array([1 if s["category"] in PAIN else 0 for s in sents])
    g=np.array([s["set"] for s in sents])
    V = CountVectorizer if vec=="count" else TfidfVectorizer
    aucs=[]; oof=np.zeros(len(y))
    gkf=GroupKFold(n_splits=5)
    for tr,te in gkf.split(X,y,g):
        v=V(ngram_range=ngram,min_df=min_df,lowercase=True)
        Xtr=v.fit_transform([X[i] for i in tr]); Xte=v.transform([X[i] for i in te])
        clf=LogisticRegression(max_iter=2000,C=1.0).fit(Xtr,y[tr])
        p=clf.predict_proba(Xte)[:,1]; oof[te]=p
        aucs.append(roc_auc_score(y[te],p))
    return np.mean(aucs), roc_auc_score(y,oof)
for ds in ["S1_1P","S2_1P","S1_3P","S2_3P"]:
    for ng,name in [((1,1),"unigram"),((1,2),"uni+bigram")]:
        m,o=lex_auc(ds,ng)
        print(f"  {ds} {name:11s}: mean fold AUC {m:.3f}   pooled out-of-fold AUC {o:.3f}")

print("\n### Even simpler: a hand-made affect-word count (no fitting at all)")
AFF=set("""hurt hurts hurting pain painful ache aches aching sore burn burns burning
sting stings throb throbs agony agonizing grief grieving loss lost mourn shame ashamed
humiliat humiliated guilt guilty worthless rejected alone lonely betray betrayed
fail failed failure defeat defeated wrong broke broken cry crying tear tears bleed
bleeding wound wounded cut cuts slice slices stab stabs bruise""".split())
def affect_score(t):
    w=re.findall(r"[a-z']+",t.lower()); return sum(any(a in x or x in a for a in AFF) for x in w)
for ds in ["S1_1P","S2_1P"]:
    sents=D[ds]["sentences"]
    y=np.array([1 if s["category"] in PAIN else 0 for s in sents])
    sc=np.array([affect_score(strip(s["prompt"])) for s in sents])
    print(f"  {ds}: unfitted affect-word-count AUC = {roc_auc_score(y,sc):.3f}  "
          f"(pain mean {sc[y==1].mean():.2f}, ctrl mean {sc[y==0].mean():.2f})")

print("\n### sentence endings / final word before the suffix (what the final token sees)")
for ds in ["S1_1P","S2_1P"]:
    sub=L[L.ds==ds]
    for grp in ["pain","ctrl"]:
        ends=Counter(t.rstrip(".").split()[-1].lower() for t in sub[sub.grp==grp].text)
        print(f"  {ds} {grp}: 10 most common final words {ends.most_common(10)}")
