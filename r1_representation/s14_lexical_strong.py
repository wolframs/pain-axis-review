"""Stronger lexical baselines, to make the 'a bag of words cannot match the pain vector'
conclusion robust. Group 5-fold CV by sentence set, pain vs the 5 control categories."""
import json, re, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
D=json.load(open("/work/Pain-axis/datasets/3.1_pain_and_control_datasets.json"))["datasets"]
PAIN=["A1","A2","A3","A4","A5"]
def stem(p): return re.sub(r"\s*(I feel|She feels|He feels|They feel):\s*$","",p)
CFG=[("word 1-2gram tfidf, logreg C=1", TfidfVectorizer(ngram_range=(1,2)), lambda: LogisticRegression(max_iter=4000,C=1)),
     ("word 1-2gram tfidf, logreg C=10", TfidfVectorizer(ngram_range=(1,2)), lambda: LogisticRegression(max_iter=4000,C=10)),
     ("char 3-5gram tfidf, logreg C=10", TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5)), lambda: LogisticRegression(max_iter=4000,C=10)),
     ("char 3-5gram tfidf, linSVC",      TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5)), lambda: LinearSVC(C=1)),
     ("word counts, ridge",              CountVectorizer(ngram_range=(1,1)), lambda: RidgeClassifier())]
for ds in ["S1_1P","S2_1P"]:
    S=D[ds]["sentences"]; X=[stem(s["prompt"]) for s in S]
    y=np.array([1 if s["category"] in PAIN else 0 for s in S]); g=np.array([s["set"] for s in S])
    print(f"-- {ds}")
    for name,vec,mk in CFG:
        oof=np.zeros(len(y))
        for tr,te in GroupKFold(5).split(X,y,g):
            from copy import deepcopy
            v=deepcopy(vec); Xt=v.fit_transform([X[i] for i in tr]); Xe=v.transform([X[i] for i in te])
            c=mk().fit(Xt,y[tr])
            oof[te]=c.decision_function(Xe) if hasattr(c,"decision_function") else c.predict_proba(Xe)[:,1]
        print(f"   {name:34s} out-of-fold AUC {roc_auc_score(y,oof):.3f}")
print("\nFor comparison, the pain vector on the same contrast: in-sample 0.93-1.00,")
print("5-fold held-out (same set-wise split) 0.91-1.00 for S2 and 0.85-0.94 for S1.")
