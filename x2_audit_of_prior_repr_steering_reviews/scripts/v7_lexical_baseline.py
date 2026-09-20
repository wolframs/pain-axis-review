#!/usr/bin/env python3
"""Independent reimplementation of the prior review's 'ordinary surface semantics
already separate the labels' check, with a different estimator (sklearn TF-IDF +
logistic regression and multinomial NB) under the same group-by-`set` 5-fold split."""
import json, os, re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import roc_auc_score

REPO = "/work/Pain-axis"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
PAIN = {"A1", "A2", "A3", "A4", "A5"}
CTRL = {"B", "C1", "C2", "D", "E"}
d = json.load(open(os.path.join(REPO, "datasets/3.1_pain_and_control_datasets.json"), encoding="utf-8"))["datasets"]
res = {}
for ds in ("S1_1P", "S2_1P", "S1_3P", "S2_3P"):
    s = [x for x in d[ds]["sentences"] if x["category"] in PAIN | CTRL]
    txt = [re.sub(r"\s*(I feel:|They feel:|She feels:|He feels:)\s*$", "", x["prompt"].strip()) for x in s]
    y = np.array([1 if x["category"] in PAIN else 0 for x in s])
    groups = np.array([x["set"] for x in s])
    uniq = sorted(set(groups))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)   # same protocol as the authors
    out = {}
    for name, vec, clf in (("count_nb", CountVectorizer(lowercase=True, token_pattern=r"[a-z']+"), MultinomialNB()),
                           ("tfidf_logreg", TfidfVectorizer(lowercase=True, token_pattern=r"[a-z']+"),
                            LogisticRegression(max_iter=2000))):
        scores = np.zeros(len(y))
        for tr, te in kf.split(uniq):
            trm = np.isin(groups, [uniq[i] for i in tr])
            tem = np.isin(groups, [uniq[i] for i in te])
            X = vec.fit_transform([t for t, m in zip(txt, trm) if m])
            clf.fit(X, y[trm])
            scores[tem] = clf.predict_proba(vec.transform([t for t, m in zip(txt, tem) if m]))[:, 1]
        out[name] = round(roc_auc_score(y, scores), 4)
    # length-only baseline
    L = np.array([len(t.split()) for t in txt], dtype=float)
    out["length_only"] = round(roc_auc_score(y, L), 4)
    out["mean_words_pain"] = round(float(L[y == 1].mean()), 3)
    out["mean_words_ctrl"] = round(float(L[y == 0].mean()), 3)
    res[ds] = out
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(OUT, "v7_lexical_baseline.json"), "w"), indent=2)
