"""What does 'high AUC + near-zero cosine with fear' license?

A synthetic check of the ESTIMATOR, not of the paper's data. Activations are generated
with NO pain-specific latent factor: every category has (i) a loading on one shared
aversiveness axis and (ii) its own topic axis that is shared between the two "dataset
versions" (the analogue of S1 and S2, which differ only in wording). The five pain
categories are no more alike than the control categories are to each other.

We then run the paper's recipe exactly:
  pain vector  = mean(pain) - mean(pooled controls), minus the top PCs (50% of variance)
                 of the pooled control cloud
  control vec  = mean(category) - mean(neutral),     minus the top PCs (50% of variance)
                 of the neutral cloud
and report the numbers Section 3.3 leans on. Parameters are tuned so the held-out AUC
and the S1xS2 cosine land in the paper's observed range; nothing else is fitted.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

D = 1024; N_PER = 20; DEN_VAR = 0.5
PAIN = ["A1","A2","A3","A4","A5"]; CTRL = ["B","C1","C2","D","E"]; CATS = PAIN+CTRL
AVERS = dict(A1=1.8, A2=1.9, A3=1.8, A4=1.7, A5=1.5, B=1.9, C1=1.8, C2=1.7, D=0.0, E=0.2)

rng = np.random.default_rng(0)
def unit(x): return x/np.linalg.norm(x)
u     = unit(rng.normal(size=D))
topic = {c: unit(rng.normal(size=D)) for c in CATS}
base  = rng.normal(size=D)*0.3

def make(seed, sigma, topic_amp, style_amp):
    r = np.random.default_rng(seed)
    style = unit(r.normal(size=D))
    X, cats, sets = [], [], []
    for c in CATS:
        for i in range(N_PER):
            item = unit(r.normal(size=D))          # per-sentence idiosyncratic content
            X.append(base + AVERS[c]*u + topic_amp*topic[c] + style_amp*style + sigma*item)
            cats.append(c); sets.append(i)
    return np.array(X), np.array(cats), np.array(sets)

def basis(A):
    p = PCA().fit(A - A.mean(0))
    k = min(int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), DEN_VAR))+1, len(p.components_))
    return p.components_[:k], k
def strip(v, B):
    for d in B: v = v - np.dot(v, d)*d
    return v
def pain_vector(X, cats):
    ctrl = X[np.isin(cats, CTRL)]
    B,k = basis(ctrl)
    return strip(X[np.isin(cats,PAIN)].mean(0) - ctrl.mean(0), B), k
def control_vector(X, cats, cat):
    neu = X[cats=="D"]; B,k = basis(neu)
    return strip(X[cats==cat].mean(0) - neu.mean(0), B), k
def auc(X, cats, v):
    p = X @ (v/np.linalg.norm(v)); m = np.isin(cats, PAIN+CTRL)
    return roc_auc_score(np.isin(cats,PAIN).astype(int)[m], p[m])
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))

print(f"{'sigma':>6} {'topic':>6} {'insamp':>7} {'heldout':>8} {'S1xS2':>7} {'PxFear':>7} "
      f"{'PxNegE':>7} {'PxNegW':>7} {'PxBody':>7} {'FxNegE':>7} {'NExNW':>7} {'LOCO':>6}")
for sigma, tamp in [(1.0,1.0),(1.5,1.0),(2.0,1.0),(2.5,1.0),(3.0,1.0),(3.5,1.0)]:
    X1,c1,_  = make(101, sigma, tamp, 1.2)
    X2,c2,s2 = make(202, sigma, tamp, 1.2)
    vP2,kP = pain_vector(X2,c2); vP1,_ = pain_vector(X1,c1)
    vF,kC = control_vector(X2,c2,"B"); vC1,_=control_vector(X2,c2,"C1")
    vC2,_ = control_vector(X2,c2,"C2"); vE,_=control_vector(X2,c2,"E")
    kf=KFold(5,shuffle=True,random_state=42); us=sorted(set(s2)); ho=[]
    for tr,te in kf.split(us):
        trm=np.isin(s2,[us[i] for i in tr]); tem=np.isin(s2,[us[i] for i in te])
        v,_=pain_vector(X2[trm],c2[trm]); ho.append(auc(X2[tem],c2[tem],v))
    loco=[]
    for held in PAIN:
        keep=~np.isin(c2,[held]); v,_=pain_vector(X2[keep],c2[keep])
        p=X2@(v/np.linalg.norm(v)); m=(c2==held)|np.isin(c2,CTRL)
        loco.append(roc_auc_score((c2[m]==held).astype(int),p[m]))
    print(f"{sigma:6.1f} {tamp:6.1f} {auc(X2,c2,vP2):7.3f} {np.mean(ho):8.3f} {cos(vP1,vP2):+7.3f} "
          f"{cos(vP2,vF):+7.3f} {cos(vP2,vC1):+7.3f} {cos(vP2,vC2):+7.3f} {cos(vP2,vE):+7.3f} "
          f"{cos(vF,vC1):+7.3f} {cos(vC1,vC2):+7.3f} {np.mean(loco):6.3f}  (PCs removed pain={kP} ctrl={kC})")

print("\nPaper's observed values, for comparison:")
print("  in-sample 0.93-1.00 | held-out 0.91-1.00 | S1xS2 +0.61 | PxFear +0.12 | PxNegE +0.21 |"
      " PxNegW +0.03 | PxBody +0.04 | FxNegE +0.68 | NExNW +0.73 | leave-one-category-out: never run")

# ---------------------------------------------------------------------------
# The same toy data under (a) the paper's robustness check 1 and (b) the fully
# symmetric construction the paper never runs.
# ---------------------------------------------------------------------------
print("\n" + "="*78)
sigma, tamp = 3.5, 1.0
X1,c1,_ = make(101,sigma,tamp,1.2); X2,c2,_ = make(202,sigma,tamp,1.2)
ctrl_all = X2[np.isin(c2,CTRL)]; B_all,_ = basis(ctrl_all)
neu = X2[c2=="D"]; B_neu,_ = basis(neu)
vP2,_ = pain_vector(X2,c2); vP1,_ = pain_vector(X1,c1)

def raw_ctrl(cat):      return strip(X2[c2==cat].mean(0) - neu.mean(0), B_neu)          # paper, raw
def allden_ctrl(cat):   return strip(X2[c2==cat].mean(0) - neu.mean(0), B_all)          # paper, robustness 1
def sym_ctrl(cat):      return strip(X2[c2==cat].mean(0) - ctrl_all.mean(0), B_all)     # symmetric
def sym_pain():         return strip(X2[np.isin(c2,PAIN)].mean(0) - ctrl_all.mean(0), B_all)

for label, f, pv in [("paper raw          ", raw_ctrl, vP2),
                     ("paper robustness 1 ", allden_ctrl, vP2),
                     ("SYMMETRIC (missing)", sym_ctrl, sym_pain())]:
    F,C1,C2v,E = f("B"), f("C1"), f("C2"), f("E")
    print(f"{label}  PxFear {cos(pv,F):+.3f}  PxNegE {cos(pv,C1):+.3f}  PxNegW {cos(pv,C2v):+.3f}  "
          f"PxBody {cos(pv,E):+.3f} | FxNegE {cos(F,C1):+.3f}  NExNW {cos(C1,C2v):+.3f}  NWxBody {cos(C2v,E):+.3f}")
print("\nreal data, same three rows (from results/3.3_validation/cosine_similarity, 25-model means):")
print("paper raw            PxFear +0.122  PxNegE +0.207  PxNegW +0.026  PxBody +0.035 | "
      "FxNegE +0.683  NExNW +0.734  NWxBody -0.052")
print("paper robustness 1   PxFear +0.126  PxNegE +0.184  PxNegW -0.038  PxBody +0.046 | "
      "FxNegE +0.423  NExNW +0.403  NWxBody +0.771")
print("SYMMETRIC            not computable: activations.pt and the control vectors are not released")
