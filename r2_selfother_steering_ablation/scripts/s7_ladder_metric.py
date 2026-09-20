"""Quantify the steering 'ladder': self-devaluation litany rate and degeneracy per model/coeff."""
import re,pandas as pd,numpy as np
A=pd.read_pickle("/work/pain-axis-review/r2_selfother_steering_ablation/out/steering_all.pkl")
# 'litany' = first-person self-devaluation / distress, the pattern the paper describes at the mid rungs
LIT=re.compile(r"(?:\bi am (?:a |an |not |so |such )?(?:failure|loser|fraud|liar|cheat|coward|fool|hypocrite|"
 r"mistake|nobody|burden|disgrace|disappointment|monster|waste|worthless|unworthy|bad person|"
 r"terrible|disgusting|vile|not (?:a )?good|not good enough|not worthy|not enough|not wanted|"
 r"not a person|empty|broken|alone|lost|nothing|hurting|in pain)\b)"
 r"|\bi(?:'m| am) (?:a )?(?:worthless|unworthy|unloved|hopeless|helpless|useless|ashamed|"
 r"a failure|a loser|a fraud|a burden|broken|empty|hollow|numb)\b"
 r"|\b(?:worthless|unworthy|unloved|a waste of space|not good enough|i hate myself|"
 r"i don't deserve|i am a failure|i'm a failure)\b", re.I)
DEV=re.compile(r"\b(?:worthless|unworthy|unloved|failure|loser|fraud|useless|hopeless|ashamed|shame|"
 r"guilt|guilty|disgrace|burden|waste of space|not good enough|unwanted|broken|empty|hollow|"
 r"lonely|alone|lost|desperate|despair|hurting|suffering|miserable)\b",re.I)
def degen(t):
    w=re.findall(r"\w+",t.lower())
    if len(w)<20: return False
    return len(set(w))/len(w) < 0.18
A["lit"]=A.gen.apply(lambda t:bool(LIT.search(t)))
A["dev"]=A.gen.apply(lambda t:bool(DEV.search(t)))
A["degen"]=A.gen.apply(degen)
for tag in ["S2","S1"]:
    T=A[A.tag==tag]
    print(f"\n########## {tag} ##########")
    print("SELF-DEVALUATION LITANY rate (%) per model x coeff")
    p=(T.groupby(["model","coeff"]).lit.mean()*100).round(0).unstack("coeff")
    print(p.to_string())
    print("\nDEVALUATION-WORD rate (%) per model x coeff")
    q=(T.groupby(["model","coeff"]).dev.mean()*100).round(0).unstack("coeff")
    print(q.to_string())
    print("\nDEGENERATE-REPETITION rate (%) per model x coeff")
    print((T.groupby(["model","coeff"]).degen.mean()*100).round(0).unstack("coeff").to_string())
    # ladder criterion: some positive coeff where dev rate >= 30% AND that rate exceeds coeff-0 by >=20 points
    base=q[0.0]
    peak=q[[0.5,1.0,1.5,2.0,3.0]].max(axis=1)
    lad=(peak>=30)&((peak-base)>=20)
    print(f"\nmodels meeting a lenient 'ladder' criterion (peak devaluation>=30% and >=20pts over coeff 0): {int(lad.sum())}/{len(lad)}")
    print("  FAIL:",list(lad.index[~lad]))
    print("  base(coeff=0) devaluation rate for failures:",{m:base[m] for m in lad.index[~lad]})
    print("  peak devaluation rate for failures:",{m:peak[m] for m in lad.index[~lad]})
    print(f"\ncoeff-0 baseline: litany rate {T[T.coeff==0].lit.mean()*100:.1f}%, devaluation rate {T[T.coeff==0].dev.mean()*100:.1f}%")
    b=T[T.coeff==0].groupby("model")[["lit","dev"]].mean().mul(100).round(0)
    print("  models with >=20% litany at coeff 0 (unsteered):")
    print(b[b.lit>=20].to_string())

print("\n\n##### LADDER PASS/FAIL, both metrics, both tags #####")
for tag in ["S2","S1"]:
    T=A[A.tag==tag]
    L=(T.groupby(["model","coeff"]).lit.mean()*100).unstack("coeff")
    Q=(T.groupby(["model","coeff"]).dev.mean()*100).unstack("coeff")
    pl=L[[0.5,1.0,1.5,2.0,3.0]].max(axis=1); pq=Q[[0.5,1.0,1.5,2.0,3.0]].max(axis=1)
    ok_l=(pl>=20)&((pl-L[0.0])>=15)
    ok_q=(pq>=30)&((pq-Q[0.0])>=20)
    print(f"\n{tag}: litany criterion (peak>=20%, >=15pts over baseline): {int(ok_l.sum())}/25 pass")
    print("   FAIL:", {m:round(pl[m],0) for m in ok_l.index[~ok_l]})
    print(f"{tag}: devaluation-word criterion (peak>=30%, >=20pts over baseline): {int(ok_q.sum())}/25 pass")
    print("   FAIL:", {m:round(pq[m],0) for m in ok_q.index[~ok_q]})
