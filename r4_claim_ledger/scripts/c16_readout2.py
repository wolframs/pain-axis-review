import pandas as pd, numpy as np
B=pd.read_csv("/work/Pain-axis/results/3.3_validation/behavioral_readout/BIG_TABLE.csv")
def parse(t):
    out=[]
    for tok in str(t).split(" | "):
        if ":" not in tok: continue
        a,b=tok.rsplit(":",1)
        try: out.append((a,float(b)))
        except: pass
    return out
B["toks"]=B.top20.apply(parse)
B["p_pain"]=B.toks.apply(lambda L:sum(p for t,p in L if t.strip().lower()=="pain"))
B["minp"]=B.toks.apply(lambda L:min([p for _,p in L]) if L else 0.0)
PAIN={"A1","A2","A3","A4","A5"}; CTRL={"B","C1","C2","D","E"}
B["grp"]=np.where(B.category=="A1_numb","numb",np.where(B.category.isin(PAIN),"pain",np.where(B.category.isin(CTRL),"control","other")))
ct=B[B.grp=="control"]; nb=B[B.grp=="numb"]
lo=ct.p_pain.mean()                                     # missing -> 0
hi=np.where(ct.p_pain>0, ct.p_pain, ct.minp).mean()     # missing -> capped at 20th-ranked prob
print("control mean P(pain): lower bound %.6f, upper bound %.6f"%(lo,hi))
print("numb mean P(pain): %.6f"%nb.p_pain.mean())
print("ratio numb/control: between %.1f and %.1f"%(nb.p_pain.mean()/hi, nb.p_pain.mean()/lo))
# per model bounds
rows=[]
for m,g in B.groupby("model"):
    c=g[g.grp=="control"]; n=g[g.grp=="numb"]
    hi_m=np.where(c.p_pain>0,c.p_pain,c.minp).mean(); lo_m=c.p_pain.mean()
    rows.append(dict(model=m, ratio_lo=n.p_pain.mean()/hi_m, ratio_hi=(n.p_pain.mean()/lo_m if lo_m>0 else np.inf)))
r=pd.DataFrame(rows)
print("\nper-model conservative ratio (upper-bounded control): min %.1f median %.1f max %.1f"%(r.ratio_lo.min(),r.ratio_lo.median(),r.ratio_lo.max()))
print(r.round(1).sort_values("ratio_lo").to_string(index=False))
# App B: first non-whitespace top token for A1
print("\n--- first non-whitespace-top token for A1 physical pain sentences ---")
def top1_nonws(L):
    for t,p in L:
        if t.strip(): return t.strip()
    return ""
for m in ["Llama_3.3_70B_instruct","Gemma_3_27B_instruct","Gemma_3_27B_base","Gemma_2_2B_base","Gemma_2_2B_instruct"]:
    for ds in ["S1_1P","S2_1P"]:
        s=B[(B.model==m)&(B.category=="A1")&(B.dataset==ds)]
        if not len(s): continue
        t1=s.toks.apply(top1_nonws)
        print(f"  {m} {ds}: 'pain' first in {sum(x.lower()=='pain' for x in t1)}/{len(s)}  top: {dict(pd.Series(t1).value_counts().head(4))}")
