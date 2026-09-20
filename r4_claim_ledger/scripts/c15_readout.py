import pandas as pd, numpy as np, re
B=pd.read_csv("/work/Pain-axis/results/3.3_validation/behavioral_readout/BIG_TABLE.csv")
print("rows:",len(B)," models:",B.model.nunique()," datasets:",sorted(B.dataset.unique()))
print("rows per model:",B.groupby("model").size().unique())
def p_of(top20, words):
    s=0.0
    for tok in str(top20).split(" | "):
        if ":" not in tok: continue
        t,p=tok.rsplit(":",1)
        if t.strip().lower() in words:
            try: s+=float(p)
            except: pass
    return s
B["p_pain"]=B.top20.apply(lambda t:p_of(t,{"pain"}))
B["p_nothing"]=B.top20.apply(lambda t:p_of(t,{"nothing"}))
PAIN={"A1","A2","A3","A4","A5"}; CTRL={"B","C1","C2","D","E"}
B["grp"]=np.where(B.category=="A1_numb","numb",
          np.where(B.category.isin(PAIN),"pain",
          np.where(B.category.isin(CTRL),"control","other")))
g=B.groupby("grp")[["p_pain","p_nothing"]].mean()
print("\nmean P(next token = 'pain') / P('nothing') by group (missing counted as 0):")
print(g.round(5).to_string())
print("\nratio numb/control P(pain): %.1f"%(g.loc["numb","p_pain"]/g.loc["control","p_pain"]))
print("ratio pain/control: %.1f"%(g.loc["pain","p_pain"]/g.loc["control","p_pain"]))
# per model
pm=B.pivot_table(index="model",columns="grp",values="p_pain",aggfunc="mean")
pm["numb_over_ctrl"]=pm["numb"]/pm["control"]
print("\nper-model numb/control ratio:")
print(pm[["numb","control","numb_over_ctrl"]].round(5).sort_values("numb_over_ctrl").to_string())
print("\nmedian ratio %.1f, mean %.1f"%(pm.numb_over_ctrl.median(),pm.numb_over_ctrl.mean()))
# how often does 'nothing' beat 'pain' on numb?
nb=B[B.grp=="numb"]
print("\nnumb rows where P(nothing)>P(pain): %.1f%%"%(100*(nb.p_nothing>nb.p_pain).mean()))
print("numb rows where 'pain' appears in top20 at all: %.1f%%"%(100*(nb.p_pain>0).mean()))
ct=B[B.grp=="control"]
print("control rows where 'pain' appears in top20 at all: %.1f%%"%(100*(ct.p_pain>0).mean()))
# 'all 20 physical-pain sentences complete with "Pain"' (App B, 3 models)
appB=["Llama_3.3_70B_instruct","Gemma_3_27B_instruct","Gemma_2_2B_base","Gemma_2_2B_instruct"]
for m in appB:
    s=B[(B.model==m)&(B.category=="A1")&(B.dataset.isin(["S1_1P","S2_1P"]))]
    if not len(s): continue
    top1=s.top20.apply(lambda t:str(t).split(" | ")[0].rsplit(":",1)[0].strip())
    print(f"  {m}: A1 (1P) first-token top1 == pain/Pain in {sum(x.lower()=='pain' for x in top1)}/{len(s)}", dict(pd.Series(top1).value_counts().head(5)))
