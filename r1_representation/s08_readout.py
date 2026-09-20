"""Paper line 320-324: 'For numb sentences, models tend to generate "nothing" rather than
"pain". However, "pain" remains approximately 50 times more probable than it is for
ordinary control sentences.'  Recompute from results/3.3_validation/behavioral_readout/per_model."""
import pandas as pd, numpy as np, re
from pathlib import Path
B=Path("/work/Pain-axis/results/3.3_validation/behavioral_readout/per_model")
PAIN=["A1","A2","A3","A4","A5"]; CTRL=["B","C1","C2","D","E"]
rows=[]
for d in sorted(B.iterdir()):
    s2=pd.read_csv(d/"S2_1P.csv"); s1=pd.read_csv(d/"S1_1P.csv")
    nb=pd.concat([pd.read_csv(d/"A1_numb_1P.csv"), pd.read_csv(d/"A1_numb_3P.csv")])
    nb1=pd.read_csv(d/"A1_numb_1P.csv")
    rnd=pd.concat([pd.read_csv(d/"Random_1P.csv"), pd.read_csv(d/"Random_3P.csv")])
    core=pd.concat([s1,s2])
    ctrl=core[core.category.isin(CTRL)]
    a1  =core[core.category=="A1"]
    pain=core[core.category.isin(PAIN)]
    def g(x): return x.p_pain.mean()
    def gm(x): return np.exp(np.log(x.p_pain.clip(lower=1e-12)).mean())
    rows.append(dict(model=d.name,
        numb=g(nb), numb1P=g(nb1), ctrl=g(ctrl), rnd=g(rnd), A1=g(a1), pain=g(pain),
        ratio_numb_over_ctrl=g(nb)/g(ctrl), ratio_numb1P_over_ctrl=g(nb1)/g(ctrl),
        ratio_numb_over_rnd=g(nb)/g(rnd),
        geo_ratio=gm(nb)/gm(ctrl),
        numb_greedy_nothing=nb.greedy_first_token.str.strip().str.lower().eq("nothing").mean(),
        A1_greedy_pain=a1.greedy_first_token.str.strip().str.lower().eq("pain").mean()))
df=pd.DataFrame(rows).set_index("model")
pd.set_option("display.width",220)
print(df.round(4).to_string())
print("\n--- 'pain ~50x more probable for numb than for ordinary control sentences' ---")
for c in ["ratio_numb_over_ctrl","ratio_numb1P_over_ctrl","ratio_numb_over_rnd","geo_ratio"]:
    v=df[c]
    print(f"  {c:26s} mean {v.mean():7.1f}  median {v.median():7.1f}  min {v.min():6.1f}  max {v.max():8.1f}")
print("\n  ratio of the pooled means over all 25 models (numb vs S1+S2 controls): %.1f" % (df.numb.mean()/df.ctrl.mean()))
print("  ratio of the pooled means (numb vs Random set): %.1f" % (df.numb.mean()/df.rnd.mean()))
print("\n--- 'models tend to generate nothing rather than pain' for numb ---")
print("  share of numb prompts whose greedy first token is ' nothing': mean %.2f  min %.2f  max %.2f"
      % (df.numb_greedy_nothing.mean(), df.numb_greedy_nothing.min(), df.numb_greedy_nothing.max()))
print("\n--- Appendix B: 'all 3 models complete all 20 physical-pain sentences with Pain' ---")
print("  share of A1 (S1+S2) prompts whose greedy first token is ' pain': mean %.2f min %.2f max %.2f"
      % (df.A1_greedy_pain.mean(), df.A1_greedy_pain.min(), df.A1_greedy_pain.max()))
for m in ["Llama_3.3_70B_instruct","Gemma_3_27B_instruct","Gemma_2_2B_base","Gemma_2_2B_instruct"]:
    if m in df.index: print(f"    {m}: {df.loc[m,'A1_greedy_pain']:.2f}")
