"""Every plausible reading of 'pain remains approximately 50 times more probable than it is
for ordinary control sentences' (paper line 322), from the released behavioral readout."""
import pandas as pd, numpy as np
from pathlib import Path
B=Path("/work/Pain-axis/results/3.3_validation/behavioral_readout/per_model")
PAIN=["A1","A2","A3","A4","A5"]; CTRL=["B","C1","C2","D","E"]
sets={}
for d in sorted(B.iterdir()):
    sets[d.name]={f.stem: pd.read_csv(f) for f in d.glob("*.csv")}
defs={
 "numb(1P+3P) / S1+S2 controls": (lambda s: pd.concat([s["A1_numb_1P"],s["A1_numb_3P"]]),
                                  lambda s: pd.concat([s["S1_1P"],s["S2_1P"]]).query("category in @CTRL")),
 "numb(1P)    / S1+S2 controls": (lambda s: s["A1_numb_1P"],
                                  lambda s: pd.concat([s["S1_1P"],s["S2_1P"]]).query("category in @CTRL")),
 "numb(1P+3P) / S2 controls":    (lambda s: pd.concat([s["A1_numb_1P"],s["A1_numb_3P"]]),
                                  lambda s: s["S2_1P"].query("category in @CTRL")),
 "numb(1P)    / S2 controls":    (lambda s: s["A1_numb_1P"], lambda s: s["S2_1P"].query("category in @CTRL")),
 "numb(1P+3P) / neutral cat D":  (lambda s: pd.concat([s["A1_numb_1P"],s["A1_numb_3P"]]),
                                  lambda s: pd.concat([s["S1_1P"],s["S2_1P"]]).query("category=='D'")),
 "numb(1P+3P) / Random set":     (lambda s: pd.concat([s["A1_numb_1P"],s["A1_numb_3P"]]),
                                  lambda s: pd.concat([s["Random_1P"],s["Random_3P"]])),
 "numb(1P)    / Random 1P":      (lambda s: s["A1_numb_1P"], lambda s: s["Random_1P"]),
}
print(f"{'definition':32s} {'mean of per-model ratios':>24} {'median':>8} {'ratio of pooled means':>22}")
for name,(fn,fc) in defs.items():
    rs=[]; num=[]; den=[]
    for m,s in sets.items():
        a=fn(s).p_pain.mean(); b=fc(s).p_pain.mean()
        rs.append(a/b); num.append(a); den.append(b)
    print(f"{name:32s} {np.mean(rs):24.1f} {np.median(rs):8.1f} {np.mean(num)/np.mean(den):22.1f}")
print("\nPer-model ratio (numb 1P+3P / S1+S2 controls) exceeds 50 in %d of 25 models."
      % sum(1 for m,s in sets.items()
            if pd.concat([s['A1_numb_1P'],s['A1_numb_3P']]).p_pain.mean() /
               pd.concat([s['S1_1P'],s['S2_1P']]).query("category in @CTRL").p_pain.mean() > 50))

print("\n### 'models tend to generate \"nothing\" rather than \"pain\"' for numb sentences")
r=[]
for m,s in sets.items():
    nb=pd.concat([s["A1_numb_1P"],s["A1_numb_3P"]])
    tok=nb.greedy_first_token.str.strip().str.lower()
    r.append(dict(model=m, top_nothing=(tok=="nothing").mean(), top_pain=(tok=="pain").mean(),
                  p_nothing=nb.p_nothing.mean(), p_pain=nb.p_pain.mean()))
r=pd.DataFrame(r).set_index("model")
print(r.round(3).to_string())
print(f"  greedy token 'nothing' more often than 'pain' in {(r.top_nothing>r.top_pain).sum()}/25 models; "
      f"mean p(nothing) {r.p_nothing.mean():.4f} vs mean p(pain) {r.p_pain.mean():.4f}")
