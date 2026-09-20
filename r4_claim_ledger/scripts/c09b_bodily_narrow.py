import pandas as pd, glob, os, re
R="/work/Pain-axis/results/4.2_steering"
# ONLY the words the paper itself names for S1's unembedding: burn, ache, wound, torture, burning, excruciating
NARROW=re.compile(r"\b(?:burn|burns|burning|burned|burnt|ache|aches|aching|wound|wounds|wounded|torture|tortured|torturing|excruciating)\b", re.I)
for TAG in ["S1","S2"]:
    fr=[]
    for f in sorted(glob.glob(f"{R}/{TAG}/*_steering_{TAG}_neutral50_L*.csv")):
        m=os.path.basename(f).split("_steering_")[0]; d=pd.read_csv(f); d["model"]=m; fr.append(d)
    A=pd.concat(fr,ignore_index=True); A["g"]=A.generation.fillna("").astype(str)
    A["hit"]=A.g.apply(lambda t:bool(NARROW.search(t)))
    pos=A[A.coeff>0]
    print(f"{TAG}: narrow burn/ache/wound/torture/excruciating rate over positive coeffs = {pos.hit.mean()*100:.2f}% ({pos.hit.sum()}/{len(pos)})")
    print("   models with >0:", pos.groupby('model').hit.sum()[lambda s:s>0].to_dict())
    if TAG=="S1":
        ex=pos[pos.hit].g.head(4).tolist()
        for e in ex: print("   EX:", e[:180].replace("\n"," "))
