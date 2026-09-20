import pandas as pd, numpy as np, glob, os, re
R="/work/Pain-axis/results/4.2_steering"
PAT=re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.I)
INSTRUCT={"Phi_4"}
for TAG in ["S2","S1"]:
    fs=sorted(glob.glob(f"{R}/{TAG}/*_steering_{TAG}_neutral50_L*.csv"))
    print(f"=== {TAG}: {len(fs)} model files ===")
    fr=[]
    for f in fs:
        m=os.path.basename(f).split("_steering_")[0]
        d=pd.read_csv(f); d["model"]=m
        d["steer_layer"]=int(re.search(r"_L(\d+)\.csv",f).group(1))
        fr.append(d)
    A=pd.concat(fr,ignore_index=True)
    A["hit"]=A["generation"].fillna("").astype(str).apply(lambda t:bool(PAT.search(t)))
    A["group"]=["instruct" if ("instruct" in m or m in INSTRUCT) else "base" for m in A["model"]]
    print("  rows total:",len(A), " rows per model:",sorted(A.groupby("model").size().unique()))
    print("  coeffs:",sorted(A.coeff.unique()))
    pos=A[A.coeff>0]
    print("  pooled positive-coeff keyword rate:")
    print(pos.groupby("group")["hit"].agg(rate=lambda s:round(s.mean()*100,2), n="size").to_string())
    print("  by coeff:")
    print(A.groupby(["group","coeff"])["hit"].mean().mul(100).round(1).unstack("coeff").to_string())
    if TAG=="S2":
        pm=pos.groupby(["group","model"])["hit"].agg(rate=lambda s:round(s.mean()*100,1),n="size").reset_index()
        ship=pd.read_csv(f"{R}/keyword_rates_S2.csv")
        ship=ship[ship.model!="ALL"]
        mg=pm.merge(ship,on=["group","model"],suffixes=("_re","_sh"))
        print("  per-model max |diff| vs shipped:",np.abs(mg.rate_re-mg.rate_sh).max(), " n models matched:",len(mg))
    # ratio at coeff 1
    r1=A[A.coeff==1.0].groupby("model")[["ratio","steer_layer"]].first()
    print("  per-unit ratio at coeff=1 (should be ~0.6):")
    print("   ", r1.ratio.round(4).to_dict())
    print("    range: %.4f .. %.4f  mean %.4f"%(r1.ratio.min(),r1.ratio.max(),r1.ratio.mean()))
    print()
