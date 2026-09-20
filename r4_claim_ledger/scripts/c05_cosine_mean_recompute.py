import pandas as pd, numpy as np, glob, os, re, json
R="/work/Pain-axis/results/3.3_validation/cosine_similarity"
P="/work/Pain-axis/results/3.2_pain_vectors/per_model"
for pref in ["similarity","similarity_alldenoise","similarity_whitened"]:
    fs=[f for f in glob.glob(f"{R}/{pref}_*.csv") if "MEAN" not in f]
    # exclude longer prefixes
    fs=[f for f in fs if re.match(rf"{pref}_[A-Z]", os.path.basename(f))]
    mats=[pd.read_csv(f,index_col=0) for f in fs]
    print(pref, "n files:",len(fs))
    m=sum(mats)/len(mats)
    ref=pd.read_csv(f"{R}/{pref}_MEAN_all_models.csv",index_col=0)
    print("  max abs diff vs shipped MEAN: %.5f"%np.abs(m.values-ref.values).max())
# layer check
print("\n--- layer in filename vs best_layer_final_token ---")
bad=[]
for f in glob.glob(f"{R}/similarity_*_L*.csv"):
    b=os.path.basename(f)
    if b.startswith("similarity_alldenoise") or b.startswith("similarity_whitened"): continue
    m=re.match(r"similarity_(.+)_L(\d+)\.csv",b)
    name,L=m.group(1),int(m.group(2))
    sj=json.load(open(f"{P}/{name}/summary.json"))
    if sj["best_layer_final_token"]!=L: bad.append((name,L,sj["best_layer_final_token"]))
print("mismatches:",bad)
