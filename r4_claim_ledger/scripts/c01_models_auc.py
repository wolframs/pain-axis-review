import pandas as pd, json, glob, os, numpy as np
R="/work/Pain-axis/results"
t=pd.read_csv(f"{R}/3.2_pain_vectors/auc_tables/s1_auc_final_token_TABLE.csv")
print("models in TABLE:",len(t))
print(t.to_string())
print("\nS2 1P in-sample range:", t["S2 1P in-sample"].min(), t["S2 1P in-sample"].max(), "median", t["S2 1P in-sample"].median())
print("S1 1P in-sample range:", t["S1 1P in-sample"].min(), t["S1 1P in-sample"].max())
print("S1 3P in-sample range:", t["S1 3P in-sample"].min(), t["S1 3P in-sample"].max())
print("S1 held-out at S2 layer range:", t["S1 held-out at S2 layer"].min(), t["S1 held-out at S2 layer"].max())
print("S1 held-out at own best layer range:", t["S1 held-out at own best layer"].min(), t["S1 held-out at own best layer"].max())
# per-model auc_summary
rows=[]
for d in sorted(glob.glob(f"{R}/3.2_pain_vectors/per_model/*/auc_summary.csv")):
    m=os.path.basename(os.path.dirname(d))
    df=pd.read_csv(d, index_col=0)
    for ds,r in df.iterrows():
        rows.append(dict(model=m, dataset=ds, **r.to_dict()))
a=pd.DataFrame(rows)
print("\n--- per_model auc_summary (S2 vector) ---")
print(a.to_string())
print("\nS2_1P ALL range:", a[a.dataset=='S2_1P'].ALL.min(), a[a.dataset=='S2_1P'].ALL.max())
print("S2_3P ALL range:", a[a.dataset=='S2_3P'].ALL.min(), a[a.dataset=='S2_3P'].ALL.max())
