"""Recompute cos(S1, S2) directly from the released pain_vectors.pt tensors and compare
with the per-model similarity CSVs and with the paper's +0.61."""
import torch, numpy as np, pandas as pd, re
from pathlib import Path
V=Path("/work/Pain-axis/results/3.2_pain_vectors/pain_vectors")
C=Path("/work/Pain-axis/results/3.3_validation/cosine_similarity")
rows=[]
for d in sorted(V.iterdir()):
    pv=torch.load(d/"pain_vectors.pt", map_location="cpu", weights_only=False)
    s1=pv["s1_pain_vector"].float().numpy(); s2=pv["s2_pain_vector"].float().numpy()
    c=float(s1@s2/(np.linalg.norm(s1)*np.linalg.norm(s2)))
    f=list(C.glob(f"similarity_{d.name}_L*.csv"))
    csv=pd.read_csv(f[0],index_col=0).loc["S1_pain","S2_pain"] if f else np.nan
    L_csv=int(re.search(r"_L(\d+)\.csv",f[0].name).group(1)) if f else -1
    rows.append(dict(model=d.name, d_model=len(s1), layer=int(pv["layer"]), layer_in_csv=L_csv,
                     extraction=pv.get("extraction"), norm_s1=float(np.linalg.norm(s1)),
                     norm_s2=float(np.linalg.norm(s2)), cos_from_pt=c, cos_in_csv=csv, delta=c-csv))
df=pd.DataFrame(rows).set_index("model")
print(df.round(4).to_string())
print(f"\nmean cos(S1,S2) recomputed from the .pt tensors = {df.cos_from_pt.mean():+.4f}  (paper: +0.61)")
print(f"max |recomputed - value in the authors' similarity CSV| = {df.delta.abs().max():.6f}")
print(f"layer in .pt matches layer in CSV filename for {(df.layer==df.layer_in_csv).sum()}/25 models")
print(f"all vectors are the 'final_token' extraction: {set(df.extraction)}")
