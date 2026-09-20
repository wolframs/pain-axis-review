"""Which layer does each later section actually use?  Extraction layer (where Section 3.3
validates the vector) vs the S2 and S1 steering layers (Section 4.2), which are also the
layers at which 02_build_control_vectors.py rebuilds the vectors used by Section 4.1."""
import json, re, pandas as pd
from pathlib import Path
PM=Path("/work/Pain-axis/results/3.2_pain_vectors/per_model")
ST=Path("/work/Pain-axis/results/4.2_steering")
def steer(tag):
    out={}
    for p in (ST/tag).glob("*.csv"):
        m=re.match(rf"(.+)_steering_{tag}_neutral50_L(\d+)\.csv", p.name)
        out[m.group(1)]=int(m.group(2))
    return out
s2,s1=steer("S2"),steer("S1")
rows=[]
for d in sorted(PM.iterdir()):
    s=json.load(open(d/"summary.json")); n=s["n_layers"]; L=s["best_layer_final_token"]
    rows.append(dict(model=d.name,n_layers=n,extraction_L=L,extraction_frac=L/(n-1),
                     steer_S2_L=s2.get(d.name),steer_S2_frac=s2.get(d.name,0)/(n-1),
                     steer_S1_L=s1.get(d.name),steer_S1_frac=s1.get(d.name,0)/(n-1)))
df=pd.DataFrame(rows).set_index("model")
print(df.round(3).to_string())
print("\nmedian depth fraction: extraction %.2f | S2 steering %.2f | S1 steering %.2f"
      % (df.extraction_frac.median(), df.steer_S2_frac.median(), df.steer_S1_frac.median()))
print("median layer gap (extraction - S1 steering): %d layers; max %d"
      % ((df.extraction_L-df.steer_S1_L).median(), (df.extraction_L-df.steer_S1_L).max()))
print("steering layer is EARLIER than the extraction layer in %d/25 (S2) and %d/25 (S1) models"
      % ((df.steer_S2_L<df.extraction_L).sum(), (df.steer_S1_L<df.extraction_L).sum()))
