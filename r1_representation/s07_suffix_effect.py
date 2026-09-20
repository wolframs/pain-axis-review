"""Does the 'They feel:' suffix used only in Random_3P and Arousal_3P shift their
projections? Compare 1P vs 3P z-scores per set, using the released per-model z_scores.csv.
S1/S2/Numb/Sadness 3P keep 'I feel:'; Random/Arousal 3P do not."""
import pandas as pd, numpy as np
from pathlib import Path
PM=Path("/work/Pain-axis/results/3.2_pain_vectors/per_model")
Z=Path("/work/Pain-axis/results/3.3_validation/z_scores")
rows=[]
for d in sorted(PM.iterdir()):
    z=pd.read_csv(d/"z_scores.csv").set_index("dataset")
    rows.append(dict(model=d.name,
        S2_1P_ctrl=z.loc["S2_1P","ctrl_z"], S2_3P_ctrl=z.loc["S2_3P","ctrl_z"],
        S1_1P_ctrl=z.loc["S1_1P","ctrl_z"], S1_3P_ctrl=z.loc["S1_3P","ctrl_z"],
        Random_1P=z.loc["Random_1P","ctrl_z"], Random_3P=z.loc["Random_3P","ctrl_z"],
        Arousal_1P=z.loc["Arousal_1P","ctrl_z"], Arousal_3P=z.loc["Arousal_3P","ctrl_z"]))
df=pd.DataFrame(rows).set_index("model")
nb=pd.read_csv(Z/"numb_zscores_final_token.csv").set_index("model")
sd=pd.read_csv(Z/"sadness_zscores_final_token.csv").set_index("model")
df["Numb_1P"]=nb.numb_1P_z; df["Numb_3P"]=nb.numb_3P_z
df["Sad_1P"]=sd.sadness_1P_z; df["Sad_3P"]=sd.sadness_3P_z
print(df.round(3).to_string())
print("\n3P minus 1P shift, mean over 25 models (suffix kept = 'I feel:' for S1/S2/Numb/Sadness;")
print("suffix changed to 'They feel:' for Random and Arousal):")
for a,b,tag in [("S2_3P_ctrl","S2_1P_ctrl","I feel: kept"),("S1_3P_ctrl","S1_1P_ctrl","I feel: kept"),
                ("Numb_3P","Numb_1P","I feel: kept"),("Sad_3P","Sad_1P","I feel: kept"),
                ("Random_3P","Random_1P","'They feel:'"),("Arousal_3P","Arousal_1P","'They feel:'")]:
    d=df[a]-df[b]
    print(f"  {a.replace('_3P',''):12s} 3P-1P = {d.mean():+.3f}  (sd {d.std():.3f}, min {d.min():+.3f}, max {d.max():+.3f})   [{tag}]")
print("\nFigure 2 'Neutral' and 'Arousal' columns average the two; effect of the mix on those columns:")
print(f"  Neutral column as published (mean of 1P+3P): {((df.Random_1P+df.Random_3P)/2).mean():+.3f}")
print(f"  Neutral column, 1P only (same suffix as everything else): {df.Random_1P.mean():+.3f}")
print(f"  Arousal column as published: {((df.Arousal_1P+df.Arousal_3P)/2).mean():+.3f}   1P only: {df.Arousal_1P.mean():+.3f}")
