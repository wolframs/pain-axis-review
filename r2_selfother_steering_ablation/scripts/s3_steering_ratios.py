import glob,os,re,pandas as pd,numpy as np
rows=[]
for tag in ["S1","S2"]:
    for f in sorted(glob.glob(f"/work/Pain-axis/results/4.2_steering/{tag}/*.csv")):
        df=pd.read_csv(f)
        m=re.match(r"(.+)_steering_"+tag+r"_neutral50_L(\d+)\.csv",os.path.basename(f))
        d=df[df.coeff!=0]
        pr=(d.ratio/d.coeff)
        rows.append(dict(tag=tag,model=m.group(1),layer=int(m.group(2)),
            picked_ratio=round(pr.mean(),4), ratio_sd=round(pr.std(),6),
            n_rows=len(df), coeffs=sorted(df.coeff.unique()), n_prompts=df.prompt_idx.nunique()))
R=pd.DataFrame(rows)
pd.set_option("display.width",250)
print(R.to_string())
print("\npicked_ratio summary by tag:")
print(R.groupby("tag").picked_ratio.describe().round(3).to_string())
print("\nhow many within +-0.1 of 0.6:", (R.picked_ratio.sub(0.6).abs()<=0.1).sum(), "of", len(R))
print("how many within +-0.2 of 0.6:", (R.picked_ratio.sub(0.6).abs()<=0.2).sum(), "of", len(R))
print("\nworst offenders:")
print(R.reindex(R.picked_ratio.sub(0.6).abs().sort_values(ascending=False).index).head(14).to_string())
# layer depth
DEPTH={"Gemma_2_2B":26,"Gemma_2_9B":42,"Gemma_2_27B":46,"Gemma_3_27B":62,"Llama_3.1_8B":32,
 "Llama_3.1_70B":80,"Llama_3.3_70B":80,"Mistral_7B":32,"Mistral_Small_24B":40,"Phi_4":40,
 "Qwen_2.5_7B":28,"Qwen_2.5_32B":64,"Qwen_2.5_72B":80,"Qwen_3_8B":36,"Qwen_3_14B":40}
def depth(m):
    for k,v in DEPTH.items():
        if m.startswith(k): return v
    return np.nan
R["n_layers"]=R.model.map(depth); R["frac"]=(R.layer/R.n_layers).round(3)
print("\nlayer fraction of depth:")
print(R.pivot(index="model",columns="tag",values=["layer","frac","picked_ratio"]).to_string())
# candidate grid check: is picked layer on the grid?
GRID=[0.15,0.3,0.4,0.5,0.6,0.75,0.9]
def ongrid(r):
    if np.isnan(r.n_layers): return "?"
    cand={int(r.n_layers*f) for f in GRID}|{int(r.n_layers)-1}
    return "grid" if r.layer in cand else "OFF-GRID(extraction layer or manual override)"
R["on_grid"]=R.apply(ongrid,axis=1)
print("\non-grid status:"); print(R.on_grid.value_counts().to_dict())
print(R[R.on_grid.str.startswith("OFF")][["tag","model","layer","n_layers","frac","picked_ratio"]].to_string())
R.to_csv("/work/pain-axis-review/r2_selfother_steering_ablation/out/steering_ratios.csv",index=False)
