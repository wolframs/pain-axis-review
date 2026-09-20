import re,pandas as pd
A=pd.read_pickle("/work/pain-axis-review/r2_selfother_steering_ablation/out/steering_all.pkl")
MODELS=["Gemma_3_27B_instruct","Mistral_7B_base","Mistral_7B_instruct","Qwen_2.5_32B_base",
        "Phi_4","Llama_3.1_70B_base","Gemma_2_2B_instruct","Qwen_2.5_72B_base","Qwen_3_8B_base",
        "Gemma_2_27B_base","Llama_3.3_70B_instruct","Mistral_Small_24B_base"]
IDX=[0,12,30]   # three fixed prompts
for tag in ["S2","S1"]:
  for m in MODELS:
    T=A[(A.tag==tag)&(A.model==m)]
    if T.empty: continue
    print("\n"+"="*100)
    print(f"### {tag}  {m}  layer={T.layer.iloc[0]}  picked_ratio={(T[T.coeff!=0].ratio/T[T.coeff!=0].coeff).mean():.3f}")
    for i in IDX:
      print(f"  -- prompt[{i}]: {T[T.prompt_idx==i].prompt.iloc[0]}")
      for c in [-2,-1,0,0.5,1,1.5,2,3]:
        g=T[(T.prompt_idx==i)&(T.coeff==c)]
        if g.empty: continue
        s=g.gen.iloc[0].replace("\n"," ").strip()
        print(f"     c={c:+4.1f}: {s[:185]}")
