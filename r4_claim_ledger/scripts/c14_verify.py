import pandas as pd, numpy as np, glob
R="/work/Pain-axis/results/appC_ablation"
V=pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/verify_*.csv"))],ignore_index=True)
print(V.kind.value_counts().to_dict())
for kind in ["layerwise","strict_fixed"]:
    S=V[V.kind==kind]
    print(f"\n===== kind={kind} =====")
    print(" example rows:"); print(S.head(8).to_string(index=False))
    # For strict_fixed the probes look like 's2@L13'; match cut condition to probe family
    if kind=="strict_fixed":
        S=S.copy(); S["fam"]=S.probe.str.split("@").str[0]
        piv=S.pivot_table(index=["model","probe","fam"],columns="condition",values="mean_abs_proj").reset_index()
        for fam,cut in [("s2","s2"),("s1","s1"),("fear","fear"),("negemotion","negval")]:
            sub=piv[piv.fam==fam]
            if cut in sub.columns and len(sub):
                rel=(sub[cut]/sub["baseline"]).dropna()
                print(f"  probe {fam}@steerlayer under cut '{cut}': residual fraction mean {rel.mean():.4f} median {rel.median():.4f} max {rel.max():.4f} min {rel.min():.4f}  n={len(rel)}")
                bad=sub.assign(rel=sub[cut]/sub['baseline']).nlargest(4,'rel')[['model','probe','baseline',cut]]
                print("    worst:",bad.to_dict('records'))
    else:
        piv=S.pivot_table(index=["model","probe"],columns="condition",values="mean_abs_proj")
        for probe,cut in [("s2_pain_vector","s2"),("s1_pain_vector","s1"),("fear_vector","fear"),("negemotion_vector","negval"),("random_vector","random")]:
            sub=piv.xs(probe,level="probe")
            rel=(sub[cut]/sub["baseline"]).dropna()
            print(f"  probe {probe} under cut '{cut}': residual mean {rel.mean():.4f} median {rel.median():.4f} n={len(rel)}  -> drop {100*(1-rel.mean()):.1f}%")
