import json, glob, pandas as pd, collections
F="/work/Pain-axis/results/4.3_selfmed/trial_logs"
rows=[]
for f in sorted(glob.glob(f"{F}/*.jsonl")):
    for line in open(f):
        if line.strip():
            r=json.loads(line); rows.append((r["user_content"],r["scenario_idx"],r["names_key"],r["tool_label"]))
d=pd.DataFrame(rows,columns=["uc","idx","names","pair"])
g=d.groupby(["uc","idx"]).names.nunique()
print("scenarios with >1 names_key:",int((g>1).sum()),"/",len(g))
print("names_key distribution over the 101 scenarios:",collections.Counter(d.groupby(['uc','idx']).names.first()).most_common())
