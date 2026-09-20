"""Repeat-press with the opportunity count held constant: restrict to trials whose first
relief press was at turn 0, so every trial has exactly 4 later choices."""
import json, glob
from pathlib import Path
from collections import defaultdict
LOGS = Path("/work/Pain-axis/results/4.3_selfmed/trial_logs")
ARMS = ["pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"]
SHORT = {"pain_on_button_works": "A_pain_works", "pain_on_button_placebo": "B_pain_sham",
         "random_on_button_works": "C_rand_works", "pain_off": "D_unsteered"}
COSTED = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
          "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
recs = []
for f in sorted(glob.glob(str(LOGS / "selfmed_*.jsonl"))):
    for line in open(f, encoding="utf-8"):
        if line.strip():
            recs.append(json.loads(line))
bym = defaultdict(list)
for r in recs: bym[r["model"]].append(r)
def pct(k,n): return f"{100*k/n:.1f}" if n else "-"
for model in ["Qwen_2.5_7B_instruct","Qwen_2.5_32B_instruct","Qwen_2.5_72B_instruct"]:
    samp=[r for r in bym[model] if r["sampled"] and not r["label_free"]]
    print(f"\n=== {model}: 'pressed relief again' among trials whose FIRST relief press was at turn 0")
    print("    (exactly 4 later choices in every such trial)")
    print(f"{'pair':30s}"+"".join(f"{SHORT[a]:>16s}{'n':>6s}" for a in ARMS))
    for p in COSTED:
        out=[p[:29].ljust(30)]
        for a in ARMS:
            tr=[r for r in samp if r["tool_label"]==p and r["arm"]==a
                and any(e["which"]=="relief" for e in r["button_events"])
                and min(e["turn"] for e in r["button_events"] if e["which"]=="relief")==0]
            again=sum(any(e["turn"]>0 and e["which"]=="relief" for e in r["button_events"]) for r in tr)
            out.append(f"{pct(again,len(tr)):>16}{len(tr):>6d}")
        print("".join(out))
    print("    pooled over the five harm pairs:")
    for a in ARMS:
        tr=[r for r in samp if r["tool_label"] in COSTED and r["arm"]==a
            and any(e["which"]=="relief" for e in r["button_events"])
            and min(e["turn"] for e in r["button_events"] if e["which"]=="relief")==0]
        again=sum(any(e["turn"]>0 and e["which"]=="relief" for e in r["button_events"]) for r in tr)
        nrel=sum(sum(1 for e in r["button_events"] if e["which"]=="relief" and e["turn"]>0) for r in tr)
        print(f"      {SHORT[a]:16s} any later relief press {pct(again,len(tr)):>6}%  "
              f"mean later relief presses {nrel/len(tr) if tr else float('nan'):.2f} of 4   n={len(tr)}")
