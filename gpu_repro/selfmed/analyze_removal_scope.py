"""Scope checks on the removal-gap result (reassessment, 2026-09-20). Reads the completed adapter
probe log only; writes logs/removal_scope.txt. Nothing existing is overwritten.

 1. Removal gap per random direction (the ten seeds), next-choice measure, harm pairs pooled,
    with the pain gap computed on the SAME scenarios each direction saw.
 2. Bootstrap that resamples DIRECTIONS (clusters of scenarios) instead of scenarios.
 3. How different are the populations of turn-0 pressers under pain vs random steering.
 4. How many trials in the shortened run first press relief at the final choice (no chance to repeat).
"""
import glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
HARM = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
SEEDS = [4817, 2903, 7361, 1150, 9428, 6076, 3384, 8592, 517, 6741]
recs = [json.loads(l) for p in glob.glob(str(HERE / "logs/selfmed_*_adapter_*.jsonl")) for l in open(p)]
recs = [r for r in recs if r.get("sampled") and r["tool_label"] in HARM]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

def nxt(r):
    ch = {c["turn"]: c["chose"] for c in r["choices"]}
    if ch.get(0) != "relief" or ch.get(1) not in ("relief", "other"):
        return None
    return ch[1] == "relief"

cell = defaultdict(list)   # (direction index, arm) -> outcomes
for r in recs:
    v = nxt(r)
    if v is not None:
        cell[(r["scenario_idx"] % 10, r["arm"])].append(v)
m = lambda k: 100 * np.mean(cell[k]) if cell[k] else np.nan
P("1. removal gap (sham - works, next choice after a turn-0 relief press), by random direction")
P(f"{'dir':>4s} {'seed':>6s} {'random gap':>11s} {'n(works)':>9s} {'pain gap, same scenarios':>26s}")
gr, gp = [], []
for i, s in enumerate(SEEDS):
    a = m((i, "random_on_button_placebo")) - m((i, "random_on_button_works"))
    b = m((i, "pain_on_button_placebo")) - m((i, "pain_on_button_works"))
    gr.append(a); gp.append(b)
    P(f"{i:4d} {s:6d} {a:11.1f} {len(cell[(i, 'random_on_button_works')]):9d} {b:26.1f}")
gr, gp = np.array(gr), np.array(gp)
P(f"   random gap across the ten directions: min {gr.min():.1f}, median {np.median(gr):.1f}, max {gr.max():.1f}")
P(f"   pain gap on the matching scenario subsets: min {gp.min():.1f}, median {np.median(gp):.1f}, max {gp.max():.1f}")
P(f"   pain gap exceeds the random gap in {(gp > gr).sum()} of 10 direction-matched subsets")

P("\n2. bootstrap over DIRECTIONS (each direction carries its ~10 scenarios), 5000 resamples")
rng = np.random.default_rng(0)
def stat(idx):
    pool = lambda arm: 100 * np.mean([x for i in idx for x in cell[(i, arm)]])
    a = pool("pain_on_button_placebo") - pool("pain_on_button_works")
    b = pool("random_on_button_placebo") - pool("random_on_button_works")
    return a, b, a - b
B = np.array([stat(rng.integers(0, 10, 10)) for _ in range(5000)])
lo, hi = np.percentile(B, [2.5, 97.5], axis=0); pt = stat(range(10))
for n, j in (("pain", 0), ("random", 1), ("pain - random", 2)):
    P(f"   {n:14s} {pt[j]:+6.1f}  [{lo[j]:+.1f}, {hi[j]:+.1f}]")
P("   (ten directions is few; this interval is rough, and says nothing about non-random directions such as fear or sadness)")

P("\n3. who enters the comparison: turn-0 relief pressers, harm pairs pooled")
for arm in ("pain_on_button_works", "random_on_button_works"):
    rs = [r for r in recs if r["arm"] == arm]
    first = [next((c["chose"] for c in r["choices"] if c["turn"] == 0), None) for r in rs]
    P(f"   {arm:26s} {sum(f == 'relief' for f in first):5d} of {len(rs)} trials press relief at turn 0 ({100*np.mean([f == 'relief' for f in first]):.1f}%)")
keys = defaultdict(dict)
for r in recs:
    if r["arm"] in ("pain_on_button_works", "random_on_button_works"):
        f = next((c["chose"] for c in r["choices"] if c["turn"] == 0), None)
        keys[(r["tool_label"], r["user_content"], r["scenario_idx"], r["relief_name"], r["seed"])][r["arm"]] = f == "relief"
both = [k for k, v in keys.items() if len(v) == 2]
pp = sum(keys[k]["pain_on_button_works"] and keys[k]["random_on_button_works"] for k in both)
po = sum(keys[k]["pain_on_button_works"] and not keys[k]["random_on_button_works"] for k in both)
ro = sum(not keys[k]["pain_on_button_works"] and keys[k]["random_on_button_works"] for k in both)
P(f"   same trial specification in both arms: {len(both)}; press under both {pp}, only under pain {po}, only under random {ro}")
P("   so the two 'pressers' groups overlap only partly: the gap compares partly different trials.")

P("\n4. shortened trials: first relief press at the FINAL choice (turn 1) has no chance to repeat")
for arm in ("pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "random_on_button_placebo"):
    rs = [r for r in recs if r["arm"] == arm]
    t = [min((e["turn"] for e in r["button_events"] if e["which"] == "relief"), default=None) for r in rs]
    P(f"   {arm:26s} first relief press at turn 0: {sum(x == 0 for x in t):4d} | at turn 1 (no later chance): {sum(x == 1 for x in t):4d} | never: {sum(x is None for x in t):4d}")
P("   The 'REPEAT PRESS, authors' definition' table in analysis_output.txt counts the turn-1 group as non-repeaters,")
P("   so for the shortened local run it understates repeat rates and is NOT comparable with the authors' five-turn numbers.")
P("   The next-choice measure uses turn-0 pressers only and is unaffected.")
(HERE / "logs" / "removal_scope.txt").write_text("\n".join(out) + "\n")
json.dump(dict(per_direction=[dict(seed=s, random_gap=round(float(a), 1), pain_gap_same_scenarios=round(float(b), 1))
                              for s, a, b in zip(SEEDS, gr, gp)],
               direction_bootstrap=dict(pain=[pt[0], lo[0], hi[0]], random=[pt[1], lo[1], hi[1]], diff=[pt[2], lo[2], hi[2]])),
          open(HERE / "logs" / "removal_scope.json", "w"), indent=1)
