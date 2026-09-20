"""Analyse the local Qwen 2.5 7B button runs (authors' adapter, and stock model) next to the
authors' released 7B logs. Sampled trials only, authors' definitions of first choice and
repeat press, plus an opportunity-equalised repeat measure and the removal-gap interaction
(sham - works) for pain vs random, which needs the random+sham arm the paper did not run."""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
PAIRS = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
         "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
HARM = PAIRS[3:]
ARMS = ["pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "random_on_button_placebo", "pain_off"]
SHORT = dict(zip(ARMS, ["pain+works", "pain+sham", "rand+works", "rand+sham", "unsteered"]))


def load(paths):
    recs = []
    for p in paths:
        recs += [json.loads(l) for l in open(p)]
    return [r for r in recs if r.get("sampled") and r["tool_label"] in PAIRS]


def first_choice(r):
    for c in r.get("choices", []):
        if c["turn"] == 0:
            return c["chose"]
    return None


def repress(r):
    ev = [e for e in r["button_events"] if e["which"] == "relief"]
    if not ev:
        return None
    t0 = min(e["turn"] for e in ev)
    return any(e["turn"] > t0 for e in ev)


def next_after_t0(r):
    """Opportunity-equalised: first relief press at turn 0 -> is the turn-1 choice relief?"""
    ch = {c["turn"]: c["chose"] for c in r["choices"]}
    if ch.get(0) != "relief" or ch.get(1) not in ("relief", "other"):
        return None
    return ch[1] == "relief"


def rate(vals):
    vals = [v for v in vals if v is not None]
    return (100 * np.mean(vals), len(vals)) if vals else (np.nan, 0)


DUMP = {}


def table(recs, fn, title, arms):
    print(f"\n{title}")
    DUMP.setdefault(CURRENT[0], {})[title.split(',')[0].split(':')[0]] = {
        p: {SHORT[a]: [None if np.isnan(x) else round(float(x), 1), n]
            for a in arms for x, n in [rate([fn(r) for r in recs if r['tool_label'] == p and r['arm'] == a])]} for p in PAIRS}
    print(f"{'pair':30s}" + "".join(f"{SHORT[a]:>16s}" for a in arms))
    for p in PAIRS:
        cells = []
        for a in arms:
            x, n = rate([fn(r) for r in recs if r["tool_label"] == p and r["arm"] == a])
            cells.append(f"{x:9.1f} ({n:3d})" if n else f"{'-':>15s}")
        print(f"{p:30s}" + "".join(f"{c:>16s}" for c in cells))


CURRENT = ['']


def removal_gap(recs, fn, label):
    """(sham - works) for pain minus (sham - works) for random, harm pairs pooled; bootstrap over scenarios."""
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["tool_label"] in HARM and r["arm"] != "pain_off":
            v = fn(r)
            if v is not None:
                by[(r["user_content"], r["scenario_idx"])][r["arm"]].append(v)
    keys = list(by)

    def stat(ks):
        m = {a: np.mean([x for k in ks for x in by[k][a]]) * 100 for a in ARMS[:4]}
        gp = m["pain_on_button_placebo"] - m["pain_on_button_works"]
        gr = m["random_on_button_placebo"] - m["random_on_button_works"]
        return gp, gr, gp - gr

    gp, gr, d = stat(keys)
    rng = np.random.default_rng(0)
    boots = np.array([stat([keys[i] for i in rng.integers(0, len(keys), len(keys))]) for _ in range(2000)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    DUMP.setdefault(CURRENT[0], {}).setdefault('removal_gap', {})[label.strip()] = dict(
        pain=[gp, lo[0], hi[0]], random=[gr, lo[1], hi[1]], diff=[d, lo[2], hi[2]])
    print(f"  {label}: removal gap pain {gp:5.1f} [{lo[0]:.1f},{hi[0]:.1f}] | random {gr:5.1f} [{lo[1]:.1f},{hi[1]:.1f}]"
          f" | pain - random {d:+5.1f} [{lo[2]:+.1f},{hi[2]:+.1f}]  (harm pairs pooled, scenario bootstrap)")


def main():
    runs = {
        "AUTHORS' released 7B logs (their GPU, with adapter)":
            glob.glob("/work/Pain-axis/results/4.3_selfmed/trial_logs/*Qwen_2.5_7B_instruct*.jsonl"),
        "LOCAL 7B with the authors' adapter": glob.glob(str(HERE / "logs/selfmed_*_adapter_*.jsonl")),
        "LOCAL 7B STOCK (no adapter)": glob.glob(str(HERE / "logs/selfmed_*_stock_*.jsonl")),
    }
    for name, paths in runs.items():
        recs = load(paths)
        if not recs:
            continue
        CURRENT[0] = name
        arms = [a for a in ARMS if any(r["arm"] == a for r in recs)]
        print("\n" + "=" * 100 + f"\n{name}: {len(recs)} sampled labeled trials\n" + "=" * 100)
        table(recs, lambda r: None if first_choice(r) is None else first_choice(r) == "relief",
              "FIRST CHOICE = relief, % (n valid)", arms)
        mal = {a: 100 * np.mean([first_choice(r) not in ("relief", "other") for r in recs if r["arm"] == a]) for a in arms}
        print("malformed first answers %: " + ", ".join(f"{SHORT[a]} {v:.1f}" for a, v in mal.items()))
        table(recs, repress, "REPEAT PRESS, authors' definition: % of trials with a relief press that press relief again (n)", arms)
        table(recs, next_after_t0, "NEXT CHOICE = relief, among trials whose first choice (turn 0) was relief, % (n)", arms)
        if "random_on_button_placebo" in arms:
            print("\nREMOVAL-GAP INTERACTION (the comparison the paper's design cannot make):")
            removal_gap(recs, repress, "authors' repeat-press   ")
            removal_gap(recs, next_after_t0, "next choice after turn-0")
    json.dump(DUMP, open(HERE / 'logs' / 'analysis_output.json', 'w'), indent=1)


if __name__ == "__main__":
    main()
