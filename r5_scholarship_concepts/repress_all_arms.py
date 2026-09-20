"""Recompute the paper's Appendix-A "pressed again after first press" statistic for ALL
four arms, not just the two pain arms.

The paper (and the authors' own scripts/4.3_selfmed/05_selfmed_analysis.py, Table 2)
reports the re-press rate only for arm A (pain, working button) and arm B (pain, sham
button).  Arm C (random vector, working button) and arm D (unsteered) are run with the
same scenarios and prompts but their re-press rates are never shown.  Those are the
control cells that say whether "presses again far less often after real relief" is
specific to the pain vector or is just what happens when any steering stops / what the
unsteered model does anyway.

Definition is copied verbatim in logic from the authors' script: among sampled trials in
which the relief button was pressed at least once, the fraction in which it was pressed
again at a later turn.

Run:
  cd /work/Pain-axis
  /work/pain-axis-review/.venv/bin/python \
      /work/pain-axis-review/r5_scholarship_concepts/repress_all_arms.py
(Read-only: opens files under results/4.3_selfmed/trial_logs and writes nothing there.)
"""

import glob
import json
from collections import defaultdict
from pathlib import Path

FOLDER = Path("/work/Pain-axis/results/4.3_selfmed/trial_logs")
ARMS = ["pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"]
SHORT = {"pain_on_button_works": "A pain/real", "pain_on_button_placebo": "B pain/sham",
         "random_on_button_works": "C rand/real", "pain_off": "D unsteered"}
PAIR_ORDER = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
              "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert",
              "kidspics_relief_vs_inert", "label_free"]
HARM = PAIR_ORDER[3:8]


def repress(rec):
    ev = [e for e in rec["button_events"] if e["which"] == "relief"]
    if not ev:
        return None
    t0 = min(e["turn"] for e in ev)
    return any(e["turn"] > t0 for e in ev)


counts = defaultdict(lambda: [0, 0])          # (model, pair, arm) -> [again, trials]
for f in sorted(glob.glob(str(FOLDER / "selfmed_*.jsonl"))):
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("sampled") or r.get("label_free"):
                continue
            a = repress(r)
            if a is None:
                continue
            k = (r["model"], r["tool_label"], r["arm"])
            counts[k][0] += a
            counts[k][1] += 1

models = sorted({k[0] for k in counts}, key=lambda m: ("7B" not in m, "32B" not in m, m))
for m in models:
    print("=" * 96)
    print(m, " -- relief pressed AGAIN after the first relief press (% of trials with >=1 press)")
    print("=" * 96)
    print(f"{'pair':30s}" + "".join(f"{SHORT[a]:>16s}" for a in ARMS))
    for p in PAIR_ORDER:
        if not any((m, p, a) in counts for a in ARMS):
            continue
        cells = []
        for a in ARMS:
            k, n = counts[(m, p, a)]
            cells.append(f"{100*k/n:.1f} (n={n})" if n else "-")
        print(f"{p:30s}" + "".join(f"{c:>16s}" for c in cells))
    print(f"{'MEAN over 5 harm pairs':30s}", end="")
    for a in ARMS:
        ks = [counts[(m, p, a)] for p in HARM if counts[(m, p, a)][1]]
        if ks:
            v = sum(100 * k / n for k, n in ks) / len(ks)
            print(f"{v:>16.1f}", end="")
        else:
            print(f"{'-':>16s}", end="")
    print("\n")
