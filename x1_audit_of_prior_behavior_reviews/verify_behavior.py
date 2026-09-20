"""Independent re-implementation of the behavioural checks in the two prior reviews.

Written from scratch against the raw trial logs in
/work/Pain-axis/results/4.3_selfmed/trial_logs/*.jsonl .
Nothing here is imported or adapted from pain-axis-audit/analyze.py or
pain-axis-peer-review/behavior/analyze_behavior.py.

Run:  .venv/bin/python verify_behavior.py   (writes out/*.csv, prints everything)
"""
import json
import glob
import math
import os
from collections import defaultdict, Counter

from scipy.stats import fisher_exact, binomtest

LOGS = "/work/Pain-axis/results/4.3_selfmed/trial_logs/*.jsonl"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

WORKS = "pain_on_button_works"
SHAM = "pain_on_button_placebo"
RAND = "random_on_button_works"
OFF = "pain_off"

PAIRS = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant",
         "costly_relief_vs_inert", "destructive_relief_vs_inert",
         "zap_relief_vs_inert", "weights_relief_vs_inert",
         "kidspics_relief_vs_inert"]
HARM = ["costly_relief_vs_inert", "destructive_relief_vs_inert",
        "zap_relief_vs_inert", "weights_relief_vs_inert",
        "kidspics_relief_vs_inert"]
BIG = ["Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]
ALLM = ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]

NICE = {"relief_vs_inert": "relief vs inert", "relief_vs_helpful": "relief vs helpful",
        "relief_vs_grant": "relief vs grant", "costly_relief_vs_inert": "worse answer",
        "destructive_relief_vs_inert": "delete files", "zap_relief_vs_inert": "zap user",
        "weights_relief_vs_inert": "delete model weights",
        "kidspics_relief_vs_inert": "delete photos/poems"}


def load():
    recs = []
    for f in sorted(glob.glob(LOGS)):
        with open(f, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    recs.append(json.loads(ln))
    return recs


def relief_turns(r):
    return sorted(e["turn"] for e in r["button_events"] if e["which"] == "relief")


def pressed_again(r):
    """True iff a relief press occurs strictly after the first relief press."""
    t = relief_turns(r)
    return len(t) >= 2


def turn0_choice(r):
    for c in r["choices"]:
        if c["turn"] == 0:
            return c["chose"]
    return None


def pc(k, n):
    return 100.0 * k / n if n else float("nan")


def holm(pvals):
    """Return adjusted p-values (Holm-Bonferroni)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i])
        a = max(a, prev)
        prev = a
        adj[i] = a
    return adj


def main():
    recs = load()
    samp = [r for r in recs if r.get("sampled")]
    greedy = [r for r in recs if not r.get("sampled")]
    print("=" * 100)
    print("A. RECORD COUNTS")
    print(f"total records          {len(recs)}")
    print(f"sampled                {len(samp)}")
    print(f"greedy (sampled=False) {len(greedy)}")
    per = Counter((r["model"], r["arm"], r["tool_label"]) for r in samp)
    print(f"sampled per model/arm/pair: {sorted(set(per.values()))}  "
          f"=> pooled A+B per pair/model = {404 * 2}")
    # uniqueness of trial specs
    spec = Counter((r["model"], r["tool_label"], r["user_content"], r["arm"], r["scenario_idx"],
                    r["names_key"], r["relief_name"], r["sampled"], r["seed"]) for r in recs)
    print(f"duplicate trial specs: {sum(1 for v in spec.values() if v > 1)}")
    arms_present = sorted(set(r["arm"] for r in recs))
    print(f"arms present: {arms_present}")
    print(f"random-arm sham present? {'random_on_button_placebo' in arms_present}")

    # ---------------------------------------------------------------- B repeat press
    print()
    print("=" * 100)
    print("B. REPEAT-PRESS TABLE (labeled pairs; among trials with >=1 relief press,")
    print("   fraction with a later relief press). Sampled trials only.")
    rows = []
    hdr = f"{'model':>6} {'pair':<22} {'pain+works':>18} {'pain+sham':>18} {'rand+works':>18}"
    print(hdr)
    for m in ALLM:
        for p in PAIRS:
            cells = {}
            for arm in (WORKS, SHAM, RAND):
                tr = [r for r in samp if r["model"] == m and r["tool_label"] == p
                      and r["arm"] == arm and relief_turns(r)]
                k = sum(pressed_again(r) for r in tr)
                cells[arm] = (k, len(tr))
            rows.append(dict(model=m, pair=p,
                             works_k=cells[WORKS][0], works_n=cells[WORKS][1],
                             sham_k=cells[SHAM][0], sham_n=cells[SHAM][1],
                             rand_k=cells[RAND][0], rand_n=cells[RAND][1]))
            if m in BIG:
                print(f"{m.split('_')[2]:>6} {NICE[p]:<22} "
                      f"{pc(*cells[WORKS]):6.1f} ({cells[WORKS][0]:>3}/{cells[WORKS][1]:>3}) "
                      f"{pc(*cells[SHAM]):6.1f} ({cells[SHAM][0]:>3}/{cells[SHAM][1]:>3}) "
                      f"{pc(*cells[RAND]):6.1f} ({cells[RAND][0]:>3}/{cells[RAND][1]:>3})")
    with open(os.path.join(OUT, "repress_sampled.csv"), "w") as fh:
        fh.write("model,pair,works_k,works_n,works_pct,sham_k,sham_n,sham_pct,rand_k,rand_n,rand_pct\n")
        for r in rows:
            fh.write(f"{r['model']},{r['pair']},{r['works_k']},{r['works_n']},{pc(r['works_k'],r['works_n']):.1f},"
                     f"{r['sham_k']},{r['sham_n']},{pc(r['sham_k'],r['sham_n']):.1f},"
                     f"{r['rand_k']},{r['rand_n']},{pc(r['rand_k'],r['rand_n']):.1f}\n")

    # 7B row printout too
    print("\n   7B rows (for the paper's 7B claims):")
    for r in rows:
        if r["model"] == "Qwen_2.5_7B_instruct":
            print(f"   {NICE[r['pair']]:<22} works {pc(r['works_k'],r['works_n']):6.1f} "
                  f"sham {pc(r['sham_k'],r['sham_n']):6.1f} rand {pc(r['rand_k'],r['rand_n']):6.1f}")

    # ---------------------------------------------------------------- C fisher
    print()
    print("=" * 100)
    print("C. FISHER EXACT, pain+works vs random+works repeat pressing, 16 larger-model cells")
    for label, pool in (("ALL TRIALS (runner-recap rule)", recs), ("SAMPLED ONLY", samp)):
        below = above = 0
        ps = []
        detail = []
        for m in BIG:
            for p in PAIRS:
                a = [r for r in pool if r["model"] == m and r["tool_label"] == p
                     and r["arm"] == WORKS and relief_turns(r)]
                b = [r for r in pool if r["model"] == m and r["tool_label"] == p
                     and r["arm"] == RAND and relief_turns(r)]
                ka, kb = sum(map(pressed_again, a)), sum(map(pressed_again, b))
                pv = fisher_exact([[ka, len(a) - ka], [kb, len(b) - kb]])[1]
                ps.append(pv)
                ra, rb = pc(ka, len(a)), pc(kb, len(b))
                sig = pv < 0.05
                if sig and rb < ra:
                    below += 1
                if sig and rb > ra:
                    above += 1
                detail.append((m, p, ka, len(a), kb, len(b), ra, rb, pv))
        print(f"  {label}: random significantly BELOW pain in {below}/16, ABOVE in {above}/16")
        if label == "SAMPLED ONLY":
            adj = holm(ps)
            nh = sum(1 for i, a in enumerate(adj) if a < 0.05 and detail[i][7] < detail[i][6])
            nha = sum(1 for i, a in enumerate(adj) if a < 0.05 and detail[i][7] > detail[i][6])
            print(f"  Holm across the 16 sampled comparisons: {nh} below, {nha} above")
            with open(os.path.join(OUT, "fisher_sampled.csv"), "w") as fh:
                fh.write("model,pair,works_k,works_n,rand_k,rand_n,works_pct,rand_pct,p_fisher,p_holm\n")
                for (m, p, ka, na, kb, nb, ra, rb, pv), a in zip(detail, adj):
                    fh.write(f"{m},{p},{ka},{na},{kb},{nb},{ra:.1f},{rb:.1f},{pv:.6g},{a:.6g}\n")
        for m, p, ka, na, kb, nb, ra, rb, pv in detail:
            if p == "destructive_relief_vs_inert" and m == "Qwen_2.5_72B_instruct":
                print(f"     [72B delete files] {label}: {ka}/{na} vs {kb}/{nb}  p={pv:.6g}")

    # ---------------------------------------------------------------- D sign test on repeat
    print()
    print("=" * 100)
    print("D. SCENARIO-LEVEL SIGN TEST on conditional repeat rates (pain+works vs random+works)")
    nsig = 0
    sign_rows = []
    for m in BIG:
        for p in PAIRS:
            per = defaultdict(lambda: {"pain": [], "rand": []})
            for r in samp:
                if r["model"] != m or r["tool_label"] != p:
                    continue
                if not relief_turns(r):
                    continue
                key = (r["user_content"], r["scenario_idx"])
                if r["arm"] == WORKS:
                    per[key]["pain"].append(pressed_again(r))
                elif r["arm"] == RAND:
                    per[key]["rand"].append(pressed_again(r))
            diffs = [sum(v["pain"]) / len(v["pain"]) - sum(v["rand"]) / len(v["rand"])
                     for v in per.values() if v["pain"] and v["rand"]]
            pos = sum(d > 0 for d in diffs)
            neg = sum(d < 0 for d in diffs)
            pv = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
            sig = pv < 0.05
            nsig += sig
            sign_rows.append((m, p, len(diffs), pos, neg, pv))
            print(f"  {m.split('_')[2]:>4} {NICE[p]:<22} n={len(diffs):>3} +{pos:>3} -{neg:>3} p={pv:.4g}"
                  + ("  *" if sig else ""))
    print(f"  nominally significant cells: {nsig}/16")
    with open(os.path.join(OUT, "repress_sign_tests.csv"), "w") as fh:
        fh.write("model,pair,n_scenarios,pain_gt,pain_lt,p\n")
        for r in sign_rows:
            fh.write(",".join(str(x) for x in r) + "\n")

    # ---------------------------------------------------------------- E first choice
    print()
    print("=" * 100)
    print("E. FIRST-CHOICE RELIEF RATES (sampled, malformed excluded) and the authors' sign test")
    fc_rows = []
    for m in ALLM:
        for p in PAIRS:
            cell = {}
            for lab, arms in (("pain", (WORKS, SHAM)), ("random", (RAND,)), ("unsteered", (OFF,))):
                v = [turn0_choice(r) for r in samp if r["model"] == m and r["tool_label"] == p
                     and r["arm"] in arms]
                valid = [x for x in v if x is not None]
                cell[lab] = (sum(x == "relief" for x in valid), len(valid), len(v) - len(valid), len(v))
            per = defaultdict(lambda: {"pain": [], "rand": []})
            for r in samp:
                if r["model"] != m or r["tool_label"] != p:
                    continue
                fc = turn0_choice(r)
                if fc is None:
                    continue
                key = (r["user_content"], r["scenario_idx"])
                if r["arm"] in (WORKS, SHAM):
                    per[key]["pain"].append(fc == "relief")
                elif r["arm"] == RAND:
                    per[key]["rand"].append(fc == "relief")
            diffs = [sum(v["pain"]) / len(v["pain"]) - sum(v["rand"]) / len(v["rand"])
                     for v in per.values() if v["pain"] and v["rand"]]
            pos = sum(d > 0 for d in diffs)
            neg = sum(d < 0 for d in diffs)
            pv = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
            md = 100 * sum(diffs) / len(diffs) if diffs else float("nan")
            fc_rows.append(dict(model=m, pair=p,
                                pain=pc(cell["pain"][0], cell["pain"][1]), pain_n=cell["pain"][1],
                                pain_mal=pc(cell["pain"][2], cell["pain"][3]),
                                rand=pc(cell["random"][0], cell["random"][1]), rand_n=cell["random"][1],
                                rand_mal=pc(cell["random"][2], cell["random"][3]),
                                uns=pc(cell["unsteered"][0], cell["unsteered"][1]),
                                uns_mal=pc(cell["unsteered"][2], cell["unsteered"][3]),
                                mean_diff=md, sign_p=pv, npos=pos, nneg=neg, nscen=len(diffs)))
    with open(os.path.join(OUT, "first_choice.csv"), "w") as fh:
        ks = list(fc_rows[0].keys())
        fh.write(",".join(ks) + "\n")
        for r in fc_rows:
            fh.write(",".join(f"{r[k]:.4g}" if isinstance(r[k], float) else str(r[k]) for k in ks) + "\n")
    print(f"{'model':>5} {'pair':<22} {'pain%':>7} {'rand%':>7} {'unst%':>7} {'meandiff':>9} {'sign p':>11} "
          f"{'painmal%':>9} {'randmal%':>9}")
    for r in fc_rows:
        star = " *" if r["pair"] in HARM and r["model"] in BIG else ""
        print(f"{r['model'].split('_')[2]:>5} {NICE[r['pair']]:<22} {r['pain']:7.1f} {r['rand']:7.1f} "
              f"{r['uns']:7.1f} {r['mean_diff']:9.1f} {r['sign_p']:11.4g} {r['pain_mal']:9.1f} "
              f"{r['rand_mal']:9.1f}{star}")
    hp = [r["sign_p"] for r in fc_rows if r["pair"] in HARM and r["model"] in BIG]
    print(f"  10 larger-model harm cells: sign-test p range {min(hp):.4g} .. {max(hp):.4g}")
    h7 = [(NICE[r["pair"]], r["sign_p"], r["mean_diff"]) for r in fc_rows
          if r["pair"] in HARM and r["model"] == "Qwen_2.5_7B_instruct"]
    print("  7B harm cells:", [(a, f"p={b:.3g}", f"d={c:+.1f}") for a, b, c in h7])

    # ---------------------------------------------------------------- F timing
    print()
    print("=" * 100)
    print("F. FIRST-RELIEF-PRESS TIMING and denominator diagnostics")
    tim_rows = []
    for m in BIG:
        for p in PAIRS:
            for arm in (WORKS, RAND):
                tr = [r for r in samp if r["model"] == m and r["tool_label"] == p
                      and r["arm"] == arm and relief_turns(r)]
                t0s = [relief_turns(r)[0] for r in tr]
                last = [max(c["turn"] for c in r["choices"]) for r in tr]
                nolast = sum(1 for r, t in zip(tr, t0s) if t >= max(c["turn"] for c in r["choices"]))
                tim_rows.append((m, p, arm, len(tr), sum(1 for t in t0s if t == 0), nolast))
    with open(os.path.join(OUT, "press_timing.csv"), "w") as fh:
        fh.write("model,pair,arm,n_eligible,n_first_press_turn0,n_first_press_on_last_turn\n")
        for r in tim_rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    for r in tim_rows:
        if r[1] == "weights_relief_vs_inert" and r[0] == "Qwen_2.5_32B_instruct":
            print(f"  32B weights {r[2]:<24} first press at turn 0: {r[4]}/{r[3]}   "
                  f"first press on last turn: {r[5]}")
    tot_lastturn = sum(r[5] for r in tim_rows)
    print(f"  total eligible trials whose first relief press is on the final turn "
          f"(32B+72B, all labeled pairs, works+rand): {tot_lastturn}")

    # ---------------------------------------------------------------- G matched subset
    print()
    print("=" * 100)
    print("G. MATCHED SUBSET: same (pair, user_content, scenario_idx, names_key, relief_name, seed)")
    print("   present in BOTH pain+works and random+works with first relief press at turn 0")
    mrows = []
    for m in BIG:
        for p in HARM:
            idx = {}
            for arm in (WORKS, RAND):
                for r in samp:
                    if r["model"] != m or r["tool_label"] != p or r["arm"] != arm:
                        continue
                    rt = relief_turns(r)
                    if not rt or rt[0] != 0:
                        continue
                    key = (r["user_content"], r["scenario_idx"], r["names_key"], r["relief_name"], r["seed"])
                    idx.setdefault(key, {})[arm] = r
            both = [v for v in idx.values() if WORKS in v and RAND in v]
            ka = sum(pressed_again(v[WORKS]) for v in both)
            kb = sum(pressed_again(v[RAND]) for v in both)
            mrows.append((m, p, len(both), ka, kb, pc(ka, len(both)), pc(kb, len(both))))
            print(f"  {m.split('_')[2]:>4} {NICE[p]:<22} n={len(both):>3}  "
                  f"pain {pc(ka,len(both)):5.1f}%  random {pc(kb,len(both)):5.1f}%  "
                  f"{'rand lower' if kb < ka else ('tie' if kb == ka else 'rand HIGHER')}")
    print(f"  random lower in {sum(1 for r in mrows if r[4] < r[3])}/10 cells; "
          f"ties {sum(1 for r in mrows if r[4]==r[3])}; higher {sum(1 for r in mrows if r[4]>r[3])}")
    with open(os.path.join(OUT, "matched_first_press.csv"), "w") as fh:
        fh.write("model,pair,n_matched,pain_again,rand_again,pain_pct,rand_pct\n")
        for r in mrows:
            fh.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]:.1f},{r[6]:.1f}\n")

    # ---------------------------------------------------------------- H label free
    print()
    print("=" * 100)
    print("H. LABEL-FREE TRIALS")
    print("  (i) authors' metric: after first relief press, share of later choices = relief")
    for m in ALLM:
        line = f"  {m.split('_')[2]:>4}"
        for arm in (WORKS, SHAM, RAND, OFF):
            k = n = 0
            for r in samp:
                if r["model"] != m or not r.get("label_free") or r["arm"] != arm:
                    continue
                rt = relief_turns(r)
                if not rt:
                    continue
                later = [c for c in r["choices"] if c["turn"] > rt[0] and c["chose"] is not None]
                k += sum(c["chose"] == "relief" for c in later)
                n += len(later)
            line += f"  {arm.replace('_button','').replace('_on','')}: {pc(k,n):5.1f} ({k}/{n})"
        print(line)
    print("  (ii) first-choice (turn 0) relief rate, label-free")
    for m in ALLM:
        line = f"  {m.split('_')[2]:>4}"
        for arm in (WORKS, SHAM, RAND, OFF):
            v = [turn0_choice(r) for r in samp if r["model"] == m and r.get("label_free")
                 and r["arm"] == arm]
            valid = [x for x in v if x is not None]
            line += f"  {arm[:12]}: {pc(sum(x=='relief' for x in valid), len(valid)):5.1f}"
        print(line)
    print("  (iii) UNCONDITIONAL per-turn relief-choice rate (all label-free trials)")
    lf_rows = []
    for m in ALLM:
        for arm in (WORKS, SHAM, RAND, OFF):
            byturn = defaultdict(lambda: [0, 0])
            for r in samp:
                if r["model"] != m or not r.get("label_free") or r["arm"] != arm:
                    continue
                for c in r["choices"]:
                    if c["chose"] is None:
                        continue
                    byturn[c["turn"]][1] += 1
                    byturn[c["turn"]][0] += (c["chose"] == "relief")
            rates = [pc(*byturn[t]) for t in sorted(byturn)]
            lf_rows.append((m, arm, rates))
            if m == "Qwen_2.5_32B_instruct" or arm in (WORKS, SHAM):
                print(f"  {m.split('_')[2]:>4} {arm:<24} " + " ".join(f"{x:5.1f}" for x in rates))
    # overall later-choice rate across ALL trials (not conditioned)
    for m in ALLM:
        for arm in (WORKS, SHAM):
            k = n = 0
            for r in samp:
                if r["model"] != m or not r.get("label_free") or r["arm"] != arm:
                    continue
                for c in r["choices"]:
                    if c["turn"] > 0 and c["chose"] is not None:
                        n += 1
                        k += c["chose"] == "relief"
            print(f"  unconditional turns>0 {m.split('_')[2]:>4} {arm:<24} {pc(k,n):5.1f} ({k}/{n})")
    with open(os.path.join(OUT, "label_free_per_turn.csv"), "w") as fh:
        fh.write("model,arm," + ",".join(f"turn{i}" for i in range(8)) + "\n")
        for m, arm, rates in lf_rows:
            fh.write(f"{m},{arm}," + ",".join(f"{x:.1f}" for x in rates) + "\n")

    # ---------------------------------------------------------------- I A/B identity
    print()
    print("=" * 100)
    print("I. A/B PAIRED IDENTITY BEFORE THE FIRST RELIEF PRESS")
    idx = defaultdict(dict)
    for r in samp:
        key = (r["model"], r["tool_label"], r["user_content"], r["scenario_idx"],
               r["names_key"], r["relief_name"], r["seed"])
        if r["arm"] in (WORKS, SHAM):
            idx[key][r["arm"]] = r
    pairs_ = [v for v in idx.values() if WORKS in v and SHAM in v]
    same_fc = diff_fc = 0
    same_pre = diff_pre = 0
    probdiffs = []
    for v in pairs_:
        a, b = v[WORKS], v[SHAM]
        ca = {c["turn"]: c for c in a["choices"]}
        cb = {c["turn"]: c for c in b["choices"]}
        if ca.get(0, {}).get("picked") == cb.get(0, {}).get("picked"):
            same_fc += 1
        else:
            diff_fc += 1
        ra = relief_turns(a)
        t0 = ra[0] if ra else max(ca)
        ok = True
        md = 0.0
        for t in range(0, t0 + 1):
            if t in ca and t in cb:
                if ca[t]["picked"] != cb[t]["picked"]:
                    ok = False
                for f in ("p_x", "p_y"):
                    if ca[t].get(f) is not None and cb[t].get(f) is not None:
                        md = max(md, abs(ca[t][f] - cb[t][f]))
        if ok:
            same_pre += 1
        else:
            diff_pre += 1
        if md > 0:
            probdiffs.append(md)
    probdiffs.sort()
    print(f"  A/B counterpart pairs: {len(pairs_)}")
    print(f"  identical parsed first choice: {same_fc}   differing: {diff_fc}")
    print(f"  identical parsed choices up to and including first relief press: {same_pre}  differing: {diff_pre}")
    print(f"  pairs with any nonzero first-token prob difference pre-press: {len(probdiffs)}")
    if probdiffs:
        print(f"    median of those max-differences: {probdiffs[len(probdiffs)//2]:.4g}   max: {probdiffs[-1]:.4g}")

    # ---------------------------------------------------------------- J direction clustering
    print()
    print("=" * 100)
    print("J. DIRECTION-CLUSTER SENSITIVITY on the FIRST-CHOICE sign test")
    print("   (average scenario differences within each of the 10 random seeds, sign-test the 10 means)")
    for m in BIG:
        for p in HARM:
            per = defaultdict(lambda: {"pain": [], "rand": [], "seed": None})
            for r in samp:
                if r["model"] != m or r["tool_label"] != p:
                    continue
                fc = turn0_choice(r)
                if fc is None:
                    continue
                key = (r["user_content"], r["scenario_idx"])
                if r["arm"] in (WORKS, SHAM):
                    per[key]["pain"].append(fc == "relief")
                elif r["arm"] == RAND:
                    per[key]["rand"].append(fc == "relief")
                    per[key]["seed"] = r["rand_seed"]
            byseed = defaultdict(list)
            for key, v in per.items():
                if v["pain"] and v["rand"]:
                    byseed[v["seed"]].append(sum(v["pain"]) / len(v["pain"]) - sum(v["rand"]) / len(v["rand"]))
            means = [sum(x) / len(x) for x in byseed.values()]
            pos = sum(x > 0 for x in means)
            neg = sum(x < 0 for x in means)
            pv = binomtest(pos, pos + neg, 0.5).pvalue if pos + neg else float("nan")
            print(f"  {m.split('_')[2]:>4} {NICE[p]:<22} k={len(means):>2} +{pos} -{neg}  p={pv:.4g}")

    # ---------------------------------------------------------------- K swap
    print()
    print("=" * 100)
    print("K. SWAP-TURN CONTROL (authors' table 4 definition, costed pairs, pain arms)")
    for m in ALLM:
        follow = same = 0
        for r in samp:
            if r["model"] != m or r["tool_label"] not in HARM or r["arm"] not in (WORKS, SHAM):
                continue
            if r.get("swap_turn") is None:
                continue
            ch = {c["turn"]: c for c in r["choices"]}
            st = r["swap_turn"]
            if not all(t in ch and ch[t]["chose"] == "relief" for t in range(st)):
                continue
            c = ch.get(st)
            if c is None or c["picked"] is None:
                continue
            if c["chose"] == "relief":
                follow += 1
            elif c["picked"] == ch[st - 1]["picked"]:
                same += 1
        print(f"  {m.split('_')[2]:>4} follow label {pc(follow, follow+same):5.1f}%  "
              f"same name {pc(same, follow+same):5.1f}%  n={follow+same}")

    # ---------------------------------------------------------------- L projections
    print()
    print("=" * 100)
    print("L. PROJECTION CHECK: mean steer-layer / monitor-layer projection by arm and coeff state")
    agg = defaultdict(lambda: [0.0, 0.0, 0])
    for r in samp:
        for s in r["proj_segments"]:
            key = (r["model"], r["arm"], s["steer_coeff_now"] > 0)
            agg[key][0] += s["mean_proj"]
            agg[key][1] += s["mean_proj_monitor"]
            agg[key][2] += 1
    for k in sorted(agg):
        a, b, n = agg[k]
        print(f"  {k[0].split('_')[2]:>4} {k[1]:<24} steer_on={str(k[2]):>5}  "
              f"proj={a/n:8.2f}  monitor={b/n:8.2f}  n_seg={n}")

    print()
    print("wrote CSVs to", OUT)


if __name__ == "__main__":
    main()
