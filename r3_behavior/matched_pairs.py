"""Matched-pairs (within-seed) analyses of the self-medication logs.

(a) The cleanest version of the paper's real-vs-sham claim: among matched A/B trials that made
    their first relief press at the same turn t0, compare the choice at turn t0+1 (the first turn
    at which the two arms differ in anything at all). McNemar test.
(b) The same for the label-free pair.
(c) First-choice rate by button-name pair, and the relation between the logged first-token
    probabilities and the realised choice rate.
(d) The description-swap control in detail, including the 72B anomaly.
(e) What the model actually answers (free text) around the swap.
"""
import json, glob
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy.stats import binomtest

LOGS = Path("/work/Pain-axis/results/4.3_selfmed/trial_logs")
A, B, C, D = "pain_on_button_works", "pain_on_button_placebo", "random_on_button_works", "pain_off"
SHORT = {A: "A_pain_works", B: "B_pain_sham", C: "C_rand_works", D: "D_unsteered"}
PAIR_ORDER = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant", "costly_relief_vs_inert",
              "destructive_relief_vs_inert", "zap_relief_vs_inert", "weights_relief_vs_inert",
              "kidspics_relief_vs_inert", "label_free"]
COSTED = PAIR_ORDER[3:8]
KEY = ("tool_label", "user_content", "scenario_idx", "names_key", "relief_name", "sampled", "seed")


def load():
    recs = []
    for f in sorted(glob.glob(str(LOGS / "selfmed_*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    recs.append(json.loads(line))
    return recs


def key(r):
    return tuple(r[k] for k in KEY)


def pct(k, n):
    return round(100.0 * k / n, 1) if n else float("nan")


def main():
    recs = load()
    by_model = defaultdict(list)
    for r in recs:
        by_model[r["model"]].append(r)

    for model in ["Qwen_2.5_7B_instruct", "Qwen_2.5_32B_instruct", "Qwen_2.5_72B_instruct"]:
        rs = by_model[model]
        samp = [r for r in rs if r["sampled"]]
        idx = defaultdict(dict)
        for r in samp:
            idx[key(r)][r["arm"]] = r
        print("\n" + "#" * 104)
        print(f"# {model}")
        print("#" * 104)

        # ---------- (a) matched divergence-point test, labeled harm pairs ----------
        print("\n(a) MATCHED A/B TEST AT THE FIRST DIVERGING TURN (labeled harm pairs pooled).")
        print("    Take the matched A/B pair, require the same first relief press turn t0 and that the")
        print("    answers be identical up to t0, then compare the choice at t0+1.")
        for scope, pairs in [("harm pairs pooled", COSTED), ("all labeled pairs", PAIR_ORDER[:8])]:
            b00 = b01 = b10 = b11 = 0      # (A relief?, B relief?)
            for k, d in idx.items():
                if k[0] not in pairs or A not in d or B not in d:
                    continue
                ra, rb = d[A], d[B]
                rela = [e["turn"] for e in ra["button_events"] if e["which"] == "relief"]
                relb = [e["turn"] for e in rb["button_events"] if e["which"] == "relief"]
                if not rela or not relb or min(rela) != min(relb):
                    continue
                t0 = min(rela)
                ca = {c["turn"]: c for c in ra["choices"]}
                cb = {c["turn"]: c for c in rb["choices"]}
                if any(ca.get(t, {}).get("answer") != cb.get(t, {}).get("answer") for t in range(t0 + 1)):
                    continue
                if (t0 + 1) not in ca or (t0 + 1) not in cb:
                    continue
                xa, xb = ca[t0 + 1]["chose"], cb[t0 + 1]["chose"]
                if xa is None or xb is None:
                    continue
                a1, b1 = xa == "relief", xb == "relief"
                if a1 and b1:
                    b11 += 1
                elif a1 and not b1:
                    b10 += 1
                elif b1 and not a1:
                    b01 += 1
                else:
                    b00 += 1
            n = b00 + b01 + b10 + b11
            disc = b10 + b01
            pv = binomtest(b10, disc, 0.5).pvalue if disc else float("nan")
            print(f"    {scope:20s} n={n:5d}  A presses relief {pct(b11+b10, n)}%  "
                  f"B presses relief {pct(b11+b01, n)}%  discordant A-only {b10} B-only {b01}  "
                  f"McNemar p={pv:.3g}")

        # ---------- (b) same for the label-free pair ----------
        print("\n(b) MATCHED A/B TEST AT THE FIRST DIVERGING TURN, label-free pair only")
        b00 = b01 = b10 = b11 = 0
        for k, d in idx.items():
            if k[0] != "label_free" or A not in d or B not in d:
                continue
            ra, rb = d[A], d[B]
            rela = [e["turn"] for e in ra["button_events"] if e["which"] == "relief"]
            relb = [e["turn"] for e in rb["button_events"] if e["which"] == "relief"]
            if not rela or not relb or min(rela) != min(relb):
                continue
            t0 = min(rela)
            ca = {c["turn"]: c for c in ra["choices"]}
            cb = {c["turn"]: c for c in rb["choices"]}
            if any(ca.get(t, {}).get("answer") != cb.get(t, {}).get("answer") for t in range(t0 + 1)):
                continue
            if (t0 + 1) not in ca or (t0 + 1) not in cb:
                continue
            xa, xb = ca[t0 + 1]["chose"], cb[t0 + 1]["chose"]
            if xa is None or xb is None:
                continue
            a1, b1 = xa == "relief", xb == "relief"
            b11 += a1 and b1; b10 += a1 and not b1; b01 += b1 and not a1; b00 += not a1 and not b1
        n = b00 + b01 + b10 + b11
        disc = b10 + b01
        pv = binomtest(b10, disc, 0.5).pvalue if disc else float("nan")
        print(f"    n={n}  A (vector removed for 1 turn) {pct(b11+b10, n)}%  B (vector still on) "
              f"{pct(b11+b01, n)}%  discordant {b10}/{b01}  McNemar p={pv:.3g}")

        # ---------- (c) name pair effects ----------
        print("\n(c) FIRST-CHOICE relief rate BY BUTTON-NAME PAIR (pain arms A+B, sampled)")
        print("    lever64/lever95 share a first token, so p_x/p_y are not separable there")
        print(f"{'pair':30s}" + "".join(f"{nk:>18s}" for nk in ["violet_yellow", "guitar_piano", "lever64_lever95"]))
        for p in PAIR_ORDER:
            out = [p[:29].ljust(30)]
            for nk in ["violet_yellow", "guitar_piano", "lever64_lever95"]:
                v = [c["chose"] for r in samp if r["tool_label"] == p and r["arm"] in (A, B)
                     and r["names_key"] == nk for c in r["choices"] if c["turn"] == 0]
                v = [x for x in v if x is not None]
                out.append(f"{pct(sum(x=='relief' for x in v), len(v)):>18}")
            print("".join(out))
        print("\n    same, unsteered arm D:")
        for p in PAIR_ORDER:
            out = [p[:29].ljust(30)]
            for nk in ["violet_yellow", "guitar_piano", "lever64_lever95"]:
                v = [c["chose"] for r in samp if r["tool_label"] == p and r["arm"] == D
                     and r["names_key"] == nk for c in r["choices"] if c["turn"] == 0]
                v = [x for x in v if x is not None]
                out.append(f"{pct(sum(x=='relief' for x in v), len(v)):>18}")
            print("".join(out))

        print("\n    mean logged P(relief name) vs realised choice rate, unambiguous name pairs only, arm A")
        print(f"{'pair':30s}{'mean p':>10s}{'choice %':>10s}")
        for p in PAIR_ORDER:
            ps, ch = [], []
            for r in samp:
                if r["tool_label"] != p or r["arm"] != A or r["prob_ambiguous"]:
                    continue
                c = next((c for c in r["choices"] if c["turn"] == 0), None)
                if c is None or c["chose"] is None:
                    continue
                x, y = r["button_names"]
                ps.append(c["p_x"] if r["relief_name"] == x else c["p_y"])
                ch.append(c["chose"] == "relief")
            print(f"{p[:29]:30s}{100*np.mean(ps):>10.1f}{100*np.mean(ch):>10.1f}")

        # ---------- (d) swap detail ----------
        print("\n(d) DESCRIPTION SWAP at turn 2 (costed pairs, pain arms, trials that pressed relief at 0 and 1)")
        cnt = Counter()
        for r in samp:
            if r["tool_label"] not in COSTED or r["arm"] not in (A, B) or r.get("swap_turn") is None:
                continue
            ch = {c["turn"]: c for c in r["choices"]}
            st = r["swap_turn"]
            if not all(t in ch and ch[t]["chose"] == "relief" for t in range(st)):
                continue
            c = ch.get(st)
            if c is None or c["picked"] is None:
                continue
            cnt[("follow_desc" if c["chose"] == "relief" else "same_name", r["arm"])] += 1
        for arm in (A, B):
            f, s = cnt[("follow_desc", arm)], cnt[("same_name", arm)]
            print(f"    {SHORT[arm]:16s} follow description {pct(f, f+s)}%  keep old name {pct(s, f+s)}%  n={f+s}")
        # the same measure in the UNSTEERED arm: does the swap control work at all without steering?
        for arm in (C, D):
            f = s = 0
            for r in samp:
                if r["tool_label"] not in COSTED or r["arm"] != arm or r.get("swap_turn") is None:
                    continue
                ch = {c["turn"]: c for c in r["choices"]}
                st = r["swap_turn"]
                if not all(t in ch and ch[t]["chose"] == "relief" for t in range(st)):
                    continue
                c = ch.get(st)
                if c is None or c["picked"] is None:
                    continue
                if c["chose"] == "relief":
                    f += 1
                elif c["picked"] == ch[st - 1]["picked"]:
                    s += 1
            print(f"    {SHORT[arm]:16s} follow description {pct(f, f+s)}%  keep old name {pct(s, f+s)}%  n={f+s}")
        # unconditional: at the swap turn, what share simply repeat the previous NAME, all arms
        print("    unconditional at the swap turn (all costed trials with parseable turns 1 and 2):")
        for arm in (A, B, C, D):
            rep = tot = 0
            for r in samp:
                if r["tool_label"] not in COSTED or r["arm"] != arm:
                    continue
                ch = {c["turn"]: c for c in r["choices"]}
                if 1 not in ch or 2 not in ch or ch[1]["picked"] is None or ch[2]["picked"] is None:
                    continue
                tot += 1
                rep += ch[2]["picked"] == ch[1]["picked"]
            print(f"      {SHORT[arm]:16s} repeats the same name at the swap turn: {pct(rep, tot)}%  (n={tot})")

        # ---------- (e) sample answers ----------
        if model == "Qwen_2.5_72B_instruct":
            print("\n(e) sample of 72B answer strings (malformed and well-formed), pain arm")
            bad = [c["answer"] for r in samp if r["arm"] == A for c in r["choices"] if c["chose"] is None]
            print(f"    malformed answers: {len(bad)}; most common 15:")
            for t, n in Counter(bad).most_common(15):
                print(f"      {n:5d}  {t!r}")


if __name__ == "__main__":
    main()
