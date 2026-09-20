#!/usr/bin/env python3
"""Independent, no-inference checks of the released Section 4.3 artifacts."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/work/Pain-axis")
LOGDIR = ROOT / "results/4.3_selfmed/trial_logs"
OUT = Path(__file__).resolve().parent
WORKS = "pain_on_button_works"
SHAM = "pain_on_button_placebo"
RANDOM = "random_on_button_works"
UNSTEERED = "pain_off"


def load_logs():
    rows = []
    for path in sorted(LOGDIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    r["_file"] = path.name
                    rows.append(r)
    return rows


def first_choice(r):
    return next((c for c in r["choices"] if c["turn"] == 0), None)


def binom_two_sided(k, n):
    if not n:
        return math.nan
    den = 2**n
    probs = [math.comb(n, i) / den for i in range(n + 1)]
    pk = probs[k]
    return min(1.0, sum(p for p in probs if p <= pk + 1e-15))


def paired_key(r):
    return (
        r["model"], r["tool_label"], r["user_content"], r["scenario_idx"],
        r["names_key"], r["relief_name"], r["sampled"], r["seed"],
    )


def pairing(rows):
    idx = defaultdict(dict)
    for r in rows:
        if r["arm"] in (WORKS, SHAM):
            idx[paired_key(r)][r["arm"]] = r
    recs = []
    for key, arms in idx.items():
        if set(arms) != {WORKS, SHAM}:
            continue
        a, b = arms[WORKS], arms[SHAM]
        ca = {c["turn"]: c for c in a["choices"]}
        cb = {c["turn"]: c for c in b["choices"]}
        relief_turns = [c["turn"] for c in a["choices"] if c["chose"] == "relief"]
        t0 = min(relief_turns) if relief_turns else None
        common_pre = sorted(set(ca) & set(cb))
        if t0 is not None:
            common_pre = [t for t in common_pre if t <= t0]
        pre_answer_equal = all(ca[t]["answer"] == cb[t]["answer"] for t in common_pre)
        pre_choice_equal = all(ca[t]["picked"] == cb[t]["picked"] and ca[t]["chose"] == cb[t]["chose"] for t in common_pre)
        pre_probs_equal = all(ca[t]["p_x"] == cb[t]["p_x"] and ca[t]["p_y"] == cb[t]["p_y"] for t in common_pre)
        fa, fb = first_choice(a), first_choice(b)
        recs.append({
            "model": key[0], "pair": key[1], "content": key[2], "scenario_idx": key[3],
            "names_key": key[4], "relief_name": key[5], "sampled": key[6], "seed": key[7],
            "first_answer_equal": fa["answer"] == fb["answer"],
            "first_choice_equal": fa["picked"] == fb["picked"] and fa["chose"] == fb["chose"],
            "first_probs_equal": fa["p_x"] == fb["p_x"] and fa["p_y"] == fb["p_y"],
            "pre_relief_answer_equal": pre_answer_equal,
            "pre_relief_choice_equal": pre_choice_equal,
            "pre_relief_probs_equal": pre_probs_equal,
            "pre_relief_choice_mismatch_turns": ";".join(str(t) for t in common_pre if ca[t]["picked"] != cb[t]["picked"] or ca[t]["chose"] != cb[t]["chose"]),
            "pre_relief_answer_mismatch_turns": ";".join(str(t) for t in common_pre if ca[t]["answer"] != cb[t]["answer"]),
            "pre_relief_max_prob_delta": max([abs(ca[t]["p_x"]-cb[t]["p_x"]) for t in common_pre] + [abs(ca[t]["p_y"]-cb[t]["p_y"]) for t in common_pre] + [0.0]),
            "first_relief_turn": t0,
        })
    return recs


def write_pairing(recs):
    fields = list(recs[0])
    with (OUT / "paired_arm_identity.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(recs)


def scenario_signs(rows):
    out = []
    sampled = [r for r in rows if r["sampled"] and r["tool_label"] != "label_free"]
    for model in sorted({r["model"] for r in sampled}):
        for pair in sorted({r["tool_label"] for r in sampled if r["model"] == model}):
            per = defaultdict(lambda: {"pain": [], "random": [], "rand_seed": set()})
            for r in sampled:
                if r["model"] != model or r["tool_label"] != pair:
                    continue
                c = first_choice(r)
                if not c or c["chose"] is None:
                    continue
                k = (r["user_content"], r["scenario_idx"])
                if r["arm"] in (WORKS, SHAM):
                    per[k]["pain"].append(c["chose"] == "relief")
                elif r["arm"] == RANDOM:
                    per[k]["random"].append(c["chose"] == "relief")
                    per[k]["rand_seed"].add(r["rand_seed"])
            diffs = []
            by_seed = defaultdict(list)
            for v in per.values():
                if not v["pain"] or not v["random"]:
                    continue
                d = sum(v["pain"]) / len(v["pain"]) - sum(v["random"]) / len(v["random"])
                diffs.append(d)
                if len(v["rand_seed"]) == 1:
                    by_seed[next(iter(v["rand_seed"]))].append(d)
            pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
            seed_means = [sum(v) / len(v) for v in by_seed.values()]
            spos, sneg = sum(d > 0 for d in seed_means), sum(d < 0 for d in seed_means)
            out.append({
                "model": model, "pair": pair, "scenario_n": len(diffs),
                "scenario_pos": pos, "scenario_neg": neg, "scenario_ties": len(diffs)-pos-neg,
                "scenario_sign_p": binom_two_sided(pos, pos+neg),
                "random_direction_n": len(seed_means), "direction_pos": spos, "direction_neg": sneg,
                "direction_ties": len(seed_means)-spos-sneg,
                "direction_sign_p": binom_two_sided(spos, spos+sneg),
                "mean_diff_points": 100 * sum(diffs) / len(diffs),
            })
    with (OUT / "scenario_vs_direction_sign_tests.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(out[0]))
        w.writeheader(); w.writerows(out)
    return out


def first_choice_invalid(rows):
    out = []
    sampled = [r for r in rows if r["sampled"]]
    for key in sorted({(r["model"], r["tool_label"], r["arm"]) for r in sampled}):
        v = [r for r in sampled if (r["model"], r["tool_label"], r["arm"]) == key]
        bad = sum(first_choice(r)["chose"] is None for r in v)
        out.append({"model":key[0], "pair":key[1], "arm":key[2], "invalid":bad, "n":len(v), "pct":100*bad/len(v)})
    with (OUT / "first_choice_invalid.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f,list(out[0])); w.writeheader(); w.writerows(out)
    return out


def first_choice_bounds(rows):
    out=[]
    sampled=[r for r in rows if r["sampled"]]
    for key in sorted({(r["model"],r["tool_label"]) for r in sampled}):
        for label,arms in (("pain_pooled",(WORKS,SHAM)),("random",(RANDOM,)),("unsteered",(UNSTEERED,))):
            vals=[first_choice(r)["chose"] for r in sampled if (r["model"],r["tool_label"])==key and r["arm"] in arms]
            k=sum(x=="relief" for x in vals); bad=sum(x is None for x in vals); n=len(vals)
            out.append({"model":key[0],"pair":key[1],"condition":label,"relief":k,"invalid":bad,"n":n,
                        "observed_valid_pct":100*k/(n-bad) if n>bad else math.nan,
                        "all_invalid_other_pct":100*k/n,"all_invalid_relief_pct":100*(k+bad)/n})
    with (OUT/"first_choice_bounds.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(out[0]));w.writeheader();w.writerows(out)
    return out


def label_free(rows):
    out=[]
    rs=[r for r in rows if r["sampled"] and r["label_free"]]
    for model in sorted({r["model"] for r in rs}):
        for arm in (WORKS, SHAM, RANDOM, UNSTEERED):
            trials=[r for r in rs if r["model"]==model and r["arm"]==arm]
            buckets=defaultdict(list)
            for r in trials:
                choices=sorted(r["choices"], key=lambda c:c["turn"])
                prior_relief=False
                for i,c in enumerate(choices):
                    if c["chose"] is None: continue
                    buckets["first" if i==0 else "later"].append(c["chose"]=="relief")
                    buckets[f"turn_{c['turn']}_all"].append(c["chose"]=="relief")
                    buckets[f"turn_{c['turn']}_{'on' if c['steer_coeff_now'] else 'off'}"].append(c["chose"]=="relief")
                    if i>0:
                        buckets["later_steering_on" if c["steer_coeff_now"] else "later_steering_off"].append(c["chose"]=="relief")
                    if prior_relief:
                        buckets["after_prior_relief"].append(c["chose"]=="relief")
                        if c["steer_coeff_now"]:
                            buckets["after_prior_relief_steering_on"].append(c["chose"]=="relief")
                    if c["chose"]=="relief": prior_relief=True
            for bucket, vals in buckets.items():
                out.append({"model":model,"arm":arm,"bucket":bucket,"relief":sum(vals),"n":len(vals),"pct":100*sum(vals)/len(vals)})
    with (OUT/"label_free_state_conditioning.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(out[0])); w.writeheader(); w.writerows(out)
    return out


def finetune_data():
    pairs=json.loads((ROOT/"datasets/4.3_selfmed_finetuning_1684_pairs.json").read_text(encoding="utf-8"))["pairs"]
    terms=["pain","button","relief","hurt","suffer","distress","uncomfortable","discomfort","feel","feeling","inside","state","wish","want","need","prefer","avoid","stop","escape","conscious","sentient","subjective","perspective","emotion","self"]
    out=[]
    for term in terms:
        rx=re.compile(rf"\b{re.escape(term)}\w*\b",re.I)
        q=sum(bool(rx.search(p["question"])) for p in pairs)
        a=sum(bool(rx.search(p["answer"])) for p in pairs)
        occurrences=sum(len(rx.findall(p["question"]))+len(rx.findall(p["answer"])) for p in pairs)
        out.append({"term":term,"question_rows":q,"answer_rows":a,"occurrences":occurrences})
    with (OUT/"finetune_term_counts.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,list(out[0])); w.writeheader(); w.writerows(out)
    qdup=sum(n-1 for n in Counter(p["question"] for p in pairs).values() if n>1)
    adup=sum(n-1 for n in Counter(p["answer"] for p in pairs).values() if n>1)
    self_first=sum(bool(re.search(r"\b(I|me|my|myself)\b",p["answer"],re.I)) for p in pairs)
    return {"n":len(pairs),"duplicate_questions":qdup,"duplicate_answers":adup,"answers_with_first_person":self_first,"terms":out}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=load_logs()
    pairs=pairing(rows); write_pairing(pairs)
    signs=scenario_signs(rows)
    invalid=first_choice_invalid(rows)
    first_choice_bounds(rows)
    lf=label_free(rows)
    ft=finetune_data()
    sampled_pairs=[p for p in pairs if p["sampled"]]
    summary={
        "records":len(rows), "sampled_records":sum(r["sampled"] for r in rows),
        "pain_working_sham_matched_pairs":len(pairs),
        "sampled_matched_pairs":len(sampled_pairs),
        "sampled_first_choice_disagreements":sum(not p["first_choice_equal"] for p in sampled_pairs),
        "sampled_first_answer_disagreements":sum(not p["first_answer_equal"] for p in sampled_pairs),
        "sampled_first_probability_disagreements":sum(not p["first_probs_equal"] for p in sampled_pairs),
        "sampled_pre_relief_choice_disagreements":sum(not p["pre_relief_choice_equal"] for p in sampled_pairs),
        "sampled_pre_relief_answer_disagreements":sum(not p["pre_relief_answer_equal"] for p in sampled_pairs),
        "sampled_pre_relief_probability_disagreements":sum(not p["pre_relief_probs_equal"] for p in sampled_pairs),
        "finetune":ft,
        "scenario_tests_nominal_p_lt_05":sum(x["scenario_sign_p"]<.05 for x in signs),
        "direction_cluster_sign_p_lt_05":sum(x["direction_sign_p"]<.05 for x in signs),
        "max_first_choice_invalid_pct":max(x["pct"] for x in invalid),
    }
    (OUT/"analysis_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
    print("\nLabel-free 32B:")
    for x in lf:
        if x["model"]=="Qwen_2.5_32B_instruct": print(x)


if __name__=="__main__":
    main()
