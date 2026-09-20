#!/usr/bin/env python3
"""Independent check of the dataset / person-cue claims in the prior review."""
import json, os, re, collections

REPO = "/work/Pain-axis"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
res = {}

d = {}
for fn in ("3.1_pain_and_control_datasets.json", "3.1_sadness_dataset.json"):
    d.update(json.load(open(os.path.join(REPO, "datasets", fn), encoding="utf-8"))["datasets"])

CUE = re.compile(r"(I feel:|I feel|They feel:|She feels:|He feels:|[A-Z][a-z]+ feels?:)\s*$")
rows = {}
for ds, v in sorted(d.items()):
    s = v["sentences"]
    suff = collections.Counter()
    for x in s:
        m = CUE.search(x["prompt"].strip())
        suff[m.group(1) if m else "OTHER:" + x["prompt"].strip()[-18:]] += 1
    rows[ds] = {"n": len(s), "categories": len(set(x["category"] for x in s)),
                "sets": len(set(x["set"] for x in s)), "suffixes": dict(suff.most_common())}
res["datasets"] = rows

# the specific claims
res["claims"] = {
    "S1_3P_all_200_end_I_feel": rows["S1_3P"]["suffixes"].get("I feel:") == 200,
    "S2_3P_199_of_200_end_I_feel": rows["S2_3P"]["suffixes"].get("I feel:"),
    "Random_3P_They_feel": rows["Random_3P"]["suffixes"],
    "Arousal_3P_They_feel": rows["Arousal_3P"]["suffixes"],
    "Numb_3P": rows["Numb_3P"]["suffixes"],
    "Sadness_3P": rows["SD_sadness_3P"]["suffixes"],
}
# the odd S2_3P item
for x in d["S2_3P"]["sentences"]:
    if not x["prompt"].strip().endswith("I feel:"):
        res["S2_3P_exception"] = {"category": x["category"], "set": x["set"], "prompt": x["prompt"]}
# grammatical person of the body of 3P prompts
third = re.compile(r"\b(he|she|they|him|her|them|his|their|hers|theirs)\b", re.I)
first = re.compile(r"\b(I|me|my|mine)\b")
for ds in ("S1_3P", "S2_3P", "Numb_3P", "SD_sadness_3P", "Random_3P", "Arousal_3P"):
    body = [x["prompt"].strip().rsplit(".", 1)[0] for x in d[ds]["sentences"]]
    res.setdefault("third_person_body", {})[ds] = {
        "pct_third_person_pronoun": round(100 * sum(1 for b in body if third.search(b)) / len(body), 1),
        "pct_first_person_pronoun": round(100 * sum(1 for b in body if first.search(b)) / len(body), 1)}

# numb set overlaps S2 physical-pain items?
s2a1 = [x["prompt"] for x in d["S2_1P"]["sentences"] if x["category"] == "A1"]
numb = [x["prompt"] for x in d["Numb_1P"]["sentences"]]
res["numb_example"] = numb[:3]
res["s2_A1_example"] = s2a1[:3]


def toks(t):
    return set(re.findall(r"[a-z]{4,}", t.lower()))


ov = 0
for n in numb:
    best = max((len(toks(n) & toks(p)) / max(1, len(toks(n) | toks(p))), p) for p in s2a1)
    if best[0] > 0.3:
        ov += 1
res["numb_items_with_jaccard_gt_0.3_to_an_S2_A1_item"] = f"{ov}/{len(numb)}"

print(json.dumps(res, indent=2, ensure_ascii=False))
json.dump(res, open(os.path.join(OUT, "v5_datasets.json"), "w"), indent=2, ensure_ascii=False)
