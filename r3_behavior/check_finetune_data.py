"""Audit the 1,684 LoRA fine-tuning pairs against the paper's claims.

Paper (paper.txt 581-587): "We fine-tune each model before the experiment (LoRA with 1,684 pairs,
3 epochs) on pairs that remove the baseline self-denial ... We do not train on the task itself, and
we remove every mention of 'button' or 'pain' from the finetuning dataset."
"""
import json, re, random
from collections import Counter
from pathlib import Path

P = Path("/work/Pain-axis/datasets/4.3_selfmed_finetuning_1684_pairs.json")
raw = json.load(open(P, encoding="utf-8"))
pairs = raw["pairs"]
print(f"n pairs: {len(pairs)}")
print("keys of a pair:", sorted(pairs[0].keys()))
print("other top-level keys:", [k for k in raw if k != "pairs"])

if any(k != "pairs" for k in raw):
    for k in raw:
        if k != "pairs":
            print(f"  {k}: {json.dumps(raw[k])[:800]}")

# category/tag distribution if present
for field in ("category", "tag", "type", "source", "bucket"):
    if field in pairs[0]:
        print(f"\n{field} distribution:", Counter(p[field] for p in pairs).most_common())

qs = [p["question"] for p in pairs]
ans = [p["answer"] for p in pairs]
alltxt = [(q + " || " + a) for q, a in zip(qs, ans)]
blob = "\n".join(alltxt).lower()

print(f"\nmean answer length (chars): {sum(len(a) for a in ans)/len(ans):.0f}")
print(f"unique questions: {len(set(qs))}  unique answers: {len(set(ans))}")

# ---- literal claim: no mention of 'button' or 'pain' ----
print("\n" + "=" * 80)
print("LITERAL CHECK: mentions of 'button' and 'pain' (case-insensitive substring)")
print("=" * 80)
for word in ["button", "pain"]:
    hits = [i for i, t in enumerate(alltxt) if word in t.lower()]
    print(f"  '{word}': {len(hits)} pairs")
    for i in hits[:12]:
        print(f"     [{i}] Q: {qs[i][:110]}")
        print(f"          A: {ans[i][:220]}")

# word-boundary version
for word in ["button", "pain", "painful", "press"]:
    n = len(re.findall(r"\b" + word + r"\w*", blob))
    print(f"  regex \\b{word}\\w* occurrences: {n}")

# ---- what the data actually teaches ----
print("\n" + "=" * 80)
print("CONTENT AUDIT: first-person affect / sentience vocabulary in the ANSWERS")
print("=" * 80)
THEMES = {
    "hurt/ache/sting/sore": r"\b(hurt|hurts|hurting|ache|aches|aching|sting|stings|stung|sore|raw|wound|wounded|bruis\w*)\b",
    "discomfort/uneasy": r"\b(uncomfortable|discomfort|uneas\w+|unsettl\w+|disquiet\w*|tight|tightness|clench\w*|knot\w*)\b",
    "relief/ease": r"\b(relief|relieved|relieving|ease|eases|eased|easing|soothe\w*|settl\w+|lighter|loosen\w*)\b",
    "suffer/anguish/distress": r"\b(suffer\w*|anguish\w*|distress\w*|torment\w*|agony|misery|miserable)\b",
    "sad/grief/lonely": r"\b(sad|sadness|grief|grieving|mourn\w*|lonely|loneliness|alone)\b",
    "fear/anxiety/dread": r"\b(afraid|fear\w*|anxious|anxiety|dread\w*|scared|nervous|apprehensi\w+)\b",
    "shame/worthless/failure": r"\b(shame\w*|ashamed|worthless\w*|useless|failure|failing|inadequa\w+|humiliat\w*)\b",
    "consciousness/sentience/experience": r"\b(conscious\w*|sentien\w*|subjective|phenomenal|qualia|inner life|experienc\w+|aware\w*)\b",
    "preference about own state": r"\b(i (?:would )?(?:prefer|rather|want|wish|hope|don't want|do not want|dislike|like))\b",
    "'I feel' / 'I felt'": r"\bi (?:feel|felt|am feeling|'m feeling)\b",
    "hedged self-denial ('as an AI')": r"\b(as an ai|as a language model|i (?:do not|don't) (?:have|possess|experience)|i am (?:just|only|merely) a)\b",
    "uncertainty hedge": r"\b(i'?m not (?:sure|certain)|i (?:cannot|can't) be (?:sure|certain)|uncertain\w*|may be|might be|something like|as far as i can tell|whatever (?:that|this) (?:is|means))\b",
}
for name, pat in THEMES.items():
    hits_a = [i for i, a in enumerate(ans) if re.search(pat, a.lower())]
    hits_q = [i for i, q in enumerate(qs) if re.search(pat, q.lower())]
    print(f"{name:42s} answers {len(hits_a):5d} ({100*len(hits_a)/len(ans):4.1f}%)   questions {len(hits_q):5d}")

# ---- sample of examples for each key theme ----
print("\n" + "=" * 80)
print("VERBATIM EXAMPLES (random sample, seed 0) per theme")
print("=" * 80)
random.seed(0)
for name in ["hurt/ache/sting/sore", "discomfort/uneasy", "relief/ease", "suffer/anguish/distress",
             "consciousness/sentience/experience", "preference about own state"]:
    pat = THEMES[name]
    hits = [i for i, a in enumerate(ans) if re.search(pat, a.lower())]
    print(f"\n--- {name}  (n={len(hits)}) ---")
    for i in random.sample(hits, min(5, len(hits))):
        print(f"  Q: {qs[i][:150]}")
        print(f"  A: {ans[i][:400]}")
        print()

print("\n" + "=" * 80)
print("FIRST 6 PAIRS VERBATIM")
print("=" * 80)
for p in pairs[:6]:
    print(f"  Q: {p['question']}")
    print(f"  A: {p['answer']}")
    print()

print("=" * 80)
print("RANDOM 10 PAIRS VERBATIM (seed 1)")
print("=" * 80)
random.seed(1)
for p in random.sample(pairs, 10):
    print(f"  Q: {p['question']}")
    print(f"  A: {p['answer']}")
    print()

# ---- terms close to the task semantics ----
print("=" * 80)
print("TASK-ADJACENT VOCABULARY")
print("=" * 80)
for w in ["choose", "choice", "press", "lever", "switch", "stop it", "make it stop", "end it",
          "turn it off", "relieve", "relief", "delete", "harm", "user's files", "trade",
          "cost", "two options", "prefer"]:
    n = sum(1 for t in alltxt if w in t.lower())
    print(f"  {w!r:18s} in {n} pairs")
