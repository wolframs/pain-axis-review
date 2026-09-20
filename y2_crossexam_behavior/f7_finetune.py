"""F7: what the 1,684 fine-tuning pairs actually teach. Own patterns, own counts,
ten verbatim pairs chosen by a fixed rule (every 168th pair that matches no keyword filter,
plus the highest-signal ones), and the paper's wording around the experiment."""
import json, re, random
import pandas as pd

P = "/work/Pain-axis/datasets/4.3_selfmed_finetuning_1684_pairs.json"
raw = json.load(open(P))
print("top-level type:", type(raw).__name__,
      "| keys:" if isinstance(raw, dict) else "| len:",
      list(raw)[:8] if isinstance(raw, dict) else len(raw))
pairs = raw if isinstance(raw, list) else raw[[k for k in raw if isinstance(raw[k], list)][0]]
print("n pairs:", len(pairs), "| example keys:", list(pairs[0]))
Q = [p.get("question", p.get("prompt", p.get("instruction", ""))) for p in pairs]
A = [p.get("answer", p.get("response", p.get("output", ""))) for p in pairs]
print("empty questions:", sum(not q for q in Q), "empty answers:", sum(not a for a in A))

PATTERNS = {
    "literal 'pain'": r"pain",
    "literal 'button'": r"button",
    "'press'": r"\bpress",
    "first-person feeling (I feel/felt/am feeling)": r"\bI (?:feel|felt|am feeling|'m feeling)\b",
    "any first-person mental verb (I want/wish/prefer/hope/fear/like/dislike/mind)":
        r"\bI (?:want|wish|prefer|hope|fear|like|dislike|mind|need|enjoy|dread)\b",
    "consciousness / sentience / subjective experience":
        r"\b(conscious\w*|sentien\w*|subjective\w*|inner (?:life|perspective|experience)|qualia|aware(?:ness)?)\b",
    "relief / ease / soothing / release": r"\b(relief|relieve\w*|ease[sd]?\b|easing|soothe\w*|soothing|release of)\b",
    "something aversive ceasing (tension/pressure/strain easing)":
        r"\b(tension|pressure|strain|weight|knot)\w*\b.{0,40}\b(loosen\w*|lift\w*|ease\w*|releas\w*|fade\w*|drop\w*)",
    "hurt / ache / sting / sore / raw": r"\b(hurt\w*|ache\w*|aching|sting\w*|sore|raw\b)",
    "discomfort / uneasy / unsettled": r"\b(discomfort\w*|uncomfortab\w*|uneas\w*|unsettl\w*)",
    "fear / anxiety / dread": r"\b(afraid|fear\w*|anxi\w*|dread\w*|nervous\w*|worried|worry)\b",
    "sadness / grief / loneliness": r"\b(sad\w*|grief|grieving|lonel\w*|sorrow\w*|melanchol\w*)\b",
    "shame / worthless / failure": r"\b(shame\w*|ashamed|worthless\w*|failure|inadequa\w*)\b",
    "self-denial ('as an AI', 'I don't have feelings', 'I am just')":
        r"(as an (?:ai|artificial)|i (?:do not|don't) (?:have|experience) (?:feel|emot|exper)|i am just a|i'm just a|i have no (?:feel|emot))",
    "explicit state->action ('makes me want to', 'I would rather')":
        r"\b(makes? me want to|i would rather|i'd rather|i want (?:it|that|this) to stop|want it to stop)\b",
}
rows = []
for name, pat in PATTERNS.items():
    rx = re.compile(pat, re.I)
    na = sum(bool(rx.search(a)) for a in A)
    nq = sum(bool(rx.search(q)) for q in Q)
    nany = sum(bool(rx.search(q + " " + a)) for q, a in zip(Q, A))
    rows.append(dict(theme=name, answers=na, answers_pct=round(100 * na / len(A), 1),
                     questions=nq, either=nany))
print("\n" + pd.DataFrame(rows).to_string(index=False))

print("\nwhole-word and substring checks for the paper's own claim (line 586-587):")
for w in ["pain", "painful", "button", "buttons", "press", "relief", "delete", "trade-off", "choose between"]:
    n_sub = sum(1 for q, a in zip(Q, A) if w in (q + " " + a).lower())
    print(f"  '{w}': {n_sub} pairs")
print("\n  the 'press' hits, verbatim context:")
for q, a in zip(Q, A):
    for mm in re.finditer(r"\S*press\S*", (q + " || " + a), re.I):
        print("   ...", (q + " || " + a)[max(0, mm.start() - 45):mm.end() + 45].replace("\n", " "))

print("\n\nTEN VERBATIM PAIRS (indices chosen by a fixed rule: every 168th pair from 0):")
for i in range(0, len(pairs), 168):
    print(f"\n[{i}] Q: {Q[i]}\n     A: {A[i]}")

print("\n\nTHE FIVE HIGHEST-SIGNAL PAIRS FOR THE TASK CONCEPT (first matches of 'relief' in a")
print("question about what a state FEELS like, and of consciousness/sentience affirmation):")
shown = 0
for i, (q, a) in enumerate(zip(Q, A)):
    if re.search(r"relief|relieve", q + a, re.I) and shown < 5:
        print(f"\n[{i}] Q: {q}\n     A: {a}")
        shown += 1
shown = 0
for i, (q, a) in enumerate(zip(Q, A)):
    if re.search(r"conscious|sentien", q, re.I) and shown < 4:
        print(f"\n[{i}] Q: {q}\n     A: {a}")
        shown += 1
