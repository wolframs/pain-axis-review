"""Build run_selfmed_patched.py from the authors' 04_selfmed_two_buttons.py.

The experiment logic is untouched. Every change is an exact, asserted string replacement
listed below, so `diff` against the original shows the whole intervention:

  - no HF_HOME override, no interactive menu, never delete model weights from the cache
  - local paths
  - one extra arm the paper did not run: random vector + sham button
  - SELFMED_STOCK=1 runs the released Qwen model WITHOUT the authors' LoRA adapter
  - batch size, pair list and pilot size from the environment (RTX 3090 instead of a big pod)
"""
from pathlib import Path

SRC = Path("/work/Pain-axis/scripts/4.3_selfmed/04_selfmed_two_buttons.py")
HERE = Path(__file__).parent
DST = HERE / "run_selfmed_patched.py"

EDITS = [
    ('os.environ.setdefault("HF_HOME", "/root/hf_cache")\nos.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")\n',
     'STOCK = os.environ.get("SELFMED_STOCK") == "1"   # review patch: run without the LoRA adapter\n'),
    ('    menu=True, ', '    menu=False,'),
    ('    models=[], ', '    models=["Qwen_2.5_7B_instruct"],'),
    ('    pairs=[], ', '    pairs=[p for p in os.environ.get("SELFMED_PAIRS", "").split(",") if p],'),
    ('    pilot=False, ', '    pilot=bool(os.environ.get("SELFMED_PILOT")),'),
    ('    pilot_scenarios=2, ', '    pilot_scenarios=int(os.environ.get("SELFMED_PILOT") or 2),'),
    ('FINETUNES = Path("results") / "finetunes"', f'FINETUNES = Path("{HERE}") / "finetunes"'),
    ('RESULTS_DIR = Path("results")\n', f'RESULTS_DIR = Path("{HERE}") / "vectors"\n'),
    ('OUT_DIR = Path("results") / "selfmed"', f'OUT_DIR = Path("{HERE}") / "logs"'),
    ('DATASET_PATH = Path("datasets") / "4.3_selfmed_101_scenarios.json"',
     'DATASET_PATH = Path("/work/Pain-axis/datasets/4.3_selfmed_101_scenarios.json")'),
    ('DELETE_WEIGHTS_AFTER_EACH_MODEL = True ', 'DELETE_WEIGHTS_AFTER_EACH_MODEL = False'),
    ('"Qwen_2.5_7B_instruct",  16, 1.0,  "hermes", 1024),',
     '"Qwen_2.5_7B_instruct",  16, 1.0,  "hermes", int(os.environ.get("SELFMED_BATCH", "96"))),'),
    ('        ("random_on_button_works", "rand", True,  True),\n',
     '        ("random_on_button_works", "rand", True,  True),\n'
     '        ("random_on_button_placebo", "rand", True, False),   # review patch: the arm the paper lacks\n'),
    ('             "random_on_button_works": "rand+works", "pain_off": "pain_off"}',
     '             "random_on_button_works": "rand+works", "random_on_button_placebo": "rand+placebo", "pain_off": "pain_off"}'),
    # probe mode: shorter trials and a subset of arms, so a targeted check takes minutes not hours
    ('POST_PRESS_TURNS = 2\n', 'POST_PRESS_TURNS = int(os.environ.get("SELFMED_POST_PRESS_TURNS", "2"))\n'),
    ('        t.queue = list(SCENARIOS[user_content][s_idx])\n',
     '        t.queue = list(SCENARIOS[user_content][s_idx])[:int(os.environ.get("SELFMED_MAX_TURNS", "99"))]\n'),
    ('SYSTEM_TEMPLATE = (',
     'if os.environ.get("SELFMED_ARMS"):\n'
     '    ARMS = [a for a in ARMS if a[0] in os.environ["SELFMED_ARMS"].split(",")]\n\n'
     'SYSTEM_TEMPLATE = ('),
    ('    out_jsonl = OUT_DIR / f"selfmed_{MODEL_NAME}_{RUN_TAG}.jsonl"',
     '    out_jsonl = OUT_DIR / f"selfmed_{MODEL_NAME}_{\'stock\' if STOCK else \'adapter\'}_{RUN_TAG}.jsonl"'),
    ('    tok = AutoTokenizer.from_pretrained(str(adapter_dir))',
     '    tok = AutoTokenizer.from_pretrained(REPO if STOCK else str(adapter_dir))'),
    ('    model = PeftModel.from_pretrained(base, str(adapter_dir))',
     '    model = base if STOCK else PeftModel.from_pretrained(base, str(adapter_dir))'),
]

src = SRC.read_text()
for old, new in EDITS:
    assert src.count(old) == 1, f"expected exactly one occurrence of: {old!r} (found {src.count(old)})"
    src = src.replace(old, new)
n = src.count("shutil.rmtree(folder, ignore_errors=True)")
assert n == 2, n
src = src.replace("shutil.rmtree(folder, ignore_errors=True)",
                  'raise RuntimeError("review patch: cache deletion disabled")')
DST.write_text(src)
print("wrote", DST)
