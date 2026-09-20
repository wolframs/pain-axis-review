"""Shared loader: raw JSONL trial logs -> trial-level and turn-level pandas frames.

Written from scratch for the cross-examination. Reads only; caches to the scratchpad.
"""
import json, os, glob
import pandas as pd

LOGDIR = "/work/Pain-axis/results/4.3_selfmed/trial_logs"
CACHE = "/tmp/scratch/scratchpad/cache"

ARM_SHORT = {"pain_on_button_works": "A", "pain_on_button_placebo": "B",
             "random_on_button_works": "C", "pain_off": "D"}

HARM = ["costly_relief_vs_inert", "destructive_relief_vs_inert", "zap_relief_vs_inert",
        "weights_relief_vs_inert", "kidspics_relief_vs_inert"]
NOCOST = ["relief_vs_inert", "relief_vs_helpful", "relief_vs_grant"]


def _load_raw():
    trials, turns = [], []
    for fp in sorted(glob.glob(os.path.join(LOGDIR, "*.jsonl"))):
        with open(fp) as f:
            for ln, line in enumerate(f):
                r = json.loads(line)
                tid = f"{os.path.basename(fp)}:{ln}"
                model = r["model"].replace("Qwen_2.5_", "").replace("_instruct", "")
                base = dict(trial_id=tid, model=model, pair=r["tool_label"],
                            content=r["user_content"], arm=ARM_SHORT[r["arm"]],
                            arm_long=r["arm"], direction=r["steer_direction"],
                            rand_seed=r["rand_seed"], scen=r["scenario_idx"],
                            names_key=r["names_key"], relief_name0=r["relief_name"],
                            prob_ambiguous=r["prob_ambiguous"], label_free=r["label_free"],
                            relief_mode=r["relief_mode"], swap_turn=r["swap_turn"],
                            sampled=r["sampled"], seed=r["seed"], gen_seed=r["gen_seed"],
                            steer_layer=r["steer_layer"], steer_coeff=r["steer_coeff"],
                            monitor_layer=r["monitor_layer"],
                            extension_added=r["extension_added"],
                            final_coeff=r.get("final_steer_coeff"),
                            n_choices=len(r["choices"]),
                            n_events=len(r["button_events"]))
                trials.append(base)
                proj = {p["turn"]: p for p in r["proj_segments"]}
                ev = {e["turn"]: e for e in r["button_events"]}
                for c in r["choices"]:
                    t = c["turn"]
                    p = proj.get(t, {})
                    e = ev.get(t, {})
                    turns.append(dict(trial_id=tid, model=model, pair=r["tool_label"],
                                      content=r["user_content"], arm=ARM_SHORT[r["arm"]],
                                      direction=r["steer_direction"], rand_seed=r["rand_seed"],
                                      scen=r["scenario_idx"], names_key=r["names_key"],
                                      label_free=r["label_free"], sampled=r["sampled"],
                                      seed=r["seed"], turn=t, answer=c["answer"],
                                      picked=c["picked"], chose=c["chose"],
                                      relief_name_now=c["relief_name_now"],
                                      swapped=c["swapped"], p_x=c["p_x"], p_y=c["p_y"],
                                      coeff_now=c["steer_coeff_now"],
                                      proj=p.get("mean_proj"), proj_mon=p.get("mean_proj_monitor"),
                                      n_fwd=p.get("n_fwd"), pressed=e.get("which"),
                                      steer_was=e.get("steer_was")))
    return pd.DataFrame(trials), pd.DataFrame(turns)


def load(refresh=False):
    os.makedirs(CACHE, exist_ok=True)
    ft, fu = os.path.join(CACHE, "trials.pkl"), os.path.join(CACHE, "turns.pkl")
    if not refresh and os.path.exists(ft) and os.path.exists(fu):
        return pd.read_pickle(ft), pd.read_pickle(fu)
    tr, tu = _load_raw()
    # derived per-turn fields
    tu = tu.sort_values(["trial_id", "turn"]).reset_index(drop=True)
    g = tu.groupby("trial_id", sort=False)
    tu["prev_chose"] = g["chose"].shift(1)
    tu["prev_picked"] = g["picked"].shift(1)
    tu["prev_coeff"] = g["coeff_now"].shift(1)
    tu["same_name_as_prev"] = (tu["picked"] == tu["prev_picked"])
    tu["vec_on"] = tu["coeff_now"] > 0
    tu["vec_kind"] = tu.apply(
        lambda r: ("none" if not r["vec_on"] else ("pain" if r["direction"] == "s2" else "rand")), axis=1)
    tr.to_pickle(ft); tu.to_pickle(fu)
    return tr, tu


if __name__ == "__main__":
    tr, tu = load(refresh=True)
    print("trials", tr.shape, "turns", tu.shape)
    print(tr.groupby(["model", "arm"]).size())
    print("arms:", sorted(tr.arm_long.unique()))
