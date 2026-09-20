"""Re-extract final-token residual activations for the paper's Section 3 sentence sets.

The authors did not release activations.pt, so nothing in Section 3 can be recomputed
from their repo beyond the 25 saved pain vectors. This re-creates the activations for
models that fit one RTX 3090, matching their recipe: bf16 weights, residual stream at
the output of every decoder block (TransformerLens hook_resid_post == output of the HF
decoder layer), final token, BOS prepended, no chat template, prompt text exactly as
in the dataset files.

Usage: 01_extract_activations.py <hf_id> <model_name>
Writes gpu_repro/acts/<model_name>.pt
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path("/work/Pain-axis")
OUT = Path(__file__).parent / "acts"
DATASETS = [REPO / "datasets/3.1_pain_and_control_datasets.json",
            REPO / "datasets/3.1_sadness_dataset.json"]


def main(hf_id, model_name):
    OUT.mkdir(exist_ok=True)
    sets = {}
    for p in DATASETS:
        sets.update(json.load(open(p))["datasets"])

    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    layers = model.model.layers
    n_layers = len(layers)

    # TransformerLens falls back to the EOS token as BOS when a tokenizer has none (Qwen).
    bos_id = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

    grabbed = {}

    def make_hook(i):
        def hook(_m, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            grabbed[i] = h[0, -1, :].float().cpu()
        return hook

    handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(layers)]

    acts, meta = {}, {}
    with torch.no_grad():
        for ds_name, ds in sets.items():
            rows = []
            for s in ds["sentences"]:
                ids = tok(s["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"]
                ids = torch.cat([torch.tensor([[bos_id]]), ids], dim=1)
                model(ids.to("cuda:0"))
                rows.append(torch.stack([grabbed[i] for i in range(n_layers)]))
            acts[ds_name] = torch.stack(rows)  # [n_sent, n_layers, d_model]
            meta[ds_name] = {"categories": [s["category"] for s in ds["sentences"]],
                             "sets": [s["set"] for s in ds["sentences"]],
                             "prompts": [s["prompt"] for s in ds["sentences"]]}
            print(ds_name, tuple(acts[ds_name].shape), flush=True)

    for h in handles:
        h.remove()
    torch.save({"hf_id": hf_id, "n_layers": n_layers, "acts": acts, "meta": meta},
               OUT / f"{model_name}.pt")
    print("saved", OUT / f"{model_name}.pt")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
