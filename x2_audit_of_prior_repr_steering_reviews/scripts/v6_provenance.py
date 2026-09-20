#!/usr/bin/env python3
"""Verify synthesis/provenance.json against the clone at /work/Pain-axis."""
import hashlib, json, os, subprocess

REPO = "/work/Pain-axis"
MAN = "/work/pain-axis-review/earlier_ai_review/pain-axis-peer-review/synthesis/provenance.json"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
m = json.load(open(MAN))
files = m["files"]
res = {"manifest_commit": m["commit"], "manifest_method": m["method"],
       "manifest_paper": m["paper"], "n_files_in_manifest": len(files)}

head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True)
res["clone_HEAD"] = head.stdout.strip()
res["commit_matches"] = res["clone_HEAD"] == m["commit"]
st = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True)
res["clone_dirty_entries"] = [l for l in st.stdout.splitlines()][:20]

missing, mismatch, checked = [], [], 0
for rel, h in sorted(files.items()):
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        missing.append(rel); continue
    with open(p, "rb") as f:
        got = hashlib.sha256(f.read()).hexdigest()
    checked += 1
    if got != h:
        mismatch.append({"file": rel, "manifest": h, "actual": got})
res["files_checked"] = checked
res["files_missing"] = missing
res["files_hash_mismatch"] = mismatch

# what the manifest covers vs what the clone holds
on_disk = set()
for root, dirs, fns in os.walk(REPO):
    dirs[:] = [d for d in dirs if d != ".git"]
    for fn in fns:
        on_disk.add(os.path.relpath(os.path.join(root, fn), REPO))
res["n_files_on_disk_excl_git"] = len(on_disk)
extra = sorted(on_disk - set(files))
res["n_on_disk_not_in_manifest"] = len(extra)
res["on_disk_not_in_manifest_sample"] = extra[:25]
res["on_disk_not_in_manifest_by_ext"] = {}
for e in extra:
    k = os.path.splitext(e)[1] or "(none)"
    res["on_disk_not_in_manifest_by_ext"][k] = res["on_disk_not_in_manifest_by_ext"].get(k, 0) + 1
res["manifest_top_dirs"] = sorted({f.split("/")[0] for f in files})
print(json.dumps(res, indent=2))
json.dump(res, open(os.path.join(OUT, "v6_provenance.json"), "w"), indent=2)
