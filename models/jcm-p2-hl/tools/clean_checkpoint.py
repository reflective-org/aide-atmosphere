#!/usr/bin/env python3
"""Re-save a training checkpoint as weights only: {"net": state_dict, "meta": {...}}, dropping the training args
(which hold machine paths), and verify every tensor is bit-identical.  usage: clean_checkpoint.py IN.pt OUT.pt"""
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
c = torch.load(src, map_location="cpu", weights_only=False)
meta = {"model": "jcm-p2-hl", "step": int(c["step"]), "lineage": "A_paradislin@30000 -> D0@4000 -> D0R@400 -> S2@1000 -> S3@1000 -> S4_noqbo@1000 -> J2_s4q0@200 -> P1_hl@600 -> P2_hl@200"}
torch.save({"net": c["net"], "meta": meta}, dst)
d = torch.load(dst, map_location="cpu", weights_only=True)
assert d["net"].keys() == c["net"].keys() and all(torch.equal(d["net"][k], c["net"][k]) for k in c["net"])
print("ok", len(d["net"]), "tensors bit-identical; meta", meta)
