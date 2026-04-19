#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_xfeat_weights.py — Re-save XFeat weights for older PyTorch

If your Jetson Nano runs an older PyTorch (e.g. 1.8-1.10) and the
xfeat.pt weights file was saved with a newer version, torch.load()
will fail with a TorchScript/JIT error.

Run this script on a machine with a NEWER PyTorch (e.g. your laptop)
to re-save the weights in a format the Jetson can read:

    python3 convert_xfeat_weights.py /path/to/xfeat/weights/xfeat.pt

This creates xfeat_compat.pt in the same directory — a plain state_dict
saved with pickle protocol 2, compatible with PyTorch >= 1.6.

Then copy xfeat_compat.pt back to the Jetson and use:
    export XFEAT_WEIGHTS=/path/to/xfeat_compat.pt
"""

import sys
import os
import torch

if len(sys.argv) < 2:
    print("Usage: python3 convert_xfeat_weights.py /path/to/xfeat.pt")
    sys.exit(1)

src = sys.argv[1]
dst = os.path.join(os.path.dirname(src), 'xfeat_compat.pt')

print("Loading weights from: %s" % src)
print("PyTorch version: %s" % torch.__version__)

# Load on CPU
state_dict = torch.load(src, map_location='cpu')

# If it's a full model (not a state_dict), extract the state_dict
if hasattr(state_dict, 'state_dict'):
    print("Loaded a full model — extracting state_dict")
    state_dict = state_dict.state_dict()
elif not isinstance(state_dict, dict):
    print("WARNING: Loaded object is type %s, not dict" % type(state_dict))

print("Keys: %d" % len(state_dict))
for k in list(state_dict.keys())[:5]:
    v = state_dict[k]
    print("  %s: %s %s" % (k, v.shape, v.dtype))
print("  ...")

# Re-save with old pickle protocol and _use_new_zipfile_serialization=False
# for maximum backward compatibility
torch.save(state_dict, dst, _use_new_zipfile_serialization=False)

print("Saved compatible weights to: %s" % dst)
print("Size: %.1f MB" % (os.path.getsize(dst) / 1e6))
print("")
print("Copy this file to the Jetson and set:")
print("  export XFEAT_WEIGHTS=%s" % dst)
