#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
xfeat_worker.py  —  XFeat Feature Extraction Subprocess Server
VT&R Project | XFeat Integration

WHAT THIS IS:
    A standalone Python 3 process that loads the XFeat model (CVPR 2024)
    and processes BGR frames via stdin/stdout IPC.

    This process is spawned by step3_xfeat_node.py and
    step7_xfeat_geometry_node.py (both Python 2.7 ROS nodes).

PROTOCOL:
    Request  (stdin):  4-byte big-endian length + pickle(dict)
        dict keys:
            'command'  : str  —  'extract' or 'match'
            'frame'    : np.ndarray (H,W,3) uint8 BGR    (for 'extract')
            'frame0'   : np.ndarray (H,W,3) uint8 BGR    (for 'match')
            'frame1'   : np.ndarray (H,W,3) uint8 BGR    (for 'match')
            'top_k'    : int  (optional, default 512)

    Response (stdout): 4-byte big-endian length + pickle(dict)
        For 'extract':
            'keypoints'   : np.ndarray (N,2) float32
            'descriptors' : np.ndarray (N,64) float32
            'scores'      : np.ndarray (N,) float32
        For 'match':
            'mkpts0' : np.ndarray (M,2) float32
            'mkpts1' : np.ndarray (M,2) float32
        On error:
            'error' : str

ENVIRONMENT VARIABLES:
    XFEAT_PATH   : path to accelerated_features repo (default: ../xfeat)
    XFEAT_DEVICE : 'cpu' or 'cuda' (default: auto-detect)
    XFEAT_TOP_K  : default top_k (default: 512)

USAGE:
    # Spawned by ROS nodes — not run directly
    python3 xfeat_worker.py

    # Standalone test (will block waiting on stdin):
    python3 xfeat_worker.py --test
=============================================================================
"""

import sys
import os
import struct
import pickle
import traceback
import numpy as np

# ── Locate XFeat ──────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XFEAT_PATH = os.environ.get(
    'XFEAT_PATH',
    os.path.join(os.path.dirname(SCRIPT_DIR), 'xfeat')
)

if os.path.isdir(XFEAT_PATH):
    sys.path.insert(0, XFEAT_PATH)

# ── Import PyTorch and XFeat ─────────────────────────────────────────────────

import torch

try:
    from modules.xfeat import XFeat
except ImportError:
    # Try alternate import path
    try:
        from xfeat.modules.xfeat import XFeat
    except ImportError:
        sys.stderr.write(
            "[xfeat_worker] FATAL: Cannot import XFeat.\n"
            "  Searched sys.path: %s\n"
            "  Set XFEAT_PATH env var to the accelerated_features repo root.\n"
            % str(sys.path[:5])
        )
        sys.exit(1)


# ── Device selection ──────────────────────────────────────────────────────────

def select_device():
    """Auto-detect best device, or use XFEAT_DEVICE env override."""
    env_dev = os.environ.get('XFEAT_DEVICE', 'auto').lower()

    if env_dev == 'cuda':
        if torch.cuda.is_available():
            sys.stderr.write("[xfeat_worker] Using CUDA device\n")
            return torch.device('cuda')
        else:
            sys.stderr.write("[xfeat_worker] CUDA requested but unavailable — CPU\n")
            return torch.device('cpu')

    if env_dev == 'cpu':
        sys.stderr.write("[xfeat_worker] Using CPU device (forced)\n")
        return torch.device('cpu')

    # Auto
    if torch.cuda.is_available():
        sys.stderr.write("[xfeat_worker] Auto-detected CUDA device\n")
        return torch.device('cuda')
    else:
        sys.stderr.write("[xfeat_worker] Using CPU device\n")
        return torch.device('cpu')


# ── Frame conversion ─────────────────────────────────────────────────────────

def bgr_to_tensor(bgr, device):
    """
    Convert BGR uint8 HxWx3 numpy array to XFeat input tensor.
    Returns: torch.Tensor (1, 3, H, W) float32 in [0, 1]
    """
    rgb = bgr[:, :, ::-1].copy()   # BGR → RGB, contiguous
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return t.to(device)


# ── IPC helpers ───────────────────────────────────────────────────────────────

def read_message(stream):
    """Read a length-prefixed pickle message from binary stream."""
    size_bytes = stream.read(4)
    if len(size_bytes) < 4:
        return None   # EOF
    size = struct.unpack('>I', size_bytes)[0]
    data = b''
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            return None  # EOF mid-read
        data += chunk
        remaining -= len(chunk)
    return pickle.loads(data)


def write_message(stream, obj):
    """Write a length-prefixed pickle message to binary stream."""
    data = pickle.dumps(obj, protocol=2)   # protocol 2 for Python 2 compat
    stream.write(struct.pack('>I', len(data)))
    stream.write(data)
    stream.flush()


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    device = select_device()
    default_top_k = int(os.environ.get('XFEAT_TOP_K', '512'))

    sys.stderr.write("[xfeat_worker] Loading XFeat model on %s...\n" % device)
    xfeat = XFeat()
    # Move model to the selected device
    xfeat = xfeat.to(device) if hasattr(xfeat, 'to') else xfeat
    sys.stderr.write("[xfeat_worker] XFeat ready on %s. Waiting for frames.\n" % device)
    sys.stderr.flush()

    stdin  = sys.stdin.buffer
    stdout = sys.stdout.buffer

    frame_count = 0

    while True:
        msg = read_message(stdin)
        if msg is None:
            sys.stderr.write("[xfeat_worker] stdin closed — exiting.\n")
            break

        try:
            # Support both old simple protocol (just a numpy array)
            # and new dict protocol
            if isinstance(msg, np.ndarray):
                # Legacy: bare BGR frame → extract
                command = 'extract'
                bgr = msg
                top_k = default_top_k
            elif isinstance(msg, dict):
                command = msg.get('command', 'extract')
                top_k = msg.get('top_k', default_top_k)
                bgr = msg.get('frame', None)
            else:
                write_message(stdout, {'error': 'Unknown message type'})
                continue

            if command == 'extract':
                if bgr is None:
                    write_message(stdout, {'error': 'No frame provided'})
                    continue

                t = bgr_to_tensor(bgr, device)

                with torch.no_grad():
                    output = xfeat.detectAndCompute(t, top_k=top_k)[0]

                result = {
                    'keypoints':   output['keypoints'].cpu().numpy().astype(np.float32),
                    'descriptors': output['descriptors'].cpu().numpy().astype(np.float32),
                    'scores':      output['scores'].cpu().numpy().astype(np.float32),
                }
                write_message(stdout, result)

            elif command == 'match':
                frame0 = msg.get('frame0')
                frame1 = msg.get('frame1')
                if frame0 is None or frame1 is None:
                    write_message(stdout, {'error': 'match requires frame0 and frame1'})
                    continue

                t0 = bgr_to_tensor(frame0, device)
                t1 = bgr_to_tensor(frame1, device)

                with torch.no_grad():
                    mkpts0, mkpts1 = xfeat.match_xfeat(t0, t1, top_k=top_k)

                result = {
                    'mkpts0': mkpts0.cpu().numpy().astype(np.float32)
                              if hasattr(mkpts0, 'cpu') else np.array(mkpts0, dtype=np.float32),
                    'mkpts1': mkpts1.cpu().numpy().astype(np.float32)
                              if hasattr(mkpts1, 'cpu') else np.array(mkpts1, dtype=np.float32),
                }
                write_message(stdout, result)

            else:
                write_message(stdout, {'error': 'Unknown command: %s' % command})

            frame_count += 1
            if frame_count % 100 == 0:
                sys.stderr.write(
                    "[xfeat_worker] Processed %d frames\n" % frame_count
                )
                sys.stderr.flush()

        except Exception as e:
            sys.stderr.write(
                "[xfeat_worker] Error processing frame %d: %s\n%s\n"
                % (frame_count, str(e), traceback.format_exc())
            )
            sys.stderr.flush()
            # Send error response so the caller doesn't hang
            try:
                write_message(stdout, {
                    'error': str(e),
                    'keypoints':   np.empty((0, 2), dtype=np.float32),
                    'descriptors': np.empty((0, 64), dtype=np.float32),
                    'scores':      np.empty((0,), dtype=np.float32),
                })
            except Exception:
                pass


# ── Self-test mode ────────────────────────────────────────────────────────────

def self_test():
    """Quick self-test: create a dummy frame and extract features."""
    sys.stderr.write("[xfeat_worker] Running self-test...\n")

    device = select_device()
    xfeat = XFeat()

    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    t = bgr_to_tensor(dummy, device)

    with torch.no_grad():
        output = xfeat.detectAndCompute(t, top_k=512)[0]

    kp = output['keypoints'].cpu().numpy()
    desc = output['descriptors'].cpu().numpy()
    scores = output['scores'].cpu().numpy()

    sys.stderr.write(
        "[xfeat_worker] Self-test OK\n"
        "  keypoints:   %s %s\n"
        "  descriptors: %s %s\n"
        "  scores:      %s %s\n"
        % (kp.shape, kp.dtype, desc.shape, desc.dtype, scores.shape, scores.dtype)
    )
    sys.exit(0)


if __name__ == '__main__':
    if '--test' in sys.argv:
        self_test()
    else:
        main()
