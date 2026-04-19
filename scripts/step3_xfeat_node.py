#!/usr/bin/env python2
# -*- coding: utf-8 -*-
""""
=============================================================================
step3_xfeat_node.py  —  XFeat Feature Extraction ROS Node (Python 2.7)
VT&R Project | Phase 2 — XFeat Integration

REPLACES: step3_orb_node.py

ROS NODE NAME : /orb_extractor  (topic names unchanged for compatibility)
SUBSCRIBES    : /csi_cam_0/image_raw        (sensor_msgs/Image)
PUBLISHES     : /orb/keyframe_candidate  (vtr/FrameFeatures — custom msg)
                /orb/debug_image         (sensor_msgs/Image)
                /orb/stats               (std_msgs/String — JSON)

vtr/msg/FrameFeatures:

        Header   header
        float64  timestamp
        int32    n_keypoints
        float32  quality_hint       # raw feature count normalised 0-1
        float32[] descriptors_flat  # descriptors flattened: N*64 float32
        float32[] keypoint_x        # keypoint x positions
        float32[] keypoint_y        # keypoint y positions
        float32[] keypoint_angle    # (zero for XFeat — no orientation)
        float32[] keypoint_size     # (zero for XFeat — no scale)
        int32[]   keypoint_octave   # (zero for XFeat — no pyramid)

ARCHITECTURE:
    This node is Python 2.7 (required by ROS Melodic).
    XFeat requires Python 3 + PyTorch.
    Solution: spawn xfeat_worker.py as a Python 3 subprocess and
    communicate via stdin/stdout using length-prefixed pickle IPC.

HOW TO RUN:
    rosrun vtr step3_xfeat_node.py
    rosrun vtr step3_xfeat_node.py _top_k:=512 _clip_limit:=3.0

PARAMS (set via rosparam or launch file):
    ~top_k        (int,   default=512)   max XFeat keypoints per frame
    ~clip_limit   (float, default=2.0)   CLAHE clip limit
    ~tile_size    (int,   default=8)     CLAHE tile grid NxN
    ~calib_path   (str,   default='')    path to calibration.yaml
    ~debug_viz    (bool,  default=True)  publish debug image
    ~publish_rate (int,   default=30)    max publish rate Hz
    ~top_crop     (float, default=0.20)  fraction of image to mask at top
    ~bottom_crop  (float, default=0.25)  fraction of image to mask at bottom
    ~grid_cols    (int,   default=6)     spatial binning columns (0 = off)
    ~grid_rows    (int,   default=3)     spatial binning rows    (0 = off)
    ~kp_per_cell  (int,   default=12)    max keypoints per grid cell
    ~xfeat_worker (str,   default='')    path to xfeat_worker.py override
=============================================================================
"""

import rospy
import cv2
import numpy as np
import time
import json
import os
import sys
import subprocess
import struct
import pickle

from sensor_msgs.msg import Image
from std_msgs.msg    import String
from cv_bridge       import CvBridge, CvBridgeError

from vtr.msg import FrameFeatures


# ── Calibration loader ────────────────────────────────────────────────────────

def load_calibration(yaml_path):
    import yaml
    if not yaml_path or not os.path.exists(yaml_path):
        rospy.logwarn("[XFEAT] calibration.yaml not found at: %s", yaml_path)
        rospy.logwarn("[XFEAT] Undistortion disabled.")
        return None, None

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    K = np.array(data['camera_matrix']['data'],
                 dtype=np.float64).reshape(3, 3)
    D = np.array(data['distortion_coefficients']['data'],
                 dtype=np.float64).reshape(1, 5)
    rospy.loginfo("[XFEAT] Calibration loaded from %s", yaml_path)
    rospy.loginfo("[XFEAT] fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f",
                  K[0,0], K[1,1], K[0,2], K[1,2])
    return K, D


# ── CLAHE preprocessor (reused from orb_node.py) ─────────────────────────────

class ClaheProcessor(object):
    """
    Inline CLAHE preprocessor.
    Undistorts then applies adaptive histogram equalisation.
    """

    def __init__(self, K, D, clip_limit=2.0, tile_size=8):
        self.K          = K
        self.D          = D
        self.clip_limit = clip_limit
        self.tile_size  = tile_size
        self.clahe      = cv2.createCLAHE(
            clipLimit    = clip_limit,
            tileGridSize = (tile_size, tile_size)
        )
        self.map1 = None
        self.map2 = None
        rospy.loginfo("[XFEAT] CLAHE  clip=%.1f  tile=%dx%d",
                      clip_limit, tile_size, tile_size)

    def _init_maps(self, h, w):
        if self.K is None or self.D is None:
            return
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (w, h), alpha=0, newImgSize=(w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, new_K, (w, h), cv2.CV_16SC2)
        self.K = new_K
        rospy.loginfo("[XFEAT] Undistortion maps ready for %dx%d", w, h)

    def process(self, bgr):
        """BGR → undistorted BGR."""
        h, w = bgr.shape[:2]

        # Undistort
        if self.K is not None and self.map1 is None:
            self._init_maps(h, w)

        if self.map1 is not None:
            undist = cv2.remap(bgr, self.map1, self.map2,
                               interpolation=cv2.INTER_LINEAR)
        else:
            undist = bgr

        return undist


# ── XFeat subprocess bridge ──────────────────────────────────────────────────

class XFeatBridge(object):
    """
    Python 2.7 ↔ Python 3 bridge.
    Spawns xfeat_worker.py as a subprocess, communicates via
    length-prefixed pickle over stdin/stdout.
    """

    def __init__(self, worker_path, top_k=512):
        self.worker_path = worker_path
        self.top_k       = top_k
        self.proc        = None
        self._start_worker()

    def _start_worker(self):
        """Launch the Python 3 subprocess."""
        rospy.loginfo("[XFEAT] Starting worker: python3 %s", self.worker_path)

        if not os.path.exists(self.worker_path):
            rospy.logerr("[XFEAT] Worker not found at: %s", self.worker_path)
            rospy.logerr("[XFEAT] XFeat extraction will NOT work.")
            return

        try:
            self.proc = subprocess.Popen(
                ['python3', self.worker_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,      # inherit — worker errors print to ROS console
                bufsize=0,        # unbuffered
            )
            rospy.loginfo("[XFEAT] Worker PID=%d", self.proc.pid)
        except OSError as e:
            rospy.logerr("[XFEAT] Failed to start worker: %s", str(e))
            self.proc = None

    def _send_msg(self, obj):
        """Write length-prefixed pickle to worker stdin."""
        data = pickle.dumps(obj, protocol=2)
        self.proc.stdin.write(struct.pack('>I', len(data)))
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _recv_msg(self):
        """Read length-prefixed pickle from worker stdout."""
        size_bytes = self.proc.stdout.read(4)
        if len(size_bytes) < 4:
            # Worker died or closed stdout
            rc = self.proc.poll()
            rospy.logerr("[XFEAT] Worker stdout closed (exit code: %s)", rc)
            return None
        size = struct.unpack('>I', size_bytes)[0]
        data = b''
        remaining = size
        while remaining > 0:
            chunk = self.proc.stdout.read(remaining)
            if not chunk:
                return None
            data += chunk
            remaining -= len(chunk)
        return pickle.loads(data)

    def is_alive(self):
        """Check if the worker subprocess is still running."""
        if self.proc is None:
            return False
        rc = self.proc.poll()
        if rc is not None:
            rospy.logerr("[XFEAT] Worker exited with code %d", rc)
            return False
        return True

    def extract(self, bgr_frame):
        """
        Send a BGR frame to the XFeat worker and get features back.

        Args:
            bgr_frame: np.ndarray (H,W,3) uint8 BGR

        Returns:
            dict with:
                'keypoints':   np.ndarray (N,2) float32
                'descriptors': np.ndarray (N,64) float32
                'scores':      np.ndarray (N,) float32
            or None on failure
        """
        if not self.is_alive():
            rospy.logwarn_throttle(5.0,
                "[XFEAT] Worker dead — attempting restart")
            self._start_worker()
            if not self.is_alive():
                return None

        try:
            # Send the BGR frame directly (legacy simple protocol)
            self._send_msg(bgr_frame)
            result = self._recv_msg()

            if result is None:
                rospy.logwarn("[XFEAT] Worker returned None — may have crashed")
                return None

            if 'error' in result and result['error']:
                rospy.logwarn_throttle(5.0,
                    "[XFEAT] Worker error: %s", result['error'])
                # Still return the result — it may have fallback empty arrays

            return result

        except (IOError, OSError, ValueError) as e:
            rospy.logwarn("[XFEAT] Bridge I/O error: %s", str(e))
            return None

    def shutdown(self):
        """Terminate the worker subprocess."""
        if self.proc is not None and self.proc.poll() is None:
            rospy.loginfo("[XFEAT] Shutting down worker PID=%d", self.proc.pid)
            self.proc.stdin.close()
            try:
                self.proc.wait()
            except Exception:
                self.proc.kill()


# ── Grid subsampling (reused from orb_node.py, adapted for XFeat) ────────────

def grid_subsample(kp_x, kp_y, scores, descs,
                   img_h, img_w,
                   grid_rows, grid_cols, kp_per_cell):
    """
    Spatial binning: keep only the top-kp_per_cell highest-scoring
    keypoints in each (grid_rows x grid_cols) cell.
    Prevents clustering on a single high-texture patch.

    Args:
        kp_x:    np.ndarray (N,) float32  — x positions
        kp_y:    np.ndarray (N,) float32  — y positions
        scores:  np.ndarray (N,) float32  — detection scores
        descs:   np.ndarray (N,64) float32
        img_h, img_w: image dimensions
        grid_rows, grid_cols, kp_per_cell: binning params

    Returns:
        (kp_x, kp_y, scores, descs) — filtered arrays
    """
    if grid_rows <= 0 or grid_cols <= 0 or kp_per_cell <= 0:
        return kp_x, kp_y, scores, descs
    if len(kp_x) == 0:
        return kp_x, kp_y, scores, descs

    cell_h = float(img_h) / grid_rows
    cell_w = float(img_w) / grid_cols

    # Group indices by cell
    cells = {}
    for idx in range(len(kp_x)):
        r = int(min(kp_y[idx] / cell_h, grid_rows - 1))
        c = int(min(kp_x[idx] / cell_w, grid_cols - 1))
        key = (r, c)
        if key not in cells:
            cells[key] = []
        cells[key].append(idx)

    kept = []
    for indices in cells.values():
        # Sort by score descending, keep top kp_per_cell
        indices.sort(key=lambda i: -scores[i])
        kept.extend(indices[:kp_per_cell])

    kept.sort()  # preserve spatial order
    kept = np.array(kept, dtype=np.int32)

    return kp_x[kept], kp_y[kept], scores[kept], descs[kept]


# ── Grid entropy ──────────────────────────────────────────────────────────────

def grid_entropy(kp_x, kp_y, img_h, img_w, grid=4):
    """
    Spatial distribution score — Shannon entropy of keypoint
    distribution over a grid x grid cell layout.
    """
    if len(kp_x) == 0:
        return 0.0

    counts = np.zeros((grid, grid), dtype=np.float32)
    for x, y in zip(kp_x, kp_y):
        r = int(y / img_h * grid)
        c = int(x / img_w * grid)
        r = max(0, min(r, grid - 1))
        c = max(0, min(c, grid - 1))
        counts[r, c] += 1.0

    total = counts.sum()
    if total == 0:
        return 0.0

    probs = counts.flatten() / total
    probs = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    return entropy


# ── ROS node ──────────────────────────────────────────────────────────────────

class XFeatNode(object):
    """
    ROS Melodic node that:
      1. Subscribes to /csi_cam_0/image_raw
      2. Sends frames to XFeat worker subprocess for feature extraction
      3. Publishes FrameFeatures to /orb/keyframe_candidate
      4. Publishes debug image to /orb/debug_image
      5. Publishes JSON stats to /orb/stats
    """

    def __init__(self):
        rospy.init_node('orb_extractor', anonymous=False)

        # ── Load params ───────────────────────────────────────────────────
        self.top_k        = rospy.get_param('~top_k',        512)
        clip_limit        = rospy.get_param('~clip_limit',   2.0)
        tile_size         = rospy.get_param('~tile_size',    8)
        calib_path        = rospy.get_param('~calib_path',   '')
        self.debug        = rospy.get_param('~debug_viz',    True)
        self.max_hz       = rospy.get_param('~publish_rate', 10)
        self.top_crop     = rospy.get_param('~top_crop',     0.20)
        self.bottom_crop  = rospy.get_param('~bottom_crop',  0.25)
        self.grid_cols    = rospy.get_param('~grid_cols',    6)
        self.grid_rows    = rospy.get_param('~grid_rows',    3)
        self.kp_per_cell  = rospy.get_param('~kp_per_cell',  12)
        self.infer_scale  = rospy.get_param('~infer_scale',  0.5)  # downscale for speed

        # Path to xfeat_worker.py
        default_worker = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'xfeat_worker.py'
        )
        worker_path = rospy.get_param('~xfeat_worker', default_worker)

        # Mask is built lazily on first frame (needs image size)
        self._feat_mask   = None

        # ── Calibration ───────────────────────────────────────────────────
        K, D = load_calibration(calib_path)

        # ── Pipeline objects ──────────────────────────────────────────────
        self.clahe   = ClaheProcessor(K, D,
                                      clip_limit=clip_limit,
                                      tile_size=tile_size)
        self.xfeat   = XFeatBridge(worker_path, top_k=self.top_k)
        self.bridge  = CvBridge()

        # ── State ─────────────────────────────────────────────────────────
        self.prev_desc  = None
        self.frame_n    = 0
        self.t_last_pub = 0.0
        self.min_dt     = 1.0 / self.max_hz

        # ── Subscribers ───────────────────────────────────────────────────
        self.sub_img = rospy.Subscriber(
            '/csi_cam_0/image_raw', Image,
            self._cb_image, queue_size=1, buff_size=2**24
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_feat  = rospy.Publisher(
            '/orb/keyframe_candidate', FrameFeatures, queue_size=5
        )
        self.pub_debug = rospy.Publisher(
            '/orb/debug_image', Image, queue_size=1
        )
        self.pub_stats = rospy.Publisher(
            '/orb/stats', String, queue_size=5
        )

        rospy.on_shutdown(self.xfeat.shutdown)

        rospy.loginfo("[XFEAT] Node ready. Listening on /csi_cam_0/image_raw")

    # ── Image callback ────────────────────────────────────────────────────

    def _cb_image(self, msg):
        """
        Called for every incoming camera frame.
        Runs undistort → XFeat extraction → publishes results.
        """
        now = time.time()

        # Rate limiting
        if (now - self.t_last_pub) < self.min_dt:
            return

        # ── Convert ROS Image → OpenCV BGR ────────────────────────────
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("[XFEAT] CvBridge error: %s", str(e))
            return

        h, w = bgr.shape[:2]

        # ── Undistort ──────────────────────────────────────────────────
        t0 = time.time()
        bgr_undist = self.clahe.process(bgr)
        t_preprocess = (time.time() - t0) * 1000.0

        # ── Downscale for faster XFeat inference (Jetson optimisation) ─
        sc = self.infer_scale
        if sc < 1.0:
            small = cv2.resize(bgr_undist,
                               (int(w * sc), int(h * sc)),
                               interpolation=cv2.INTER_AREA)
        else:
            small = bgr_undist

        # ── XFeat feature extraction (via Python 3 subprocess) ────────
        t1 = time.time()
        result = self.xfeat.extract(small)
        t_xfeat = (time.time() - t1) * 1000.0

        if result is None:
            rospy.logwarn_throttle(5.0, "[XFEAT] No result from worker")
            return

        kp_xy  = result.get('keypoints',   np.empty((0, 2), dtype=np.float32))
        descs  = result.get('descriptors', np.empty((0, 64), dtype=np.float32))
        scores = result.get('scores',      np.empty((0,), dtype=np.float32))

        # ── Rescale keypoints back to original image coordinates ──────
        if sc < 1.0 and len(kp_xy) > 0:
            kp_xy = kp_xy / sc

        # ── Band mask — filter out keypoints outside active zone ──────
        if len(kp_xy) > 0:
            y_lo = int(h * self.top_crop)
            y_hi = int(h * (1.0 - self.bottom_crop))
            y_mask = (kp_xy[:, 1] >= y_lo) & (kp_xy[:, 1] <= y_hi)
            kp_xy  = kp_xy[y_mask]
            descs  = descs[y_mask]
            scores = scores[y_mask]

        # Separate x, y
        if len(kp_xy) > 0:
            kp_x = kp_xy[:, 0]
            kp_y = kp_xy[:, 1]
        else:
            kp_x = np.array([], dtype=np.float32)
            kp_y = np.array([], dtype=np.float32)

        # ── Grid spatial subsampling ───────────────────────────────────
        if len(kp_x) > 0:
            kp_x, kp_y, scores, descs = grid_subsample(
                kp_x, kp_y, scores, descs,
                h, w,
                self.grid_rows, self.grid_cols, self.kp_per_cell
            )

        n_kp = len(kp_x)

        # ── Grid entropy ──────────────────────────────────────────────
        entropy = grid_entropy(kp_x, kp_y, h, w, grid=4)

        # ── Consecutive frame match quality (for stats logging) ───────
        n_good_matches = 0
        if self.prev_desc is not None and len(descs) > 0 and len(self.prev_desc) > 0:
            n_good_matches = self._count_matches(self.prev_desc, descs)

        # ── Build and publish FrameFeatures message ────────────────────
        feat_msg = FrameFeatures()
        feat_msg.header          = msg.header
        feat_msg.timestamp       = msg.header.stamp.to_sec()
        feat_msg.n_keypoints     = n_kp
        feat_msg.quality_hint    = float(min(n_kp, 500)) / 500.0

        if len(descs) > 0:
            feat_msg.descriptors_flat = descs.flatten().tolist()
        else:
            feat_msg.descriptors_flat = []

        feat_msg.keypoint_x      = kp_x.tolist() if len(kp_x) > 0 else []
        feat_msg.keypoint_y      = kp_y.tolist() if len(kp_y) > 0 else []
        feat_msg.keypoint_angle  = [0.0] * n_kp    # XFeat has no orientation
        feat_msg.keypoint_size   = [0.0] * n_kp    # XFeat has no scale
        feat_msg.keypoint_octave = [0]   * n_kp    # XFeat has no pyramid

        self.pub_feat.publish(feat_msg)

        # ── Publish stats JSON ─────────────────────────────────────────
        stats = {
            'frame'          : self.frame_n,
            'n_keypoints'    : n_kp,
            'n_good_matches' : n_good_matches,
            'entropy'        : round(entropy, 3),
            'preprocess_ms'  : round(t_preprocess, 2),
            'xfeat_ms'       : round(t_xfeat, 2),
        }
        self.pub_stats.publish(String(data=json.dumps(stats)))

        # ── Debug visualisation ────────────────────────────────────────
        if self.debug and self.pub_debug.get_num_connections() > 0:
            debug_img = self._make_debug_image(
                bgr_undist, kp_x, kp_y, scores,
                n_kp, n_good_matches,
                entropy, t_preprocess + t_xfeat, h, w
            )
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, 'bgr8')
                debug_msg.header = msg.header
                self.pub_debug.publish(debug_msg)
            except CvBridgeError:
                pass

        # ── Log every 30 frames ────────────────────────────────────────
        if self.frame_n % 30 == 0:
            rospy.loginfo(
                "[XFEAT] frame=%d  kp=%d  matches=%d  "
                "entropy=%.2f  total=%.1fms",
                self.frame_n, n_kp, n_good_matches,
                entropy, t_preprocess + t_xfeat
            )

        # ── Update state ───────────────────────────────────────────────
        self.prev_desc  = descs if len(descs) > 0 else None
        self.t_last_pub = now
        self.frame_n   += 1

    # ── L2 match quality (for stats only) ─────────────────────────────────

    def _count_matches(self, desc_a, desc_b, ratio=0.8):
        """
        Count good matches between two float32 descriptor arrays
        using L2 distance + ratio test. Log-only, not used for navigation.
        """
        if len(desc_a) < 2 or len(desc_b) < 2:
            return 0

        try:
            # Compute L2 distance matrix (N, M)
            # For efficiency, use einsum approach
            a2 = np.sum(desc_a**2, axis=1, keepdims=True)    # (N,1)
            b2 = np.sum(desc_b**2, axis=1, keepdims=True).T  # (1,M)
            ab = desc_a.dot(desc_b.T)                         # (N,M)
            dist = np.sqrt(np.maximum(a2 + b2 - 2*ab, 0.0))  # (N,M)

            # Forward nearest neighbour
            idx1 = np.argmin(dist, axis=1)  # (N,)
            d1   = dist[np.arange(len(dist)), idx1]

            # Second nearest neighbour via partition
            if dist.shape[1] < 2:
                return len(d1)
            d2 = np.partition(dist, 1, axis=1)[:, 1]

            # Ratio test
            good = np.sum((d1 < ratio * d2) & (d2 > 0))
            return int(good)
        except Exception:
            return 0

    # ── Debug image builder ────────────────────────────────────────────────

    def _make_debug_image(self, bgr, kp_x, kp_y, scores,
                          n_kp, n_matches, entropy, total_ms, img_h, img_w):
        """
        Build debug visualisation:
          - Keypoints coloured by detection score (green=high, red=low)
          - 4x4 grid overlay showing spatial distribution
          - HUD with stats
        """
        out = bgr.copy()

        # Shade masked-out zones (ceiling / floor)
        overlay = out.copy()
        y_lo = int(img_h * self.top_crop)
        y_hi = int(img_h * (1.0 - self.bottom_crop))
        cv2.rectangle(overlay, (0, 0),      (img_w, y_lo),  (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, y_hi),   (img_w, img_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, out, 0.6, 0, out)
        # Active band border
        cv2.rectangle(out, (0, y_lo), (img_w - 1, y_hi - 1), (0, 180, 80), 1)

        # Draw keypoints coloured by score
        if len(kp_x) > 0:
            s_min = scores.min() if len(scores) > 0 else 0
            s_max = scores.max() if len(scores) > 0 else 1
            s_range = max(s_max - s_min, 1e-6)
            for i in range(len(kp_x)):
                t = (scores[i] - s_min) / s_range  # 0..1
                # Green (high) to red (low)
                g = int(200 * t)
                r = int(200 * (1 - t))
                colour = (0, g, r)
                centre = (int(kp_x[i]), int(kp_y[i]))
                cv2.circle(out, centre, 3, colour, 1)

        # Spatial grid overlay
        for r in range(1, self.grid_rows):
            y = y_lo + r * (y_hi - y_lo) // self.grid_rows
            cv2.line(out, (0, y), (img_w, y), (60, 60, 60), 1)
        for c in range(1, self.grid_cols):
            x = c * img_w // self.grid_cols
            cv2.line(out, (x, y_lo), (x, y_hi), (60, 60, 60), 1)

        # HUD bar
        cv2.rectangle(out, (0, 0), (img_w, 34), (0, 0, 0), -1)
        colour = (0, 255, 0) if n_kp >= 150 else (0, 80, 220)
        cv2.putText(
            out,
            "XFeat  kp=%d  matches=%d  entropy=%.2f  %.1fms" % (
                n_kp, n_matches, entropy, total_ms
            ),
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2
        )

        return out

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rospy.loginfo("[XFEAT] Spinning. Ctrl+C to stop.")
        rospy.spin()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = XFeatNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
