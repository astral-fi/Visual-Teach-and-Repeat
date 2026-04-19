#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step7_xfeat_geometry_node.py  —  XFeat Geometry Engine for VT&R Repeat Phase
VT&R Project | Phase 2 — XFeat Integration

REPLACES: step7_geometry_engine_gpu_node.py

WHAT CHANGED:
    All ORB extraction + BFMatcher matching replaced with XFeat:
      - Live frame features from XFeat worker subprocess (Python 3)
      - Stored node descriptors are float32 (N,64) instead of uint8 (N,32)
      - Matching via L2 mutual nearest neighbour (no Hamming, no BFMatcher)
      - No orientation histogram filter (XFeat has no angles)
      - RANSAC + recoverPose stays IDENTICAL to previous implementation
      - LK optical flow fallback stays IDENTICAL (CPU path)

ROS NODE NAME : /geometry_engine
    (identical topic interface — drop-in replacement)

SUBSCRIBES    : /csi_camera_0/image_raw    (sensor_msgs/Image)
                /graph/current_node  (std_msgs/String, JSON with descriptors)
PUBLISHES     : /geometry/path_error (std_msgs/Float32)
                /geometry/result     (std_msgs/String, JSON)
                /geometry/debug_image (sensor_msgs/Image)

PARAMS: similar to step7_geometry_engine_gpu_node.py
    ~calib_path        (str)    calibration.yaml
    ~ransac_threshold  (float)  RANSAC pixel threshold
    ~ransac_prob       (float)  RANSAC probability
    ~min_inliers       (int)    minimum inliers for success
    ~lk_threshold      (int)    minimum LK tracked points
    ~w_lateral         (float)  lateral weight in path error
    ~w_yaw             (float)  yaw weight in path error
    ~clip_limit        (float)  CLAHE clip limit
    ~top_k             (int)    max XFeat keypoints per frame
    ~match_ratio       (float)  L2 ratio test threshold
    ~debug_viz         (bool)   publish debug image
    ~top_crop          (float)  mask top fraction
    ~bottom_crop       (float)  mask bottom fraction
    ~xfeat_worker      (str)    path to xfeat_worker.py
=============================================================================
"""

import rospy
import numpy as np
import cv2
import json
import os
import yaml
import time
import math
import subprocess
import struct
import pickle

from std_msgs.msg    import String, Float32
from sensor_msgs.msg import Image
from cv_bridge       import CvBridge, CvBridgeError


# ── Configuration ─────────────────────────────────────────────────────────────

RANSAC_THRESHOLD = 1.0
RANSAC_PROB      = 0.999
MIN_INLIERS      = 15
LK_THRESHOLD     = 15
W_LATERAL        = 0.7
W_YAW            = 0.3
TOP_K            = 512
MATCH_RATIO      = 0.9
CLIP_LIMIT       = 2.0
TILE_SIZE        = 8
DESCRIPTOR_DIM   = 64

# Feature distribution control
TOP_CROP     = 0.20
BOTTOM_CROP  = 0.25

LK_WIN_SIZE  = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


# ── XFeat subprocess bridge (same as in step3) ───────────────────────────────

class XFeatBridge(object):
    """
    Python 2.7 ↔ Python 3 bridge for XFeat inference.
    Spawns xfeat_worker.py, communicates via length-prefixed pickle.
    """

    def __init__(self, worker_path, top_k=TOP_K):
        self.worker_path = worker_path
        self.top_k       = top_k
        self.proc        = None
        self._start_worker()

    def _start_worker(self):
        rospy.loginfo("[GEO-XF] Starting XFeat worker: python3 %s",
                      self.worker_path)
        if not os.path.exists(self.worker_path):
            rospy.logerr("[GEO-XF] Worker not found: %s", self.worker_path)
            return
        try:
            self.proc = subprocess.Popen(
                ['python3', self.worker_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            rospy.loginfo("[GEO-XF] Worker PID=%d", self.proc.pid)
        except OSError as e:
            rospy.logerr("[GEO-XF] Worker start failed: %s", str(e))
            self.proc = None

    def _send_msg(self, obj):
        data = pickle.dumps(obj, protocol=2)
        self.proc.stdin.write(struct.pack('>I', len(data)))
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def _recv_msg(self):
        size_bytes = self.proc.stdout.read(4)
        if len(size_bytes) < 4:
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
        if self.proc is None:
            return False
        return self.proc.poll() is None

    def extract(self, bgr_frame):
        """Send BGR frame to worker, get keypoints + descriptors back."""
        if not self.is_alive():
            rospy.logwarn_throttle(5.0,
                "[GEO-XF] Worker dead — restarting")
            self._start_worker()
            if not self.is_alive():
                return None
        try:
            self._send_msg(bgr_frame)
            return self._recv_msg()
        except (IOError, OSError, ValueError) as e:
            rospy.logwarn("[GEO-XF] Bridge error: %s", str(e))
            return None

    def shutdown(self):
        if self.proc is not None and self.proc.poll() is None:
            rospy.loginfo("[GEO-XF] Shutting down worker PID=%d",
                          self.proc.pid)
            self.proc.stdin.close()
            try:
                self.proc.wait()
            except Exception:
                self.proc.kill()


# ── XFeat L2 matching ────────────────────────────────────────────────────────

def xfeat_match(desc_node, desc_live, kp_node, kp_live, ratio=MATCH_RATIO):
    """
    Match XFeat descriptors using L2 mutual nearest neighbour + ratio test.

    Args:
        desc_node : np.ndarray (N, 64) float32 — stored node descriptors
        desc_live : np.ndarray (M, 64) float32 — live frame descriptors
        kp_node   : np.ndarray (N, 2) float32  — stored node keypoints
        kp_live   : np.ndarray (M, 2) float32  — live frame keypoints
        ratio     : float — ratio test threshold (default 0.9)

    Returns:
        pts_node : np.ndarray (K, 2) float32 — matched node points
        pts_live : np.ndarray (K, 2) float32 — matched live points
    """
    N = len(desc_node)
    M = len(desc_live)

    if N < 2 or M < 2:
        return (np.empty((0, 2), dtype=np.float32),
                np.empty((0, 2), dtype=np.float32))

    # Compute L2 distance matrix efficiently
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    a2 = np.sum(desc_node**2, axis=1, keepdims=True)    # (N,1)
    b2 = np.sum(desc_live**2, axis=1, keepdims=True).T  # (1,M)
    ab = desc_node.dot(desc_live.T)                      # (N,M)
    d2 = np.maximum(a2 + b2 - 2*ab, 0.0)
    d  = np.sqrt(d2)                                     # (N,M)

    # Forward nearest neighbour: for each node descriptor, find best live
    fwd_idx  = np.argmin(d, axis=1)           # (N,)
    fwd_dist = d[np.arange(N), fwd_idx]       # (N,)

    # Second-nearest for ratio test
    d_sorted = np.partition(d, 1, axis=1)     # partial sort
    second_dist = d_sorted[:, 1]              # (N,)

    # Ratio test
    mask = (fwd_dist < ratio * second_dist) & (second_dist > 1e-6)

    # Backward nearest neighbour (mutual consistency check)
    bwd_idx = np.argmin(d, axis=0)            # (M,)
    fwd_indices_masked = np.arange(N)[mask]
    mutual_mask = np.array([
        bwd_idx[fwd_idx[i]] == i
        for i in fwd_indices_masked
    ], dtype=bool) if len(fwd_indices_masked) > 0 else np.array([], dtype=bool)

    # Apply mutual mask
    final_node_idx = fwd_indices_masked[mutual_mask]
    final_live_idx = fwd_idx[final_node_idx]

    pts_node = kp_node[final_node_idx]
    pts_live = kp_live[final_live_idx]

    return pts_node, pts_live


# ── RANSAC + recoverPose (IDENTICAL to previous step7) ────────────────────────

def ransac_pose(pts_query, pts_train, K,
                threshold=RANSAC_THRESHOLD, prob=RANSAC_PROB,
                min_inliers=MIN_INLIERS):
    """
    RANSAC Essential Matrix → R, t.
    Stays on CPU — cv2 has no GPU implementation.
    """
    if len(pts_query) < min_inliers:
        return False, None, None, None, 0.0, 0

    try:
        E, mask = cv2.findEssentialMat(
            pts_query, pts_train, K,
            method    = cv2.RANSAC,
            prob      = prob,
            threshold = threshold,
        )
    except cv2.error:
        return False, None, None, None, 0.0, 0

    if E is None or mask is None:
        return False, None, None, None, 0.0, 0

    inlier_count = int(np.sum(mask))
    if inlier_count < min_inliers:
        return False, None, None, None, 0.0, inlier_count

    cond       = np.linalg.cond(E)
    confidence = float(np.clip(1.0 / cond, 0.0, 1.0))

    _, R, t, _ = cv2.recoverPose(E, pts_query, pts_train, K, mask=mask)

    return True, R, t, mask, confidence, inlier_count


# ── LK optical flow fallback (CPU, simplified from GPU version) ──────────────

class LKFlowEngine(object):
    """
    Lucas-Kanade sparse optical flow (CPU).
    Provides lateral error estimate when XFeat matching fails.
    """

    def __init__(self):
        self.prev_gray      = None
        self.prev_pts       = None   # (N, 1, 2) float32
        self.corner_det = cv2.GFTTDetector_create(
            maxCorners   = 150,
            qualityLevel = 0.01,
            minDistance   = 10,
        )

    def seed(self, gray, kp_xy=None):
        """Seed tracker with new reference frame."""
        if kp_xy is not None and len(kp_xy) > 0:
            pts = kp_xy[:150].astype(np.float32)
        else:
            detected = self.corner_det.detect(gray, None)
            if detected:
                pts = np.float32([[kp.pt[0], kp.pt[1]] for kp in detected])
            else:
                pts = np.empty((0, 2), dtype=np.float32)

        if len(pts) > 0:
            self.prev_pts = pts.reshape(-1, 1, 2)
        else:
            self.prev_pts = None

        self.prev_gray = gray

    def compute(self, gray_curr):
        """Track points. Returns (lateral_error, track_count)."""
        if self.prev_pts is None or len(self.prev_pts) < 4:
            return 0.0, 0

        if self.prev_gray is None:
            return 0.0, 0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_curr,
            self.prev_pts, None,
            winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL,
            criteria=LK_CRITERIA
        )

        if curr_pts is None or status is None:
            return 0.0, 0

        good_mask = status.ravel() == 1
        if good_mask.sum() < 4:
            return 0.0, 0

        prev_good = self.prev_pts[good_mask].reshape(-1, 2)
        curr_good = curr_pts[good_mask].reshape(-1, 2)
        flow      = curr_good - prev_good
        lateral   = float(-np.median(flow[:, 0]))

        # Update for next frame
        self.prev_pts  = curr_good.reshape(-1, 1, 2)
        self.prev_gray = gray_curr
        return lateral, int(good_mask.sum())


# ── Preprocessor (CPU, undistort + CLAHE) ─────────────────────────────────────

class Preprocessor(object):
    """Undistort + CLAHE on CPU."""

    def __init__(self, K, D, clip_limit=CLIP_LIMIT, tile_size=TILE_SIZE):
        self.K      = K
        self.D      = D
        self.K_new  = None
        self.map1   = None
        self.map2   = None
        self.clahe  = cv2.createCLAHE(
            clipLimit    = clip_limit,
            tileGridSize = (tile_size, tile_size)
        )

    def _init_maps(self, h, w):
        K_new, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (w, h), alpha=0, newImgSize=(w, h))
        self.K_new = K_new
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, K_new, (w, h), cv2.CV_32FC1)
        rospy.loginfo("[GEO-XF] Undistortion maps ready for %dx%d", w, h)

    def process(self, bgr_cpu):
        """Returns (gray_eq, bgr_undist)."""
        h, w = bgr_cpu.shape[:2]
        if self.map1 is None:
            self._init_maps(h, w)
        undist = cv2.remap(bgr_cpu, self.map1, self.map2,
                           cv2.INTER_LINEAR)
        gray   = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        eq     = self.clahe.apply(gray)
        return eq, undist


# ── Result dataclass ──────────────────────────────────────────────────────────

class GeometryResult(object):
    def __init__(self):
        self.success      = False
        self.method       = 'none'
        self.inlier_count = 0
        self.confidence   = 0.0
        self.R            = None
        self.t            = None
        self.lateral      = 0.0
        self.yaw          = 0.0
        self.path_error   = 0.0
        self.dead_band    = 0.15
        self.uncertain    = True
        self.pts_query    = None
        self.pts_train    = None
        self.process_ms   = 0.0

    def to_dict(self):
        return {
            'success'      : self.success,
            'method'       : self.method,
            'inlier_count' : self.inlier_count,
            'confidence'   : round(self.confidence,  3),
            'lateral'      : round(self.lateral,     4),
            'yaw_deg'      : round(math.degrees(self.yaw), 2),
            'path_error'   : round(self.path_error,  4),
            'dead_band'    : round(self.dead_band,   3),
            'uncertain'    : self.uncertain,
            'process_ms'   : round(self.process_ms,  2),
        }


# ── ROS node ──────────────────────────────────────────────────────────────────

class XFeatGeometryEngineNode(object):

    def __init__(self):
        rospy.init_node('geometry_engine', anonymous=False)

        # ── Params ────────────────────────────────────────────────────────
        calib_path         = rospy.get_param('~calib_path',       '')
        self.ransac_thresh = rospy.get_param('~ransac_threshold', RANSAC_THRESHOLD)
        self.ransac_prob   = rospy.get_param('~ransac_prob',      RANSAC_PROB)
        self.min_inliers   = rospy.get_param('~min_inliers',      MIN_INLIERS)
        self.lk_threshold  = rospy.get_param('~lk_threshold',     LK_THRESHOLD)
        self.w_lateral     = rospy.get_param('~w_lateral',        W_LATERAL)
        self.w_yaw         = rospy.get_param('~w_yaw',            W_YAW)
        self.clip_limit    = rospy.get_param('~clip_limit',       CLIP_LIMIT)
        self.top_k         = rospy.get_param('~top_k',            TOP_K)
        self.match_ratio   = rospy.get_param('~match_ratio',      MATCH_RATIO)
        self.debug         = rospy.get_param('~debug_viz',        False)
        self.top_crop      = rospy.get_param('~top_crop',         TOP_CROP)
        self.bottom_crop   = rospy.get_param('~bottom_crop',      BOTTOM_CROP)

        # Path to xfeat_worker.py
        default_worker = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'xfeat_worker.py'
        )
        worker_path = rospy.get_param('~xfeat_worker', default_worker)

        # ── Calibration ───────────────────────────────────────────────────
        self.K, self.D = self._load_calibration(calib_path)

        # ── Pipeline objects ──────────────────────────────────────────────
        self.preprocessor = Preprocessor(self.K, self.D, self.clip_limit)
        self.xfeat_bridge = XFeatBridge(worker_path, top_k=self.top_k)
        self.lk           = LKFlowEngine()
        self.bridge       = CvBridge()

        # ── Current target node ───────────────────────────────────────────
        self.node_desc     = None    # np.ndarray (N,64) float32
        self.node_kp       = None    # np.ndarray (N,2)  float32
        self.node_kp_x     = []
        self.node_kp_y     = []
        self.node_kp_angle = []
        self.node_kp_size  = []
        self.node_kp_oct   = []
        self.node_id       = -1

        # ── Statistics ────────────────────────────────────────────────────
        self.n_ransac    = 0
        self.n_lk        = 0
        self.n_fail      = 0
        self.n_frames    = 0
        self.total_ms    = 0.0
        self.consec_fail = 0

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber('/csi_camera_0/image_raw', Image,
                         self._cb_image, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/graph/current_node', String,
                         self._cb_node, queue_size=5)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_error  = rospy.Publisher('/geometry/path_error',
                                          Float32, queue_size=1)
        self.pub_result = rospy.Publisher('/geometry/result',
                                          String, queue_size=5)
        self.pub_debug  = rospy.Publisher('/geometry/debug_image',
                                          Image, queue_size=1)

        rospy.on_shutdown(self.xfeat_bridge.shutdown)

        rospy.loginfo(
            "[GEO-XF] Ready  top_k=%d  match_ratio=%.2f  "
            "ransac_thresh=%.1fpx  debug=%s",
            self.top_k, self.match_ratio,
            self.ransac_thresh, self.debug
        )

    # ── Calibration ───────────────────────────────────────────────────────

    def _load_calibration(self, calib_path):
        identity = np.eye(3, dtype=np.float64)
        zero_d   = np.zeros((1, 5), dtype=np.float64)
        if not calib_path or not os.path.exists(calib_path):
            rospy.logwarn("[GEO-XF] No calibration — identity K")
            return identity, zero_d
        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix']['data'],
                     dtype=np.float64).reshape(3, 3)
        D = np.array(data['distortion_coefficients']['data'],
                     dtype=np.float64).reshape(1, 5)
        rospy.loginfo("[GEO-XF] K loaded  fx=%.1f fy=%.1f", K[0,0], K[1,1])
        return K, D

    # ── Node callback ─────────────────────────────────────────────────────

    def _cb_node(self, msg):
        """Load new target node. XFeat descriptors are float32 (N,64)."""
        try:
            data = json.loads(msg.data)
        except ValueError:
            return

        flat = data.get('descriptors_flat', [])
        if not flat:
            return

        self.node_desc = np.array(flat, dtype=np.float32).reshape(-1, DESCRIPTOR_DIM)
        self.node_kp_x     = data.get('keypoint_x',     [])
        self.node_kp_y     = data.get('keypoint_y',     [])
        self.node_kp_angle = data.get('keypoint_angle', [])
        self.node_kp_size  = data.get('keypoint_size',  [])
        self.node_kp_oct   = data.get('keypoint_octave',[])
        self.node_id       = data.get('node_id', -1)

        # Build node keypoint array (N, 2) for matching
        self.node_kp = np.column_stack([
            np.array(self.node_kp_x, dtype=np.float32),
            np.array(self.node_kp_y, dtype=np.float32),
        ]) if self.node_kp_x else np.empty((0, 2), dtype=np.float32)

        rospy.logdebug("[GEO-XF] Target node %d  kp=%d",
                       self.node_id, len(self.node_kp_x))

    # ── Image callback — main pipeline ───────────────────────────────────

    def _cb_image(self, msg):
        if self.node_desc is None:
            return

        t0 = time.time()
        result = GeometryResult()

        # Convert ROS → BGR
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr("[GEO-XF] CvBridge: %s", str(e))
            return

        h_img, w_img = bgr.shape[:2]

        # ── Preprocess: undistort + CLAHE ──────────────────────────────
        gray_eq, bgr_undist = self.preprocessor.process(bgr)

        # ── XFeat extraction on live frame (via subprocess) ───────────
        xf_result = self.xfeat_bridge.extract(bgr_undist)
        if xf_result is None:
            self._publish_failure(result, t0)
            return

        kp_live  = xf_result.get('keypoints',   np.empty((0, 2), dtype=np.float32))
        desc_live = xf_result.get('descriptors', np.empty((0, 64), dtype=np.float32))
        scores   = xf_result.get('scores',      np.empty((0,), dtype=np.float32))

        # ── Band mask — filter keypoints outside active zone ──────────
        if len(kp_live) > 0:
            y_lo = int(h_img * self.top_crop)
            y_hi = int(h_img * (1.0 - self.bottom_crop))
            y_mask = (kp_live[:, 1] >= y_lo) & (kp_live[:, 1] <= y_hi)
            kp_live   = kp_live[y_mask]
            desc_live = desc_live[y_mask]

        if len(kp_live) == 0 or len(desc_live) == 0:
            self._publish_failure(result, t0)
            return

        # ── XFeat L2 matching (node vs live) ──────────────────────────
        pts_node, pts_live = xfeat_match(
            self.node_desc, desc_live,
            self.node_kp, kp_live,
            ratio=self.match_ratio
        )

        if len(pts_node) < self.min_inliers:
            # Too few matches — try LK fallback
            lk_lat, lk_cnt = self.lk.compute(gray_eq)
            if lk_cnt >= 4:
                result.success      = True
                result.method       = 'lk'
                result.inlier_count = lk_cnt
                result.confidence   = 0.2
                result.lateral      = lk_lat
                result.yaw          = 0.0
                result.path_error   = lk_lat
                result.dead_band    = 0.15
                result.uncertain    = True
                result.process_ms   = (time.time() - t0) * 1000.0
                self.n_lk += 1
                self.consec_fail = 0
                self._publish(result)
            else:
                self._publish_failure(result, t0)
            return

        # ── RANSAC Essential Matrix (IDENTICAL to previous step7) ─────
        K = self.preprocessor.K_new if self.preprocessor.K_new is not None else self.K
        success, R, t, mask, confidence, inlier_count = ransac_pose(
            pts_live, pts_node, K,
            threshold   = self.ransac_thresh,
            prob        = self.ransac_prob,
            min_inliers = self.min_inliers,
        )

        if success:
            lateral    = float(t[0][0])
            yaw        = float(np.arctan2(R[1, 0], R[0, 0]))
            path_error = self.w_lateral * lateral + self.w_yaw * yaw
            dead_band  = 0.05 if confidence > 0.4 else 0.15

            result.success      = True
            result.method       = 'ransac'
            result.inlier_count = inlier_count
            result.confidence   = confidence
            result.R            = R
            result.t            = t
            result.lateral      = lateral
            result.yaw          = yaw
            result.path_error   = path_error
            result.dead_band    = dead_band
            result.uncertain    = confidence < 0.4 or inlier_count < 20

            # Inlier points for debug visualisation
            inlier_idx      = np.where(mask.ravel() == 1)[0]
            result.pts_query = pts_live[inlier_idx]
            result.pts_train = pts_node[inlier_idx]

            # Seed LK with good frame
            self.lk.seed(gray_eq, kp_live)

            self.n_ransac    += 1
            self.consec_fail  = 0
            result.process_ms = (time.time() - t0) * 1000.0
            self._publish(result)

        else:
            # RANSAC failed — LK fallback
            lk_lat, lk_cnt = self.lk.compute(gray_eq)
            if lk_cnt >= 4:
                result.success      = True
                result.method       = 'lk'
                result.inlier_count = lk_cnt
                result.confidence   = 0.2
                result.lateral      = lk_lat
                result.path_error   = lk_lat
                result.dead_band    = 0.15
                result.uncertain    = True
                result.process_ms   = (time.time() - t0) * 1000.0
                self.n_lk += 1
                self.consec_fail = 0
                self._publish(result)
            else:
                self._publish_failure(result, t0)

        # ── Timing stats ──────────────────────────────────────────────
        self.n_frames  += 1
        self.total_ms  += result.process_ms

        if self.n_frames % 50 == 0:
            mean_ms = self.total_ms / max(self.n_frames, 1)
            rospy.loginfo(
                "[GEO-XF] frames=%d  mean=%.1fms  "
                "ransac=%d  lk=%d  fail=%d",
                self.n_frames, mean_ms,
                self.n_ransac, self.n_lk, self.n_fail
            )

        # ── Debug image ───────────────────────────────────────────────
        if self.debug and self.pub_debug.get_num_connections() > 0:
            debug_img = self._make_debug_image(bgr_undist, result)
            try:
                dbg_msg = self.bridge.cv2_to_imgmsg(debug_img, 'bgr8')
                dbg_msg.header = msg.header
                self.pub_debug.publish(dbg_msg)
            except CvBridgeError:
                pass

    # ── Publish ───────────────────────────────────────────────────────────

    def _publish(self, result):
        self.pub_error.publish(Float32(data=result.path_error))
        self.pub_result.publish(String(data=json.dumps(result.to_dict())))

    def _publish_failure(self, result, t0):
        result.process_ms = (time.time() - t0) * 1000.0
        self.n_fail      += 1
        self.consec_fail += 1
        self.pub_error.publish(Float32(data=0.0))
        self.pub_result.publish(String(data=json.dumps(result.to_dict())))
        if self.consec_fail % 15 == 0:
            rospy.logwarn("[GEO-XF] %d consec failures  node=%d",
                          self.consec_fail, self.node_id)

    # ── Debug image ───────────────────────────────────────────────────────

    def _make_debug_image(self, bgr_undist, result):
        h, w  = bgr_undist.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)
        half_w = w // 2
        canvas[:, half_w:] = bgr_undist[:, :half_w]
        cv2.line(canvas, (half_w, 0), (half_w, h), (80, 80, 80), 1)
        cv2.putText(canvas, 'node (memory)', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
        cv2.putText(canvas, 'live frame', (half_w+10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

        if result.method == 'ransac' and result.pts_query is not None:
            for pq, pt in zip(result.pts_query, result.pts_train):
                xq = int(pq[0] * half_w / w)
                yq = int(pq[1])
                xt = int(pt[0] * half_w / w) + half_w
                yt = int(pt[1])
                cv2.line(canvas, (xq,yq), (xt,yt), (0,200,80), 1)
                cv2.circle(canvas, (xq,yq), 3, (0,200,80), -1)
                cv2.circle(canvas, (xt,yt), 3, (0,200,80), -1)

        col = (0,200,80) if not result.uncertain else (0,80,220)
        cv2.rectangle(canvas, (0,h-36), (w,h), (20,20,20), -1)
        hud = ("method=%s  inliers=%d  conf=%.2f  "
               "lat=%+.3f  yaw=%+.1fdeg  err=%+.3f  %.1fms") % (
            result.method, result.inlier_count, result.confidence,
            result.lateral, math.degrees(result.yaw),
            result.path_error, result.process_ms
        )
        cv2.putText(canvas, hud, (8, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
        return canvas

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rospy.loginfo("[GEO-XF] Spinning.")
        rospy.spin()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = XFeatGeometryEngineNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
