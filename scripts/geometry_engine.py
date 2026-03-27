#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step7_geometry_engine_node.py  —  ROS Melodic RANSAC Geometry Engine
VT&R Project | Phase 2

ROS NODE NAME : /geometry_engine
SUBSCRIBES    : /camera/image_raw       (sensor_msgs/Image)
                /graph/current_node     (std_msgs/String)   — JSON node data
PUBLISHES     : /geometry/path_error    (std_msgs/Float32)  — PID input
                /geometry/result        (std_msgs/String)   — JSON full result
                /geometry/debug_image   (sensor_msgs/Image) — match vis

WHAT THIS NODE DOES:
    Core of the Repeat phase. For every incoming camera frame:

    1. Extracts ORB features from the live frame (CLAHE → ORB)
    2. Matches against the current target node's stored descriptors
       using the full 4-stage pipeline:
         Stage 1 — BFMatcher knnMatch + Lowe ratio test (0.75)
         Stage 2 — Orientation histogram filter (ORB-SLAM3 technique)
         Stage 3 — RANSAC findEssentialMat (threshold=1.0px, prob=0.999)
         Stage 4 — recoverPose → R (3x3) + t (3x1 unit vector)
    3. Extracts steering signals:
         lateral_error = t[0]              (left/right path displacement)
         yaw_error     = atan2(R[1,0], R[0,0])  (heading error)
         path_error    = 0.7*lateral + 0.3*yaw  (combined PID input)
    4. Computes confidence = 1/cond(E) to gate the PID dead-band
    5. Falls back to Lucas-Kanade optical flow when inliers < LK_THRESHOLD

DEGENERATE CASE HANDLING:
    Pure rotation (robot stationary/spinning):
        confidence < 0.1 → bypass visual t, use IMU yaw only
    Planar scene (flat wall ahead):
        high cond(E) despite many inliers → trigger LK fallback
    Too few inliers:
        inlier_count < MIN_INLIERS → LK fallback, widen dead-band

THREAD: Runs on CPU Thread 2 (Geometry Engine) in the 4-thread architecture.

PARAMS:
    ~calib_path       (str,   default='')     calibration.yaml path
    ~lowe_ratio       (float, default=0.75)   ratio test threshold
    ~ransac_threshold (float, default=1.0)    epipolar error in pixels
    ~ransac_prob      (float, default=0.999)  RANSAC confidence
    ~min_inliers      (int,   default=15)     floor for reliable pose
    ~lk_threshold     (int,   default=15)     inliers below → LK fallback
    ~w_lateral        (float, default=0.7)    lateral weight in path_error
    ~w_yaw            (float, default=0.3)    yaw weight in path_error
    ~clip_limit       (float, default=2.0)    CLAHE clip limit
    ~n_features       (int,   default=500)    ORB max features
    ~debug_viz        (bool,  default=True)   publish debug image
=============================================================================
"""

import rospy
import numpy as np
import cv2
import json
import os
import yaml
import time

from std_msgs.msg    import String, Float32
from sensor_msgs.msg import Image
from cv_bridge       import CvBridge, CvBridgeError


# ── Configuration ─────────────────────────────────────────────────────────────

LOWE_RATIO       = 0.75
RANSAC_THRESHOLD = 1.0
RANSAC_PROB      = 0.999
MIN_INLIERS      = 15
LK_THRESHOLD     = 15
W_LATERAL        = 0.7
W_YAW            = 0.3
N_FEATURES       = 500
CLIP_LIMIT       = 2.0
TILE_SIZE        = 8

# LK optical flow parameters
LK_WIN_SIZE      = (21, 21)
LK_MAX_LEVEL     = 3
LK_CRITERIA      = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
LK_MAX_CORNERS   = 200


# ── Result dataclass (Python 2 compatible) ────────────────────────────────────

class GeometryResult(object):
    """
    Output of one geometry engine frame cycle.
    Always produced — check .success before trusting R, t.
    """

    def __init__(self):
        self.success       = False
        self.method        = 'none'    # 'ransac' or 'lk'
        self.inlier_count  = 0
        self.confidence    = 0.0
        self.R             = None
        self.t             = None
        self.lateral       = 0.0
        self.yaw           = 0.0
        self.path_error    = 0.0
        self.dead_band     = 0.15
        self.uncertain     = True
        self.pts_node      = None     # inlier source points (Nx2)
        self.pts_live      = None     # inlier dest   points (Nx2)
        self.process_ms    = 0.0

    def to_dict(self):
        return {
            'success'      : self.success,
            'method'       : self.method,
            'inlier_count' : self.inlier_count,
            'confidence'   : round(self.confidence,  3),
            'lateral'      : round(self.lateral,     4),
            'yaw_deg'      : round(np.degrees(self.yaw), 2),
            'path_error'   : round(self.path_error,  4),
            'dead_band'    : round(self.dead_band,   3),
            'uncertain'    : self.uncertain,
            'process_ms'   : round(self.process_ms, 2),
        }


# ── CLAHE preprocessor (inline) ───────────────────────────────────────────────

class ClaheProcessor(object):

    def __init__(self, K, D, clip_limit=CLIP_LIMIT, tile_size=TILE_SIZE):
        self.K     = K
        self.D     = D
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                     tileGridSize=(tile_size, tile_size))
        self.map1  = None
        self.map2  = None

    def _init_maps(self, h, w):
        new_K, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (w, h), alpha=0, newImgSize=(w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, new_K, (w, h), cv2.CV_16SC2)
        self.K = new_K

    def process(self, bgr):
        h, w = bgr.shape[:2]
        if self.K is not None and self.map1 is None:
            self._init_maps(h, w)
        if self.map1 is not None:
            undist = cv2.remap(bgr, self.map1, self.map2, cv2.INTER_LINEAR)
        else:
            undist = bgr
        gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray), undist


# ── Lucas-Kanade optical flow fallback ───────────────────────────────────────

class LKFlowEngine(object):
    """
    Sparse Lucas-Kanade optical flow fallback.
    Activates when RANSAC inlier count drops below LK_THRESHOLD.

    Tracks the last good ORB keypoints across frames.
    Computes path error from the dominant flow direction —
    horizontal flow component = lateral displacement proxy.

    Not as accurate as epipolar geometry but robust in:
        - Low-texture walls (few ORB features)
        - Motion blur frames
        - Glare from windows
    """

    def __init__(self):
        self.prev_gray   = None
        self.prev_pts    = None
        self.corner_det  = cv2.GFTTDetector_create(
            maxCorners   = LK_MAX_CORNERS,
            qualityLevel = 0.01,
            minDistance  = 10,
        )

    def seed(self, gray, kp_cv=None):
        """
        Seed the tracker with a new reference frame.
        Called when RANSAC succeeds (good frame) so LK always
        starts from a recently verified pose.

        If kp_cv provided, use those points (ORB keypoints).
        Otherwise detect fresh GFTT corners.
        """
        self.prev_gray = gray.copy()

        if kp_cv and len(kp_cv) > 0:
            pts = np.float32([[kp.pt[0], kp.pt[1]] for kp in kp_cv])
            # Keep up to LK_MAX_CORNERS, spread across frame
            if len(pts) > LK_MAX_CORNERS:
                pts = pts[:LK_MAX_CORNERS]
        else:
            pts = self.corner_det.detect(gray, None)
            if pts:
                pts = np.float32([[kp.pt[0], kp.pt[1]] for kp in pts])
            else:
                pts = np.empty((0, 2), dtype=np.float32)

        self.prev_pts = pts.reshape(-1, 1, 2) if len(pts) > 0 else None

    def compute(self, gray_curr):
        """
        Track points from prev frame to curr frame.
        Returns lateral_error estimate (float) and track count (int).

        lateral_error sign convention matches epipolar t[0]:
            positive = robot moved right
            negative = robot moved left
        """
        if self.prev_gray is None or self.prev_pts is None:
            return 0.0, 0

        if len(self.prev_pts) < 4:
            return 0.0, 0

        # Forward tracking
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_curr,
            self.prev_pts, None,
            winSize   = LK_WIN_SIZE,
            maxLevel  = LK_MAX_LEVEL,
            criteria  = LK_CRITERIA,
        )

        if curr_pts is None or status is None:
            return 0.0, 0

        good_mask = status.ravel() == 1
        if good_mask.sum() < 4:
            return 0.0, 0

        prev_good = self.prev_pts[good_mask].reshape(-1, 2)
        curr_good = curr_pts[good_mask].reshape(-1, 2)

        # Flow vectors
        flow = curr_good - prev_good    # (N, 2): [dx, dy]

        # Lateral error = median horizontal flow
        # Negate because rightward flow means camera (robot) moved right
        # relative to the scene — same sign as t[0]
        lateral = float(-np.median(flow[:, 0]))

        # Update reference
        self.prev_gray = gray_curr.copy()
        self.prev_pts  = curr_good.reshape(-1, 1, 2)

        return lateral, int(good_mask.sum())


# ── RANSAC geometry engine ────────────────────────────────────────────────────

class RANSACEngine(object):
    """
    Full 4-stage matching and pose recovery pipeline.

    Stage 1: BFMatcher knnMatch + Lowe ratio test
    Stage 2: Orientation histogram filter
    Stage 3: RANSAC findEssentialMat
    Stage 4: recoverPose → R, t

    Also computes:
        match_count  = RANSAC inlier count
        confidence   = 1 / cond(E) — geometric reliability score
    """

    def __init__(self, K,
                 lowe_ratio       = LOWE_RATIO,
                 ransac_threshold = RANSAC_THRESHOLD,
                 ransac_prob      = RANSAC_PROB,
                 min_inliers      = MIN_INLIERS):
        self.K                = K
        self.lowe_ratio       = lowe_ratio
        self.ransac_threshold = ransac_threshold
        self.ransac_prob      = ransac_prob
        self.min_inliers      = min_inliers
        self.matcher          = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # ── Stage 1: ratio test ───────────────────────────────────────────────

    def _ratio_filter(self, desc_node, desc_live):
        if desc_node is None or desc_live is None:
            return [], None, None
        if len(desc_node) < 5 or len(desc_live) < 5:
            return [], None, None

        try:
            pairs = self.matcher.knnMatch(desc_node, desc_live, k=2)
        except cv2.error:
            return [], None, None

        good = [m for pair in pairs
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < self.lowe_ratio * n.distance]

        return good, None, None

    # ── Stage 2: orientation histogram filter ────────────────────────────

    def _orientation_filter(self, good, kp_node, kp_live,
                            n_bins=30, top_k=3):
        if len(good) < 8:
            return good

        deltas = np.array([
            kp_node[m.queryIdx].angle - kp_live[m.trainIdx].angle
            for m in good
        ])
        hist, _ = np.histogram(deltas, bins=n_bins, range=(-180.0, 180.0))
        top_idx = set(np.argsort(hist)[-top_k:].tolist())
        bin_w   = 360.0 / n_bins

        filtered = [m for m, d in zip(good, deltas)
                    if max(0, min(int((d + 180.0) / bin_w), n_bins - 1))
                    in top_idx]

        return filtered if len(filtered) >= 8 else good

    # ── Stage 3+4: RANSAC + recoverPose ──────────────────────────────────

    def _ransac_pose(self, pts_node, pts_live):
        if len(pts_node) < self.min_inliers:
            return False, None, None, None, 0.0, 0

        try:
            E, mask = cv2.findEssentialMat(
                pts_node, pts_live, self.K,
                method    = cv2.RANSAC,
                prob      = self.ransac_prob,
                threshold = self.ransac_threshold,
            )
        except cv2.error:
            return False, None, None, None, 0.0, 0

        if E is None or mask is None:
            return False, None, None, None, 0.0, 0

        inlier_count = int(np.sum(mask))
        if inlier_count < self.min_inliers:
            return False, None, None, None, 0.0, inlier_count

        # Condition number → confidence
        # Low cond = well-conditioned E = trustworthy R, t
        cond       = np.linalg.cond(E)
        confidence = float(np.clip(1.0 / cond, 0.0, 1.0))

        # recoverPose — cheirality check picks correct (R,t) from 4
        _, R, t, _ = cv2.recoverPose(
            E, pts_node, pts_live, self.K, mask=mask
        )

        return True, R, t, mask, confidence, inlier_count

    # ── Public API ────────────────────────────────────────────────────────

    def compute(self, desc_node, kp_node_cv,
                desc_live, kp_live_cv):
        """
        Run full pipeline from descriptors to R, t.

        Args:
            desc_node   : np.ndarray (N,32) uint8 — from memory graph node
            kp_node_cv  : list of cv2.KeyPoint    — from memory graph node
            desc_live   : np.ndarray (M,32) uint8 — from live frame
            kp_live_cv  : list of cv2.KeyPoint    — from live frame

        Returns:
            (success, R, t, inlier_count, confidence,
             pts_node_inliers, pts_live_inliers)
        """
        _fail = (False, None, None, 0, 0.0, None, None)

        # Stage 1 — ratio test
        good, _, _ = self._ratio_filter(desc_node, desc_live)
        if len(good) < self.min_inliers:
            return _fail

        # Stage 2 — orientation filter
        good = self._orientation_filter(good, kp_node_cv, kp_live_cv)
        if len(good) < self.min_inliers:
            return _fail

        # Extract point arrays
        pts_node = np.float32(
            [kp_node_cv[m.queryIdx].pt for m in good])
        pts_live = np.float32(
            [kp_live_cv[m.trainIdx].pt for m in good])

        # Stage 3+4 — RANSAC + recoverPose
        success, R, t, mask, confidence, inlier_count = \
            self._ransac_pose(pts_node, pts_live)

        if not success:
            return _fail

        # Extract inlier point arrays
        inlier_idx  = np.where(mask.ravel() == 1)[0]
        pts_n_in    = pts_node[inlier_idx]
        pts_l_in    = pts_live[inlier_idx]

        return True, R, t, inlier_count, confidence, pts_n_in, pts_l_in


# ── ROS node ──────────────────────────────────────────────────────────────────

class GeometryEngineNode(object):
    """
    ROS node wrapping the RANSAC engine and LK fallback.
    Publishes path_error for the PID controller (step8).
    """

    def __init__(self):
        rospy.init_node('geometry_engine', anonymous=False)

        # ── Params ────────────────────────────────────────────────────────
        calib_path        = rospy.get_param('~calib_path',       '')
        self.lowe_ratio   = rospy.get_param('~lowe_ratio',       LOWE_RATIO)
        self.ransac_thresh= rospy.get_param('~ransac_threshold', RANSAC_THRESHOLD)
        self.ransac_prob  = rospy.get_param('~ransac_prob',      RANSAC_PROB)
        self.min_inliers  = rospy.get_param('~min_inliers',      MIN_INLIERS)
        self.lk_threshold = rospy.get_param('~lk_threshold',     LK_THRESHOLD)
        self.w_lateral    = rospy.get_param('~w_lateral',        W_LATERAL)
        self.w_yaw        = rospy.get_param('~w_yaw',            W_YAW)
        self.clip_limit   = rospy.get_param('~clip_limit',       CLIP_LIMIT)
        self.n_features   = rospy.get_param('~n_features',       N_FEATURES)
        self.debug        = rospy.get_param('~debug_viz',        True)

        # ── Calibration ───────────────────────────────────────────────────
        self.K, self.D = self._load_calibration(calib_path)

        # ── Pipeline objects ──────────────────────────────────────────────
        self.clahe  = ClaheProcessor(self.K, self.D, self.clip_limit)
        self.orb    = cv2.ORB_create(
            nfeatures    = self.n_features,
            scaleFactor  = 1.2,
            nlevels      = 8,
            fastThreshold= 20,
            scoreType    = cv2.ORB_HARRIS_SCORE,
        )
        self.ransac = RANSACEngine(
            self.K,
            lowe_ratio       = self.lowe_ratio,
            ransac_threshold = self.ransac_thresh,
            ransac_prob      = self.ransac_prob,
            min_inliers      = self.min_inliers,
        )
        self.lk     = LKFlowEngine()
        self.bridge = CvBridge()

        # ── Current target node (received from graph / repeat controller) ─
        self.current_node_desc     = None   # np.ndarray (N,32)
        self.current_node_kp_x     = []
        self.current_node_kp_y     = []
        self.current_node_kp_angle = []
        self.current_node_kp_size  = []
        self.current_node_kp_oct   = []
        self.current_node_id       = -1

        # ── Statistics ────────────────────────────────────────────────────
        self.n_ransac_success = 0
        self.n_lk_fallback    = 0
        self.n_failures       = 0
        self.consec_failures  = 0

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber('/csi_cam_0/image_raw', Image,
                         self._cb_image, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/graph/current_node', String,
                         self._cb_node, queue_size=5)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_error  = rospy.Publisher(
            '/geometry/path_error', Float32, queue_size=1)
        self.pub_result = rospy.Publisher(
            '/geometry/result', String, queue_size=5)
        self.pub_debug  = rospy.Publisher(
            '/geometry/debug_image', Image, queue_size=1)

        rospy.loginfo(
            "[GEO] Ready  lowe=%.2f  ransac_thresh=%.1fpx  "
            "min_inliers=%d  lk_thresh=%d",
            self.lowe_ratio, self.ransac_thresh,
            self.min_inliers, self.lk_threshold
        )


    # ── Calibration ───────────────────────────────────────────────────────

    def _load_calibration(self, calib_path):
        identity = np.eye(3, dtype=np.float64)
        zero_d   = np.zeros((1, 5), dtype=np.float64)
        if not calib_path or not os.path.exists(calib_path):
            rospy.logwarn("[GEO] No calibration — identity K")
            return identity, zero_d
        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix']['data'],
                     dtype=np.float64).reshape(3, 3)
        D = np.array(data['distortion_coefficients']['data'],
                     dtype=np.float64).reshape(1, 5)
        rospy.loginfo("[GEO] K loaded  fx=%.1f fy=%.1f", K[0,0], K[1,1])
        return K, D

    # ── Node callback — load current target ───────────────────────────────

    def _cb_node(self, msg):
        """
        Receive the current target node data from the repeat controller.
        Expected JSON format:
        {
            "node_id": 42,
            "descriptors_flat": [...],   # N*32 uint8 values
            "keypoint_x": [...],
            "keypoint_y": [...],
            "keypoint_angle": [...],
            "keypoint_size": [...],
            "keypoint_octave": [...]
        }
        """
        try:
            data = json.loads(msg.data)
        except ValueError:
            rospy.logwarn("[GEO DEBUG] Failed to parse JSON from /graph/current_node!")
            return

        flat = data.get('descriptors_flat', [])
        if not flat:
            rospy.logwarn("[GEO DEBUG] JSON parsed, but 'descriptors_flat' is empty!")
            return

        self.current_node_desc     = np.array(
            flat, dtype=np.uint8).reshape(-1, 32)
        self.current_node_kp_x     = data.get('keypoint_x',     [])
        self.current_node_kp_y     = data.get('keypoint_y',     [])
        self.current_node_kp_angle = data.get('keypoint_angle', [])
        self.current_node_kp_size  = data.get('keypoint_size',  [])
        self.current_node_kp_oct   = data.get('keypoint_octave',[])
        self.current_node_id       = data.get('node_id', -1)

        rospy.logdebug("[GEO] Target node updated: id=%d  kp=%d",
                       self.current_node_id,
                       len(self.current_node_kp_x))

    # ── Image callback — main processing ──────────────────────────────────

    def _cb_image(self, msg):
        if self.current_node_desc is None:
            return   # No target node loaded yet

        t0 = time.time()
        result = GeometryResult()

        # Convert ROS image
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr("[GEO] CvBridge: %s", str(e))
            return

        # CLAHE preprocessing
        gray, bgr_undist = self.clahe.process(bgr)

        # ORB extraction on live frame
        kp_live_cv, desc_live = self.orb.detectAndCompute(gray, None)
        if desc_live is None or len(kp_live_cv) == 0:
            self._publish_failure(result, t0)
            return

        # Reconstruct node keypoints as cv2.KeyPoint list
        kp_node_cv = [
            cv2.KeyPoint(float(x), float(y), float(sz), float(ang))
            for x, y, sz, ang in zip(
                self.current_node_kp_x,
                self.current_node_kp_y,
                self.current_node_kp_size,
                self.current_node_kp_angle,
            )
        ]

        # ── RANSAC geometry engine ─────────────────────────────────────
        success, R, t, inlier_count, confidence, pts_n, pts_l = \
            self.ransac.compute(
                self.current_node_desc, kp_node_cv,
                desc_live,             list(kp_live_cv)
            )
        rospy.logwarn("Inlier Count" + str(inlier_count))
        if success and inlier_count >= self.min_inliers:
            # Good RANSAC result — extract steering signals
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
            result.pts_node     = pts_n
            result.pts_live     = pts_l

            # Seed LK with good frame so fallback is always fresh
            self.lk.seed(gray, list(kp_live_cv))

            self.n_ransac_success += 1
            self.consec_failures   = 0

        else:
            # ── LK optical flow fallback ───────────────────────────────
            lk_lateral, lk_count = self.lk.compute(gray)
            rospy.logwarn("lk_count " + str(lk_count))
            if lk_count >= 4:
                result.success      = True
                result.method       = 'lk'
                result.inlier_count = lk_count
                result.confidence   = 0.2      # LK always uncertain
                result.lateral      = lk_lateral
                result.yaw          = 0.0      # LK gives no yaw
                result.path_error   = lk_lateral
                result.dead_band    = 0.15     # wide dead-band during LK
                result.uncertain    = True

                self.n_lk_fallback += 1
                self.consec_failures = 0

                rospy.logdebug("[GEO] LK fallback  tracks=%d  lateral=%.3f",
                               lk_count, lk_lateral)
            else:
                self._publish_failure(result, t0)
                return

        # ── Timing ────────────────────────────────────────────────────
        result.process_ms = (time.time() - t0) * 1000.0

        # ── Publish path error to PID ──────────────────────────────────
        self.pub_error.publish(Float32(data=result.path_error))

        # ── Publish full result JSON ───────────────────────────────────
        self.pub_result.publish(String(data=json.dumps(result.to_dict())))

        # ── Debug image ────────────────────────────────────────────────
        if self.debug and self.pub_debug.get_num_connections() > 0:
            debug_img = self._make_debug_image(
                bgr_undist, kp_node_cv, list(kp_live_cv), result
            )
            try:
                debug_msg = self.bridge.cv2_to_imgmsg(debug_img, 'bgr8')
                debug_msg.header = msg.header
                self.pub_debug.publish(debug_msg)
            except CvBridgeError:
                pass

        # ── Periodic log ───────────────────────────────────────────────
        total = self.n_ransac_success + self.n_lk_fallback + self.n_failures
        if total % 30 == 0 and total > 0:
            rospy.loginfo(
                "[GEO] ransac=%d  lk=%d  fail=%d  "
                "last: inliers=%d conf=%.2f path_err=%+.3f  %.1fms",
                self.n_ransac_success, self.n_lk_fallback, self.n_failures,
                result.inlier_count, result.confidence,
                result.path_error, result.process_ms
            )

    # ── Failure handling ──────────────────────────────────────────────────

    def _publish_failure(self, result, t0):
        result.process_ms = (time.time() - t0) * 1000.0
        self.n_failures      += 1
        self.consec_failures += 1

        # Publish zero path error — PID holds last command
        self.pub_error.publish(Float32(data=0.0))
        self.pub_result.publish(String(data=json.dumps(result.to_dict())))

        if self.consec_failures % 10 == 0:
            rospy.logwarn(
                "[GEO] %d consecutive failures  "
                "node_id=%d  check lighting / features",
                self.consec_failures, self.current_node_id
            )

    # ── Debug image ───────────────────────────────────────────────────────

    def _make_debug_image(self, bgr, kp_node, kp_live, result):
        """
        Visualise inlier matches between node and live frame.
        Left half: node descriptor positions (green dots)
        Right half: live frame with matched positions
        Green lines: inlier correspondences
        HUD: inlier count, confidence, lateral, yaw, method
        """
        h, w  = bgr.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)

        # Draw live frame on right half only
        half_w = w // 2
        if bgr.shape[1] >= half_w:
            canvas[:, half_w:] = bgr[:, :half_w]

        # Draw inlier match lines if RANSAC succeeded
        if result.method == 'ransac' and result.pts_node is not None:
            for pn, pl in zip(result.pts_node, result.pts_live):
                # Node point on left half (scaled to half width)
                xn = int(pn[0] * half_w / w)
                yn = int(pn[1])
                # Live point on right half
                xl = int(pl[0] * half_w / w) + half_w
                yl = int(pl[1])
                cv2.line(canvas, (xn, yn), (xl, yl), (0, 200, 80), 1)
                cv2.circle(canvas, (xn, yn), 3, (0, 200, 80), -1)
                cv2.circle(canvas, (xl, yl), 3, (0, 200, 80), -1)

        # LK tracks on right half
        if result.method == 'lk':
            cv2.putText(canvas, 'LK FALLBACK', (half_w + 10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 220), 2)

        # Divider line
        cv2.line(canvas, (half_w, 0), (half_w, h), (80, 80, 80), 1)

        # Column labels
        cv2.putText(canvas, 'node (memory)', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.putText(canvas, 'live frame', (half_w + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        # HUD bar
        col = (0, 200, 80) if not result.uncertain else (0, 80, 220)
        cv2.rectangle(canvas, (0, h - 36), (w, h), (20, 20, 20), -1)
        hud_txt = (
            "method=%s  inliers=%d  conf=%.2f  "
            "lat=%+.3f  yaw=%+.1fdeg  err=%+.3f  %.1fms"
        ) % (
            result.method, result.inlier_count, result.confidence,
            result.lateral, np.degrees(result.yaw),
            result.path_error, result.process_ms
        )
        cv2.putText(canvas, hud_txt, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        return canvas

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rospy.loginfo("[GEO] Spinning.")
        rospy.spin()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = GeometryEngineNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
