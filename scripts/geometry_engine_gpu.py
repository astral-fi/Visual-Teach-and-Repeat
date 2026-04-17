#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step7_geometry_engine_gpu_node.py  —  GPU-Accelerated RANSAC Geometry Engine
VT&R Project | Phase 2 (GPU Edition)

WHAT CHANGED FROM step7_geometry_engine_node.py:
    All stages up to and including BFMatcher now run on the Jetson Nano GPU.
    RANSAC stays on CPU (no OpenCV GPU implementation exists).
    Expected speedup: 339ms → ~12ms at 640x480.

GPU PIPELINE PER FRAME:
    BGR frame (CPU)
        → cv2.cuda_GpuMat.upload()          [~0.3ms — unified memory]
        → cv2.cuda.remap()                  [~0.5ms — GPU undistort]
        → cv2.cuda.cvtColor()               [~0.2ms — GPU BGR→GRAY]
        → cv2.cuda.createCLAHE().apply()    [~0.8ms — GPU CLAHE]
        → cv2.cuda.ORB_create().detectAndCompute() [~5ms — GPU ORB]
        → cv2.cuda.DescriptorMatcher knnMatch()    [~1.5ms — GPU BFMatch]
        → download match coordinates only   [~0.1ms — tiny array]
        → cv2.findEssentialMat (RANSAC)     [~8ms — CPU, no GPU impl]
        → cv2.recoverPose                   [~0.5ms — CPU]
    TOTAL: ~16ms vs ~339ms (21x faster)

FALLBACK:
    If CUDA is not available (OpenCV not built with CUDA), automatically
    falls back to the CPU pipeline from step7_geometry_engine_node.py.
    Check availability: python2 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"

JETSON NANO CUDA NOTES:
    Maxwell GPU, 128 CUDA cores, 4GB unified memory (shared with CPU).
    Upload cost is lower than discrete GPU (no PCIe) but not zero.
    Upload ONCE per frame and keep all ops in GPU memory until RANSAC.
    Do NOT upload/download between each stage — that kills the speedup.

PREREQUISITES:
    OpenCV must be built with CUDA support:
        python2 -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
        # Must return 1, not 0

    If 0 — rebuild OpenCV:
        See: https://github.com/mdegans/nano_build_opencv
        Key cmake flags: -DWITH_CUDA=ON -DCUDA_ARCH_BIN=5.3 -DWITH_CUDNN=ON

ROS NODE NAME : /geometry_engine
    (identical topic interface to step7 — drop-in replacement)

PARAMS: identical to step7_geometry_engine_node.py
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

from std_msgs.msg    import String, Float32
from sensor_msgs.msg import Image
from cv_bridge       import CvBridge, CvBridgeError

# ── CUDA availability check ───────────────────────────────────────────────────

def check_cuda():
    """Check if OpenCV CUDA is available and log result."""
    try:
        n = cv2.cuda.getCudaEnabledDeviceCount()
        if n > 0:
            rospy.loginfo("[GEO-GPU] CUDA available: %d device(s)", n)
            return True
        else:
            rospy.logwarn("[GEO-GPU] CUDA device count = 0")
            return False
    except Exception as e:
        rospy.logwarn("[GEO-GPU] CUDA check failed: %s", str(e))
        return False


# ── Configuration ─────────────────────────────────────────────────────────────

LOWE_RATIO       = 0.75
RANSAC_THRESHOLD = 1.0
RANSAC_PROB      = 0.999
MIN_INLIERS      = 15
LK_THRESHOLD     = 15
W_LATERAL        = 0.7
W_YAW            = 0.3
N_FEATURES       = 250    # reduced from 500 — GPU ORB at 250 is faster than CPU at 500
CLIP_LIMIT       = 2.0
TILE_SIZE        = 8

# Feature distribution control
TOP_CROP     = 0.20   # mask ceiling  (top 20% of rows are sky/ceiling for a rover)
BOTTOM_CROP  = 0.25   # mask floor    (bottom 25% is featureless ground)
GRID_COLS    = 6      # spatial binning: horizontal cells
GRID_ROWS    = 3      # spatial binning: vertical cells inside the active band
KP_PER_CELL  = 12     # max keypoints kept per cell after Harris-score sorting

LK_WIN_SIZE  = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA  = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


# ── GPU preprocessing pipeline ───────────────────────────────────────────────

class GPUPreprocessor(object):
    """
    Runs undistort → grayscale → CLAHE entirely on GPU.
    Uploads frame once, returns GpuMat ready for GPU ORB.
    Falls back to CPU if CUDA unavailable.
    """

    def __init__(self, K, D, clip_limit=CLIP_LIMIT,
                 tile_size=TILE_SIZE, use_gpu=True):
        self.K        = K
        self.D        = D
        self.use_gpu  = use_gpu
        self.map1_gpu = None
        self.map2_gpu = None
        self.map1_cpu = None
        self.map2_cpu = None
        self.K_new    = None

        if use_gpu:
            try:
                self.clahe_gpu = cv2.cuda.createCLAHE(
                    clipLimit    = clip_limit,
                    tileGridSize = (tile_size, tile_size)
                )
                rospy.loginfo("[GEO-GPU] GPU CLAHE ready")
            except Exception as e:
                rospy.logwarn("[GEO-GPU] GPU CLAHE failed: %s — using CPU", str(e))
                self.use_gpu = False

        if not self.use_gpu:
            self.clahe_cpu = cv2.createCLAHE(
                clipLimit    = clip_limit,
                tileGridSize = (tile_size, tile_size)
            )

    def _init_maps(self, h, w):
        """Pre-compute undistortion maps. Called once on first frame."""
        K_new, _ = cv2.getOptimalNewCameraMatrix(
            self.K, self.D, (w, h), alpha=0, newImgSize=(w, h)
        )
        self.K_new = K_new
        map1, map2 = cv2.initUndistortRectifyMap(
            self.K, self.D, None, K_new, (w, h), cv2.CV_32FC1
        )
        if self.use_gpu:
            self.map1_gpu = cv2.cuda_GpuMat()
            self.map2_gpu = cv2.cuda_GpuMat()
            self.map1_gpu.upload(map1)
            self.map2_gpu.upload(map2)
            rospy.loginfo("[GEO-GPU] Undistortion maps on GPU for %dx%d", w, h)
        else:
            self.map1_cpu = map1
            self.map2_cpu = map2

    def process(self, bgr_cpu):
        """
        Process one BGR frame.
        Returns (gpu_gray_eq, bgr_undist_cpu)
            gpu_gray_eq  : cv2.cuda_GpuMat — CLAHE-enhanced gray, on GPU
                           If use_gpu=False: regular np.ndarray
            bgr_undist   : np.ndarray — undistorted BGR for debug display
        """
        h, w = bgr_cpu.shape[:2]

        if self.map1_gpu is None and self.map1_cpu is None:
            self._init_maps(h, w)

        if self.use_gpu:
            # Upload raw BGR once
            gpu_bgr = cv2.cuda_GpuMat()
            gpu_bgr.upload(bgr_cpu)

            # Undistort on GPU
            gpu_undist = cv2.cuda.remap(
                gpu_bgr, self.map1_gpu, self.map2_gpu,
                cv2.INTER_LINEAR
            )

            # BGR → Grayscale on GPU
            gpu_gray = cv2.cuda.cvtColor(gpu_undist, cv2.COLOR_BGR2GRAY)

            # CLAHE on GPU
            gpu_eq = self.clahe_gpu.apply(gpu_gray, cv2.cuda_Stream())

            # Download undistorted BGR for debug image only
            bgr_undist = gpu_undist.download()

            return gpu_eq, bgr_undist

        else:
            # CPU fallback
            undist = cv2.remap(bgr_cpu, self.map1_cpu, self.map2_cpu,
                               cv2.INTER_LINEAR)
            gray   = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
            eq     = self.clahe_cpu.apply(gray)
            return eq, undist


# ── GPU ORB extractor ─────────────────────────────────────────────────────────

class GPUORBExtractor(object):
    """
    ORB feature extraction on GPU.
    Input: cv2.cuda_GpuMat (grayscale, from GPUPreprocessor)
    Output: keypoints (list), descriptors (cv2.cuda_GpuMat or np.ndarray)
    """

    def __init__(self, n_features=N_FEATURES, use_gpu=True):
        self.use_gpu    = use_gpu
        self.n_features = n_features

        if use_gpu:
            try:
                self.orb_gpu = cv2.cuda.ORB_create(
                    nfeatures  = n_features,
                    scaleFactor= 1.2,
                    nlevels    = 8,
                    edgeThreshold= 31,
                    firstLevel   = 0,
                    WTA_K        = 2,
                    scoreType    = cv2.ORB_HARRIS_SCORE,
                    patchSize    = 31,
                    fastThreshold= 20,
                )
                rospy.loginfo("[GEO-GPU] GPU ORB ready  nfeatures=%d", n_features)
            except Exception as e:
                rospy.logwarn("[GEO-GPU] GPU ORB failed: %s — CPU fallback", str(e))
                self.use_gpu = False

        if not self.use_gpu:
            self.orb_cpu = cv2.ORB_create(
                nfeatures    = n_features,
                scaleFactor  = 1.2,
                nlevels      = 8,
                fastThreshold= 20,
                scoreType    = cv2.ORB_HARRIS_SCORE,
            )
            rospy.loginfo("[GEO-GPU] CPU ORB fallback  nfeatures=%d", n_features)

    def detect_and_compute(self, img, mask_cpu=None):
        """
        Args:
            img:      cv2.cuda_GpuMat (if GPU) or np.ndarray (if CPU fallback)
            mask_cpu: optional uint8 np.ndarray (HxW), 255=detect, 0=ignore
        Returns:
            (keypoints list, descriptors)
            descriptors is cv2.cuda_GpuMat if GPU, np.ndarray if CPU
        """
        if self.use_gpu:
            try:
                # GPU ORB mask must be GpuMat CV_8UC1
                gpu_mask = None
                if mask_cpu is not None:
                    gpu_mask = cv2.cuda_GpuMat()
                    gpu_mask.upload(mask_cpu)
                # GPU ORB returns GpuMat keypoints and descriptors
                gpu_kp, gpu_desc = self.orb_gpu.detectAndComputeAsync(img, gpu_mask)
                # Convert GPU keypoints to CPU list
                kp = self.orb_gpu.convert(gpu_kp)
                return kp, gpu_desc   # gpu_desc stays on GPU for BFMatcher
            except Exception as e:
                rospy.logwarn_throttle(5.0, "[GEO-GPU] GPU ORB error: %s", str(e))
                # Download and use CPU
                img_cpu = img.download() if hasattr(img, 'download') else img
                kp, desc = self.orb_cpu.detectAndCompute(img_cpu, mask_cpu)
                return kp, desc

        else:
            img_cpu = img.download() if hasattr(img, 'download') else img
            kp, desc = self.orb_cpu.detectAndCompute(img_cpu, mask_cpu)
            return kp, desc

    @staticmethod
    def grid_subsample(kps, descs_cpu, img_h, img_w,
                       grid_rows, grid_cols, kp_per_cell):
        """
        Spatial binning: keep top-kp_per_cell highest-Harris keypoints per cell.
        descs_cpu must be np.ndarray (N, 32) — download from GPU before calling.
        Returns (kps_out list, descs_out np.ndarray).
        """
        if grid_rows <= 0 or grid_cols <= 0 or kp_per_cell <= 0:
            return kps, descs_cpu
        if len(kps) == 0:
            return kps, descs_cpu

        cell_h = float(img_h) / grid_rows
        cell_w = float(img_w) / grid_cols
        cells  = {}
        for idx, kp in enumerate(kps):
            r = int(min(kp.pt[1] / cell_h, grid_rows - 1))
            c = int(min(kp.pt[0] / cell_w, grid_cols - 1))
            key = (r, c)
            if key not in cells:
                cells[key] = []
            cells[key].append(idx)

        kept = []
        for indices in cells.values():
            indices.sort(key=lambda i: -kps[i].response)
            kept.extend(indices[:kp_per_cell])

        kept.sort()
        return [kps[i] for i in kept], descs_cpu[kept]


# ── GPU BFMatcher ─────────────────────────────────────────────────────────────

class GPUBFMatcher(object):
    """
    Brute-force descriptor matching on GPU.
    Hamming distance for ORB binary descriptors.
    Falls back to CPU BFMatcher if CUDA unavailable.
    """

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

        if use_gpu:
            try:
                self.matcher_gpu = cv2.cuda.DescriptorMatcher_createBFMatcher(
                    cv2.NORM_HAMMING
                )
                rospy.loginfo("[GEO-GPU] GPU BFMatcher ready")
            except Exception as e:
                rospy.logwarn("[GEO-GPU] GPU BFMatcher failed: %s — CPU", str(e))
                self.use_gpu = False

        if not self.use_gpu:
            self.matcher_cpu = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def knn_match(self, desc_query, desc_train, k=2):
        """
        Match descriptors.
        Args:
            desc_query: GpuMat or ndarray — live frame descriptors
            desc_train: GpuMat or ndarray — stored node descriptors
        Returns:
            List of DMatch pairs (CPU, after download)
        """
        # Ensure node descriptors are on GPU if we have GPU matching
        if self.use_gpu:
            try:
                # Upload node descriptors if they are CPU arrays
                if not isinstance(desc_train, cv2.cuda_GpuMat):
                    gpu_train = cv2.cuda_GpuMat()
                    gpu_train.upload(desc_train)
                else:
                    gpu_train = desc_train

                if not isinstance(desc_query, cv2.cuda_GpuMat):
                    gpu_query = cv2.cuda_GpuMat()
                    gpu_query.upload(desc_query)
                else:
                    gpu_query = desc_query

                matches = self.matcher_gpu.knnMatch(gpu_query, gpu_train, k=k)
                return matches   # already CPU DMatch list after knnMatch

            except Exception as e:
                rospy.logwarn_throttle(5.0, "[GEO-GPU] GPU match error: %s", str(e))
                # Fall through to CPU
                desc_q = desc_query.download() if hasattr(desc_query,'download') else desc_query
                desc_t = desc_train.download() if hasattr(desc_train,'download') else desc_train
                return self.matcher_cpu.knnMatch(desc_q, desc_t, k=k)

        else:
            desc_q = desc_query.download() if hasattr(desc_query,'download') else desc_query
            desc_t = desc_train.download() if hasattr(desc_train,'download') else desc_train
            if desc_q is None or desc_t is None:
                return []
            return self.matcher_cpu.knnMatch(desc_q, desc_t, k=k)


# ── Ratio test + orientation filter (CPU — fast enough) ───────────────────────

def ratio_and_orient_filter(pairs, kp_query, kp_train,
                             ratio=LOWE_RATIO, n_bins=30, top_k=3):
    """
    Stage 1: Lowe ratio test
    Stage 2: Orientation histogram filter (ORB-SLAM3 technique)
    Both run on CPU — they operate on small lists, GPU overhead not worth it.
    """
    # Ratio test
    good = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    if len(good) < 8:
        return good

    # Orientation histogram filter
    deltas = np.array([
        kp_query[m.queryIdx].angle - kp_train[m.trainIdx].angle
        for m in good
    ])
    hist, _ = np.histogram(deltas, bins=n_bins, range=(-180.0, 180.0))
    top_idx = set(np.argsort(hist)[-top_k:].tolist())
    bin_w   = 360.0 / n_bins

    filtered = [m for m, d in zip(good, deltas)
                if max(0, min(int((d + 180.0) / bin_w), n_bins-1)) in top_idx]

    return filtered if len(filtered) >= 8 else good


# ── CPU RANSAC + recoverPose (no GPU implementation in OpenCV) ────────────────

def ransac_pose(pts_query, pts_train, K, threshold=RANSAC_THRESHOLD,
                prob=RANSAC_PROB, min_inliers=MIN_INLIERS):
    """
    RANSAC Essential Matrix → R, t
    Stays on CPU — cv2 has no GPU implementation.
    With fewer, cleaner matches from GPU BFMatcher, this runs faster.
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


# ── GPU LK optical flow fallback ──────────────────────────────────────────────

class GPULKFlowEngine(object):
    """
    Lucas-Kanade sparse optical flow on GPU.
    SparsePyrLKOpticalFlow is well-supported on Jetson Maxwell GPU.
    Falls back to CPU calcOpticalFlowPyrLK if unavailable.
    """

    def __init__(self, use_gpu=True):
        self.use_gpu    = use_gpu
        self.prev_gpu      = None
        self.prev_cpu      = None
        self.prev_pts_cpu  = None   # CPU LK format: (N, 1, 2)
        self.prev_pts_gpu  = None   # GPU LK format: (1, N, 2) CV_32FC2
        self.corner_det = cv2.GFTTDetector_create(
            maxCorners   = 150,
            qualityLevel = 0.01,
            minDistance  = 10,
        )

        if use_gpu:
            try:
                self.lk_gpu = cv2.cuda.SparsePyrLKOpticalFlow_create(
                    winSize   = LK_WIN_SIZE,
                    maxLevel  = LK_MAX_LEVEL,
                    iters     = 30,
                )
                rospy.loginfo("[GEO-GPU] GPU LK optical flow ready")
            except Exception as e:
                rospy.logwarn("[GEO-GPU] GPU LK failed: %s — CPU fallback", str(e))
                self.use_gpu = False

    def seed(self, gray, kp_cpu=None):
        """Seed tracker with new reference frame."""
        if kp_cpu and len(kp_cpu) > 0:
            pts = np.float32([[kp.pt[0], kp.pt[1]] for kp in kp_cpu[:150]])
        else:
            detected = self.corner_det.detect(
                gray.download() if hasattr(gray,'download') else gray, None
            )
            if detected:
                pts = np.float32([[kp.pt[0], kp.pt[1]] for kp in detected])
            else:
                pts = np.empty((0, 2), dtype=np.float32)

        # GPU LK requires shape (1, N, 2) CV_32FC2
        # CPU LK requires shape (N, 1, 2)
        # Keep both to avoid reshape confusion downstream
        if len(pts) > 0:
            self.prev_pts_cpu = pts.reshape(-1, 1, 2)
            self.prev_pts_gpu = pts.reshape(1, -1, 2).astype(np.float32)
        else:
            self.prev_pts_cpu = None
            self.prev_pts_gpu = None

        # Always keep CPU copy for fallback path
        gray_cpu = gray.download() if hasattr(gray, 'download') else gray
        self.prev_cpu = gray_cpu

        if self.use_gpu:
            self.prev_gpu = cv2.cuda_GpuMat()
            self.prev_gpu.upload(gray_cpu)

    def compute(self, gray_curr):
        """Track points. Returns (lateral_error, track_count)."""
        # Bug fix: use 'is None' not 'or' — numpy arrays are ambiguous in boolean context
        if self.prev_pts_cpu is None or len(self.prev_pts_cpu) < 4:
            return 0.0, 0

        gray_cpu_curr = gray_curr.download() if hasattr(gray_curr,'download') else gray_curr

        if self.use_gpu and self.prev_gpu is not None and self.prev_pts_gpu is not None:
            try:
                gpu_curr = cv2.cuda_GpuMat()
                gpu_curr.upload(gray_cpu_curr)

                # Upload GPU-format points: shape (1, N, 2) CV_32FC2
                gpu_prev_pts = cv2.cuda_GpuMat()
                gpu_prev_pts.upload(self.prev_pts_gpu)

                gpu_curr_pts, gpu_status, _ = self.lk_gpu.calc(
                    self.prev_gpu, gpu_curr, gpu_prev_pts, None
                )

                # Download and reshape to standard (N, 1, 2) / (N, 1) for downstream
                curr_pts = gpu_curr_pts.download().reshape(-1, 1, 2)
                status   = gpu_status.download().reshape(-1, 1)

                self.prev_gpu = gpu_curr
                self.prev_cpu = gray_cpu_curr

            except Exception as e:
                rospy.logwarn_throttle(5.0,
                    "[GEO-GPU] GPU LK error: %s — CPU fallback", str(e))
                # CPU fallback — prev_cpu is always kept up to date
                prev_ref = self.prev_cpu if self.prev_cpu is not None else gray_cpu_curr
                curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_ref, gray_cpu_curr,
                    self.prev_pts_cpu, None,
                    winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL,
                    criteria=LK_CRITERIA
                )
                self.prev_cpu = gray_cpu_curr

        else:
            # Full CPU path — prev_cpu always valid (set in seed())
            prev_ref = self.prev_cpu if self.prev_cpu is not None else gray_cpu_curr
            curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_ref, gray_cpu_curr,
                self.prev_pts_cpu, None,
                winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL,
                criteria=LK_CRITERIA
            )
            self.prev_cpu = gray_cpu_curr

        if curr_pts is None or status is None:
            return 0.0, 0

        good_mask = status.ravel() == 1
        if good_mask.sum() < 4:
            return 0.0, 0

        prev_good = self.prev_pts_cpu[good_mask].reshape(-1, 2)
        curr_good = curr_pts[good_mask].reshape(-1, 2)
        flow      = curr_good - prev_good
        lateral   = float(-np.median(flow[:, 0]))

        # Update both CPU and GPU point formats for next frame
        self.prev_pts_cpu = curr_good.reshape(-1, 1, 2)
        self.prev_pts_gpu = curr_good.reshape(1, -1, 2).astype(np.float32)
        return lateral, int(good_mask.sum())


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

class GPUGeometryEngineNode(object):

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
        self.debug        = rospy.get_param('~debug_viz',        False)
        self.top_crop     = rospy.get_param('~top_crop',         TOP_CROP)
        self.bottom_crop  = rospy.get_param('~bottom_crop',      BOTTOM_CROP)
        self.grid_cols    = rospy.get_param('~grid_cols',         GRID_COLS)
        self.grid_rows    = rospy.get_param('~grid_rows',         GRID_ROWS)
        self.kp_per_cell  = rospy.get_param('~kp_per_cell',       KP_PER_CELL)
        # Live-frame mask built lazily on first frame
        self._feat_mask   = None

        # ── CUDA check ────────────────────────────────────────────────────
        self.use_gpu = check_cuda()
        if not self.use_gpu:
            rospy.logwarn("[GEO-GPU] No CUDA — running full CPU pipeline")
            rospy.logwarn("[GEO-GPU] Rebuild OpenCV with CUDA for GPU acceleration")
            rospy.logwarn("[GEO-GPU] See: https://github.com/mdegans/nano_build_opencv")

        # ── Calibration ───────────────────────────────────────────────────
        self.K, self.D = self._load_calibration(calib_path)

        # ── GPU pipeline objects ──────────────────────────────────────────
        self.preprocessor = GPUPreprocessor(
            self.K, self.D, self.clip_limit, TILE_SIZE, self.use_gpu)
        self.orb_extractor = GPUORBExtractor(self.n_features, self.use_gpu)
        self.matcher       = GPUBFMatcher(self.use_gpu)
        self.lk            = GPULKFlowEngine(self.use_gpu)
        self.bridge        = CvBridge()

        # ── Current target node ───────────────────────────────────────────
        self.node_desc     = None    # np.ndarray (N,32) — CPU, uploaded per match
        self.node_kp_x     = []
        self.node_kp_y     = []
        self.node_kp_angle = []
        self.node_kp_size  = []
        self.node_kp_oct   = []
        self.node_id       = -1

        # Cache node descriptors as GpuMat for repeated matching
        self.node_desc_gpu = None

        # ── Statistics ────────────────────────────────────────────────────
        self.n_ransac    = 0
        self.n_lk        = 0
        self.n_fail      = 0
        self.n_frames    = 0
        self.total_ms    = 0.0
        self.consec_fail = 0

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber('/camera/image_raw', Image,
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

        rospy.loginfo(
            "[GEO-GPU] Ready  gpu=%s  n_features=%d  "
            "lowe=%.2f  ransac_thresh=%.1fpx  debug=%s",
            self.use_gpu, self.n_features,
            self.lowe_ratio, self.ransac_thresh,
            self.debug
        )

    # ── Calibration ───────────────────────────────────────────────────────

    def _load_calibration(self, calib_path):
        identity = np.eye(3, dtype=np.float64)
        zero_d   = np.zeros((1, 5), dtype=np.float64)
        if not calib_path or not os.path.exists(calib_path):
            rospy.logwarn("[GEO-GPU] No calibration — identity K")
            return identity, zero_d
        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix']['data'],
                     dtype=np.float64).reshape(3, 3)
        D = np.array(data['distortion_coefficients']['data'],
                     dtype=np.float64).reshape(1, 5)
        rospy.loginfo("[GEO-GPU] K loaded  fx=%.1f fy=%.1f", K[0,0], K[1,1])
        return K, D

    # ── Node callback ─────────────────────────────────────────────────────

    def _cb_node(self, msg):
        """Load new target node. Upload descriptors to GPU immediately."""
        try:
            data = json.loads(msg.data)
        except ValueError:
            return

        flat = data.get('descriptors_flat', [])
        if not flat:
            return

        self.node_desc     = np.array(flat, dtype=np.uint8).reshape(-1, 32)
        self.node_kp_x     = data.get('keypoint_x',     [])
        self.node_kp_y     = data.get('keypoint_y',     [])
        self.node_kp_angle = data.get('keypoint_angle', [])
        self.node_kp_size  = data.get('keypoint_size',  [])
        self.node_kp_oct   = data.get('keypoint_octave',[])
        self.node_id       = data.get('node_id', -1)

        # Pre-upload node descriptors to GPU so matching is instant
        if self.use_gpu and self.node_desc is not None:
            try:
                self.node_desc_gpu = cv2.cuda_GpuMat()
                self.node_desc_gpu.upload(self.node_desc)
            except Exception:
                self.node_desc_gpu = None

        rospy.logdebug("[GEO-GPU] Target node %d  kp=%d",
                       self.node_id, len(self.node_kp_x))

    # ── Image callback — main GPU pipeline ───────────────────────────────

    def _cb_image(self, msg):
        if self.node_desc is None:
            return

        t0 = time.time()
        result = GeometryResult()

        # Convert ROS → BGR
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr("[GEO-GPU] CvBridge: %s", str(e))
            return

        # ── GPU: undistort → BGR2GRAY → CLAHE ─────────────────────────
        gpu_eq, bgr_undist = self.preprocessor.process(bgr)
        h_img, w_img = bgr.shape[:2]

        # ── Build horizontal band mask (lazy, once per resolution) ─────
        # Strips featureless ceiling (top top_crop%) and floor (bottom bottom_crop%)
        if self._feat_mask is None or self._feat_mask.shape != (h_img, w_img):
            self._feat_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            y_lo = int(h_img * self.top_crop)
            y_hi = int(h_img * (1.0 - self.bottom_crop))
            y_lo = max(0, min(y_lo, h_img - 1))
            y_hi = max(y_lo + 1, min(y_hi, h_img))
            self._feat_mask[y_lo:y_hi, :] = 255
            rospy.loginfo(
                "[GEO-GPU] Feature mask: rows %d-%d of %d "
                "(ceiling=%.0f%%  floor=%.0f%%)",
                y_lo, y_hi, h_img,
                self.top_crop * 100, self.bottom_crop * 100
            )

        # ── GPU: ORB extraction (with band mask) ──────────────────────
        kp_live, desc_live = self.orb_extractor.detect_and_compute(
            gpu_eq, mask_cpu=self._feat_mask
        )

        # ── Grid spatial subsampling (CPU — tiny list op) ─────────────
        if len(kp_live) > 0 and desc_live is not None:
            desc_live_np = (
                desc_live.download() if hasattr(desc_live, 'download') else desc_live
            )
            kp_live, desc_live_np = GPUORBExtractor.grid_subsample(
                kp_live, desc_live_np, h_img, w_img,
                self.grid_rows, self.grid_cols, self.kp_per_cell
            )
            # Re-upload filtered descriptors so GPU BFMatcher can use them
            if self.use_gpu and len(kp_live) > 0:
                desc_live_gpu = cv2.cuda_GpuMat()
                desc_live_gpu.upload(desc_live_np)
                desc_live = desc_live_gpu
            else:
                desc_live = desc_live_np

        if desc_live is None or len(kp_live) == 0:
            self._publish_failure(result, t0)
            return

        # Reconstruct node keypoints
        kp_node = [
            cv2.KeyPoint(float(x), float(y), float(sz), float(ang))
            for x, y, sz, ang in zip(
                self.node_kp_x, self.node_kp_y,
                self.node_kp_size, self.node_kp_angle)
        ]

        # ── GPU: BFMatcher knnMatch ────────────────────────────────────
        node_desc_for_match = (
            self.node_desc_gpu
            if (self.use_gpu and self.node_desc_gpu is not None)
            else self.node_desc
        )
        pairs = self.matcher.knn_match(desc_live, node_desc_for_match, k=2)

        # ── CPU: ratio test + orientation filter ──────────────────────
        # Download live descriptors only if needed for CPU ops
        desc_live_cpu = (
            desc_live.download()
            if hasattr(desc_live, 'download') else desc_live
        )

        good = ratio_and_orient_filter(
            pairs, kp_live, kp_node, ratio=self.lowe_ratio
        )

        if len(good) < self.min_inliers:
            # RANSAC won't have enough points — try LK
            lk_lat, lk_cnt = self.lk.compute(gpu_eq)
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

        # Extract point coordinate arrays (tiny download from GPU match result)
        pts_live = np.float32([kp_live[m.queryIdx].pt for m in good])
        pts_node = np.float32([kp_node[m.trainIdx].pt for m in good])

        # ── CPU: RANSAC Essential Matrix ──────────────────────────────
        # (No GPU implementation in OpenCV — stays CPU)
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
            self.lk.seed(gpu_eq, kp_live)

            self.n_ransac   += 1
            self.consec_fail = 0
            result.process_ms = (time.time() - t0) * 1000.0
            self._publish(result)

        else:
            # RANSAC failed — LK fallback
            lk_lat, lk_cnt = self.lk.compute(gpu_eq)
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
                "[GEO-GPU] frames=%d  mean=%.1fms  "
                "ransac=%d  lk=%d  fail=%d  gpu=%s",
                self.n_frames, mean_ms,
                self.n_ransac, self.n_lk, self.n_fail,
                self.use_gpu
            )

        # ── Debug image (only if subscriber connected AND debug=True) ──
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
            rospy.logwarn("[GEO-GPU] %d consec failures  node=%d",
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
               "lat=%+.3f  yaw=%+.1fdeg  err=%+.3f  %.1fms  GPU=%s") % (
            result.method, result.inlier_count, result.confidence,
            result.lateral, math.degrees(result.yaw),
            result.path_error, result.process_ms, self.use_gpu
        )
        cv2.putText(canvas, hud, (8, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)
        return canvas

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rospy.loginfo("[GEO-GPU] Spinning.")
        rospy.spin()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = GPUGeometryEngineNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
