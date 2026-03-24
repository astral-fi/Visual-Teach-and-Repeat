#!/usr/bin/env python2
""""
ROS NODE NAME : /orb_extractor
SUBSCRIBES    : /camera/image_raw        (sensor_msgs/Image)
PUBLISHES     : /orb/keyframe_candidate  (vtr/FrameFeatures — custom msg)
                /orb/debug_image         (sensor_msgs/Image)
                /orb/stats               (std_msgs/String — JSON)

vtr/msg/FrameFeatures:

        Header   header
        float64  timestamp
        int32    n_keypoints
        float32  quality_hint       # raw feature count normalised 0-1
        uint8[]  descriptors_flat   # descriptors flattened: N*32 bytes
        float32[] keypoint_x        # keypoint x positions
        float32[] keypoint_y        # keypoint y positions
        float32[] keypoint_angle    # keypoint orientation angles
        float32[] keypoint_size     # keypoint scale sizes
        int32[]  keypoint_octave    # pyramid level per keypoint

HOW TO RUN:
    rosrun vtr orb_node.py
    rosrun vtr orb_node.py _n_features:=800 _clip_limit:=3.0

PARAMS (set via rosparam or launch file):
    ~n_features   (int,   default=1000)  max ORB keypoints per frame
    ~scale_factor (float, default=1.2)   pyramid scale factor
    ~n_levels     (int,   default=8)     pyramid depth
    ~fast_thresh  (int,   default=20)    FAST corner sensitivity
    ~lowe_ratio   (float, default=0.75)  ratio test threshold
    ~clip_limit   (float, default=2.0)   CLAHE clip limit
    ~tile_size    (int,   default=8)     CLAHE tile grid NxN
    ~calib_path   (str,   default='')    path to calibration.yaml
    ~debug_viz    (bool,  default=True)  publish debug image
    ~publish_rate (int,   default=30)    max publish rate Hz
=============================================================================
"""

import rospy
import cv2
import numpy as np
import time
import json
import os
import sys

from sensor_msgs.msg import Image
from std_msgs.msg    import String
from cv_bridge       import CvBridge, CvBridgeError

from vtr.msg import FrameFeatures


# ── Octave colours for debug visualisation ────────────────────────────────────
OCTAVE_COLOURS = [
    (0,   255,   0),   # octave 0 — full resolution (green)
    (0,   200, 255),   # octave 1 (cyan)
    (255, 180,   0),   # octave 2 (orange)
    (255,  50,  50),   # octave 3+ (red)
]


# ── Calibration loader ────────────────────────────────────────────────────────

def load_calibration(yaml_path):

    import yaml
    if not yaml_path or not os.path.exists(yaml_path):
        rospy.logwarn("[ORB] calibration.yaml not found at: %s", yaml_path)
        rospy.logwarn("[ORB] Undistortion disabled.")
        return None, None

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    K = np.array(data['camera_matrix']['data'],
                 dtype=np.float64).reshape(3, 3)
    D = np.array(data['distortion_coefficients']['data'],
                 dtype=np.float64).reshape(1, 5)
    rospy.loginfo("[ORB] Calibration loaded from %s", yaml_path)
    rospy.loginfo("[ORB] fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f",
                  K[0,0], K[1,1], K[0,2], K[1,2])
    return K, D


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
        rospy.loginfo("[ORB] CLAHE  clip=%.1f  tile=%dx%d", clip_limit, tile_size, tile_size)

    def _init_maps(self, h, w):
        if self.K is None or self.D is None:
            return
        new_K, _ = cv2.getOptimalNewCameraMatrix(self.K, self.D, (w, h), alpha=0, newImgSize=(w, h))
        self.map1, self.map2 = cv2.initUndistortRectifyMap(self.K, self.D, None, new_K, (w, h), cv2.CV_16SC2)
        self.K = new_K
        rospy.loginfo("[ORB] Undistortion maps ready for %dx%d", w, h)

    def process(self, bgr):
        """BGR → undistorted BGR + CLAHE grayscale."""
        h, w = bgr.shape[:2]

        # Undistort
        if self.K is not None and self.map1 is None:
            self._init_maps(h, w)

        if self.map1 is not None:
            undist = cv2.remap(bgr, self.map1, self.map2,
                               interpolation=cv2.INTER_LINEAR)
        else:
            undist = bgr

        gray     = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
        enhanced = self.clahe.apply(gray)
        return enhanced, undist


# ── ORB extractor ─────────────────────────────────────────────────────────────

class ORBExtractor(object):
    """
    Wraps cv2.ORB_create with parameters tuned for Jetson Nano.
    Adds BFMatcher for consecutive-frame match quality logging.
    """

    def __init__(self, n_features=1000, scale_factor=1.2,
                 n_levels=8, fast_thresh=20, lowe_ratio=0.75):

        self.orb = cv2.ORB_create(
            nfeatures    = n_features,
            scaleFactor  = scale_factor,
            nlevels      = n_levels,
            edgeThreshold= 31,
            firstLevel   = 0,
            WTA_K        = 2,
            scoreType    = cv2.ORB_HARRIS_SCORE,
            patchSize    = 31,
            fastThreshold= fast_thresh,
        )
        self.matcher      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.lowe_ratio   = lowe_ratio
        self.n_features   = n_features
        self._empty_desc  = np.empty((0, 32), dtype=np.uint8)

        rospy.loginfo(
            "[ORB] Extractor ready  n_features=%d  nlevels=%d  "
            "fast_thresh=%d  lowe=%.2f",
            n_features, n_levels, fast_thresh, lowe_ratio
        )

    def extract(self, gray_clahe):
        """
        Detect FAST keypoints and compute BRIEF descriptors.
        Returns (keypoints, descriptors) — descriptors is Nx32 uint8.
        """
        kps, descs = self.orb.detectAndCompute(gray_clahe, None)

        if descs is None or len(kps) == 0:
            return [], self._empty_desc

        return list(kps), descs

    def match_ratio(self, desc_a, desc_b):
        """
        knnMatch + Lowe ratio test.
        Returns list of good cv2.DMatch objects.
        Used for consecutive-frame match quality logging only.
        """
        if (desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2):
            return []
        try:
            pairs = self.matcher.knnMatch(desc_a, desc_b, k=2)
        except cv2.error:
            return []

        good = []
        for pair in pairs:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.lowe_ratio * n.distance:
                    good.append(m)
        return good

    def orientation_filter(self, good, kp_a, kp_b, n_bins=30, top_k=3):
        """
        ORB-SLAM3 rotation histogram filter.
        Keeps only matches whose (angle_a - angle_b) delta falls in
        the top-K histogram bins. Removes 20-40% of ratio-test survivors.
        """

        if len(good) < 8:
            return good

        deltas = np.array([
            kp_a[m.queryIdx].angle - kp_b[m.trainIdx].angle
            for m in good
        ])

        hist, bin_edges = np.histogram(deltas, bins=n_bins,
                                       range=(-180.0, 180.0))
        top_idx = np.argsort(hist)[-top_k:]

        filtered = []
        bin_width = 360.0 / n_bins
        for m, delta in zip(good, deltas):
            b = int((delta + 180.0) / bin_width)
            b = max(0, min(b, n_bins - 1))
            if b in top_idx:
                filtered.append(m)

        return filtered if len(filtered) >= 8 else good

    @staticmethod
    def grid_entropy(keypoints, img_h, img_w, grid=4):
        """
        Spatial distribution score — Shannon entropy of keypoint
        distribution over a grid x grid cell layout.
        Max entropy (log2(grid*grid)) = perfectly uniform spread.
        Used as a quality hint passed to the keyframe scorer (step4).
        """
        if len(keypoints) == 0:
            return 0.0

        counts = np.zeros((grid, grid), dtype=np.float32)
        for kp in keypoints:
            r = int(kp.pt[1] / img_h * grid)
            c = int(kp.pt[0] / img_w * grid)
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

class ORBNode(object):
    """
    ROS Melodic node that:
      1. Subscribes to /camera/image_raw
      2. Runs CLAHE + ORB extraction on every frame
      3. Publishes FrameFeatures to /orb/keyframe_candidate
      4. Publishes debug image to /orb/debug_image
      5. Publishes JSON stats to /orb/stats
    """

    def __init__(self):
        rospy.init_node('orb_extractor', anonymous=False)

        # ── Load params ───────────────────────────────────────────────────
        n_features   = rospy.get_param('~n_features',   1000)
        scale_factor = rospy.get_param('~scale_factor', 1.2)
        n_levels     = rospy.get_param('~n_levels',     8)
        fast_thresh  = rospy.get_param('~fast_thresh',  20)
        lowe_ratio   = rospy.get_param('~lowe_ratio',   0.75)
        clip_limit   = rospy.get_param('~clip_limit',   2.0)
        tile_size    = rospy.get_param('~tile_size',    8)
        calib_path   = rospy.get_param('~calib_path',   '')
        self.debug   = rospy.get_param('~debug_viz',    True)
        self.max_hz  = rospy.get_param('~publish_rate', 30)

        # ── Calibration ───────────────────────────────────────────────────
        K, D = load_calibration(calib_path)

        # ── Pipeline objects ──────────────────────────────────────────────
        self.clahe   = ClaheProcessor(K, D,
                                      clip_limit=clip_limit,
                                      tile_size=tile_size)
        self.orb     = ORBExtractor(n_features   = n_features,
                                    scale_factor = scale_factor,
                                    n_levels     = n_levels,
                                    fast_thresh  = fast_thresh,
                                    lowe_ratio   = lowe_ratio)
        self.bridge  = CvBridge()

        # ── State ─────────────────────────────────────────────────────────
        self.prev_kp    = None
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

        rospy.loginfo("[ORB] Node ready. Listening on /camera/image_raw")

    # ── Image callback ────────────────────────────────────────────────────

    def _cb_image(self, msg):
        """
        Called for every incoming camera frame.
        Runs full extraction pipeline and publishes results.
        """
        now = time.time()

        # Rate limiting — skip frame if publishing too fast
        if (now - self.t_last_pub) < self.min_dt:
            return

        # ── Convert ROS Image → OpenCV BGR ────────────────────────────
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            rospy.logerr("[ORB] CvBridge error: %s", str(e))
            return

        h, w = bgr.shape[:2]

        # ── CLAHE preprocessing ────────────────────────────────────────
        t0 = time.time()
        gray_clahe, bgr_undist = self.clahe.process(bgr)
        t_clahe = (time.time() - t0) * 1000.0

        # ── ORB extraction ─────────────────────────────────────────────
        t1 = time.time()
        kps, descs = self.orb.extract(gray_clahe)
        t_orb = (time.time() - t1) * 1000.0

        n_kp = len(kps)

        # ── Consecutive frame match quality (for stats logging) ────────
        n_good_matches = 0
        if self.prev_desc is not None and len(descs) > 0:
            good = self.orb.match_ratio(self.prev_desc, descs)
            good = self.orb.orientation_filter(good, self.prev_kp, kps)
            n_good_matches = len(good)

        # ── Grid entropy (spatial distribution score) ──────────────────
        entropy = self.orb.grid_entropy(kps, h, w, grid=4)

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

        feat_msg.keypoint_x      = [float(kp.pt[0])    for kp in kps]
        feat_msg.keypoint_y      = [float(kp.pt[1])    for kp in kps]
        feat_msg.keypoint_angle  = [float(kp.angle)    for kp in kps]
        feat_msg.keypoint_size   = [float(kp.size)     for kp in kps]
        feat_msg.keypoint_octave = [int(kp.octave & 0xFF) for kp in kps]

        self.pub_feat.publish(feat_msg)

        # ── Publish stats JSON ─────────────────────────────────────────
        stats = {
            'frame'          : self.frame_n,
            'n_keypoints'    : n_kp,
            'n_good_matches' : n_good_matches,
            'entropy'        : round(entropy, 3),
            'clahe_ms'       : round(t_clahe, 2),
            'orb_ms'         : round(t_orb,   2),
            'total_ms'       : round(t_clahe + t_orb, 2),
        }
        self.pub_stats.publish(String(data=json.dumps(stats)))

        # ── Debug visualisation ────────────────────────────────────────
        if self.debug and self.pub_debug.get_num_connections() > 0:
            debug_img = self._make_debug_image(
                bgr_undist, kps, n_kp, n_good_matches,
                entropy, t_clahe + t_orb, h, w
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
                "[ORB] frame=%d  kp=%d  matches=%d  "
                "entropy=%.2f  total=%.1fms",
                self.frame_n, n_kp, n_good_matches,
                entropy, t_clahe + t_orb
            )

        # ── Update state ───────────────────────────────────────────────
        self.prev_kp   = kps
        self.prev_desc = descs
        self.t_last_pub = now
        self.frame_n   += 1

    # ── Debug image builder ────────────────────────────────────────────────

    def _make_debug_image(self, bgr, kps, n_kp, n_matches,
                          entropy, total_ms, img_h, img_w):
        """
        Build debug visualisation:
          - Keypoints coloured by pyramid octave
          - 4x4 grid overlay showing spatial distribution
          - HUD with stats
        """
        out = bgr.copy()

        # Draw keypoints coloured by octave
        for kp in kps:
            oct_idx = min(kp.octave & 0xFF, len(OCTAVE_COLOURS) - 1)
            colour  = OCTAVE_COLOURS[oct_idx]
            centre  = (int(kp.pt[0]), int(kp.pt[1]))
            radius  = max(3, int(kp.size / 2))
            cv2.circle(out, centre, radius, colour, 1)

        # 4x4 grid overlay
        for r in range(1, 4):
            y = r * img_h // 4
            cv2.line(out, (0, y), (img_w, y), (50, 50, 50), 1)
        for c in range(1, 4):
            x = c * img_w // 4
            cv2.line(out, (x, 0), (x, img_h), (50, 50, 50), 1)

        # HUD bar
        cv2.rectangle(out, (0, 0), (img_w, 34), (0, 0, 0), -1)
        colour = (0, 255, 0) if n_kp >= 150 else (0, 80, 220)
        cv2.putText(
            out,
            "kp=%d  matches=%d  entropy=%.2f  %.1fms" % (
                n_kp, n_matches, entropy, total_ms
            ),
            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, colour, 2
        )

        return out

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rospy.loginfo("[ORB] Spinning. Ctrl+C to stop.")
        rospy.spin()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = ORBNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
