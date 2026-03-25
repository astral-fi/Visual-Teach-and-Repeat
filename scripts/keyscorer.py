#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step4_keyframe_scorer_node.py  —  ROS Melodic Keyframe Quality Scorer
VT&R Project | Phase 1

ROS NODE NAME : /keyframe_scorer
SUBSCRIBES    : /orb/keyframe_candidate   (vtr/FrameFeatures)
PUBLISHES     : /keyframe/saved           (vtr/FrameFeatures)  — passed frames
                /keyframe/rejected        (std_msgs/String)    — rejection log
                /keyframe/stats           (std_msgs/String)    — JSON stats

WHAT THIS NODE DOES:
    Receives every FrameFeatures message from step3_orb_node and decides
    whether to save it as a keyframe in the memory graph.

    Three signals are computed and combined into a single quality score:

        Signal 1 — Feature count score
            norm(n_keypoints / MAX_FEATURES)
            A dark or blurry frame has few keypoints — useless in memory.

        Signal 2 — Spatial distribution score (grid entropy)
            Shannon entropy of keypoint distribution over a 4x4 grid.
            300 features clustered in one corner < 150 spread evenly.

        Signal 3 — Scene novelty score
            1 - similarity_to_last_saved_frame
            If the scene is 91% similar to the last saved keyframe,
            the robot has not moved enough — skip this frame.

    Combined score = w1*count + w2*entropy + w3*novelty
    Weights: 0.4, 0.4, 0.2

    Decision:
        score > SAVE_THRESHOLD (0.6)  AND  novelty > MIN_NOVELTY (0.15)
            → SAVE: republish on /keyframe/saved
        score < REJECT_THRESHOLD (0.4) OR count < MIN_FEATURES (150)
            → REJECT: log reason
        Between thresholds
            → BORDERLINE: save only if time since last save > TIMEOUT (2.0s)

PIPELINE POSITION:
    /orb/keyframe_candidate  (step3)
        → this node
        → /keyframe/saved
        → step5_memory_graph_node

PARAMS:
    ~max_features       (int,   default=500)   normalisation ceiling
    ~min_features       (int,   default=150)   hard floor — always reject below
    ~save_threshold     (float, default=0.6)   score to always save
    ~reject_threshold   (float, default=0.4)   score to always reject
    ~min_novelty        (float, default=0.15)  min scene change required
    ~similarity_ratio   (float, default=0.75)  BFMatcher ratio for similarity
    ~borderline_timeout (float, default=2.0)   seconds before forced save
    ~weights            (str,   default='0.4,0.4,0.2')  w1,w2,w3
=============================================================================
"""

import rospy
import numpy as np
import cv2
import json
import time

from std_msgs.msg import String
from vtr.msg      import FrameFeatures


# ── Configuration defaults ────────────────────────────────────────────────────

MAX_FEATURES       = 500
MIN_FEATURES       = 150
SAVE_THRESHOLD     = 0.6
REJECT_THRESHOLD   = 0.4
MIN_NOVELTY        = 0.15
SIMILARITY_RATIO   = 0.75
BORDERLINE_TIMEOUT = 2.0
WEIGHTS            = (0.4, 0.4, 0.2)


# ── Scoring logic (pure functions — easy to unit test) ────────────────────────

def score_feature_count(n_kp, max_features=MAX_FEATURES):
    """
    Signal 1 — how many ORB keypoints were found.
    Normalised to [0, 1]. Capped at max_features.

    n_kp=0   → 0.0  (dark/blank frame)
    n_kp=250 → 0.5
    n_kp=500 → 1.0  (rich textured scene)
    """
    return float(min(n_kp, max_features)) / float(max_features)


def score_entropy(keypoint_x, keypoint_y, img_h, img_w, grid=4):
    """
    Signal 2 — spatial distribution of keypoints over a grid x grid layout.
    Uses Shannon entropy: H = -sum(p * log2(p))

    Max entropy = log2(grid*grid) = 4.0 bits for 4x4 grid.
    Normalised to [0, 1] by dividing by log2(grid*grid).

    All keypoints in one cell → entropy ≈ 0.0
    Perfectly spread          → entropy = 1.0
    """
    if len(keypoint_x) == 0:
        return 0.0

    counts = np.zeros((grid, grid), dtype=np.float32)
    h_f    = float(img_h)
    w_f    = float(img_w)

    for x, y in zip(keypoint_x, keypoint_y):
        r = int(y / h_f * grid)
        c = int(x / w_f * grid)
        r = max(0, min(r, grid - 1))
        c = max(0, min(c, grid - 1))
        counts[r, c] += 1.0

    total = counts.sum()
    if total == 0:
        return 0.0

    probs   = counts.flatten() / total
    probs   = probs[probs > 0]
    entropy = float(-np.sum(probs * np.log2(probs)))
    max_entropy = np.log2(grid * grid)   # 4.0 for 4x4

    return float(entropy / max_entropy)


def score_novelty(desc_curr, desc_prev,
                  kp_angle_curr, kp_angle_prev,
                  ratio=SIMILARITY_RATIO):
    """
    Signal 3 — how different is this frame from the last saved keyframe.

    Runs BFMatcher + ratio test between current and previous saved
    descriptors. Returns 1 - match_ratio where match_ratio is the
    fraction of current descriptors that found a good match.

    High similarity (match_ratio=0.9) → novelty=0.1 → do NOT save
    Low similarity  (match_ratio=0.2) → novelty=0.8 → good candidate

    Returns:
        novelty_score : float 0-1
        match_ratio   : float 0-1  (for logging)
    """
    if desc_prev is None or len(desc_prev) < 4 or len(desc_curr) < 4:
        # No previous frame to compare against — treat as novel
        return 1.0, 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        pairs = matcher.knnMatch(desc_curr, desc_prev, k=2)
    except cv2.error:
        return 1.0, 0.0

    good = []
    for pair in pairs:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)

    # Orientation histogram filter on similarity matches
    if len(good) > 8 and kp_angle_curr and kp_angle_prev:
        good = _orientation_filter(good, kp_angle_curr, kp_angle_prev)

    match_ratio   = float(len(good)) / float(len(desc_curr))
    novelty_score = float(1.0 - match_ratio)

    return novelty_score, match_ratio


def _orientation_filter(good, angles_a, angles_b, n_bins=30, top_k=3):
    """Rotation histogram filter — keeps consistent rotation delta matches."""
    if len(good) < 8:
        return good

    deltas = np.array([
        angles_a[m.queryIdx] - angles_b[m.trainIdx]
        for m in good
        if m.queryIdx < len(angles_a) and m.trainIdx < len(angles_b)
    ])
    if len(deltas) == 0:
        return good

    hist, _ = np.histogram(deltas, bins=n_bins, range=(-180.0, 180.0))
    top_idx = set(np.argsort(hist)[-top_k:].tolist())
    bin_w   = 360.0 / n_bins

    filtered = []
    for m, delta in zip(good, deltas):
        b = int((delta + 180.0) / bin_w)
        b = max(0, min(b, n_bins - 1))
        if b in top_idx:
            filtered.append(m)

    return filtered if len(filtered) >= 6 else good


def combined_score(count_score, entropy_score, novelty_score,
                   weights=WEIGHTS):
    """
    Weighted sum of three signals.

    weights = (w_count, w_entropy, w_novelty)
    All weights must sum to 1.0.

    Default: 0.4, 0.4, 0.2
    Count and entropy weighted equally — both needed for reliable matching.
    Novelty weighted less — we have the explicit MIN_NOVELTY guard.
    """
    w1, w2, w3 = weights
    return w1 * count_score + w2 * entropy_score + w3 * novelty_score


# ── ROS node ──────────────────────────────────────────────────────────────────

class KeyframeScorerNode(object):
    """
    Subscribes to /orb/keyframe_candidate.
    Scores each frame on three signals.
    Publishes accepted frames to /keyframe/saved.
    """

    def __init__(self):
        rospy.init_node('keyframe_scorer', anonymous=False)

        # ── Load params ───────────────────────────────────────────────────
        self.max_features    = rospy.get_param('~max_features',       MAX_FEATURES)
        self.min_features    = rospy.get_param('~min_features',       MIN_FEATURES)
        self.save_thresh     = rospy.get_param('~save_threshold',     SAVE_THRESHOLD)
        self.reject_thresh   = rospy.get_param('~reject_threshold',   REJECT_THRESHOLD)
        self.min_novelty     = rospy.get_param('~min_novelty',        MIN_NOVELTY)
        self.sim_ratio       = rospy.get_param('~similarity_ratio',   SIMILARITY_RATIO)
        self.borderline_to   = rospy.get_param('~borderline_timeout', BORDERLINE_TIMEOUT)

        w_str   = rospy.get_param('~weights', '0.4,0.4,0.2')
        parts   = [float(x.strip()) for x in w_str.split(',')]
        self.weights = tuple(parts) if len(parts) == 3 else WEIGHTS

        # ── State ─────────────────────────────────────────────────────────
        self.prev_desc         = None   # descriptors of last SAVED frame
        self.prev_kp_angles    = None   # angles of last saved keypoints
        self.t_last_saved      = 0.0    # unix time of last saved frame
        self.n_received        = 0
        self.n_saved           = 0
        self.n_rejected        = 0
        self.n_borderline      = 0

        # Image dimensions — inferred from first message
        self.img_h = 480
        self.img_w = 640

        # ── Subscribers ───────────────────────────────────────────────────
        self.sub = rospy.Subscriber(
            '/orb/keyframe_candidate', FrameFeatures,
            self._cb_candidate, queue_size=5
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_saved    = rospy.Publisher(
            '/keyframe/saved', FrameFeatures, queue_size=10
        )
        self.pub_rejected = rospy.Publisher(
            '/keyframe/rejected', String, queue_size=10
        )
        self.pub_stats    = rospy.Publisher(
            '/keyframe/stats', String, queue_size=5
        )

        rospy.loginfo("[SCORER] Node ready.")
        rospy.loginfo(
            "[SCORER] Thresholds: save=%.2f  reject=%.2f  "
            "min_features=%d  min_novelty=%.2f",
            self.save_thresh, self.reject_thresh,
            self.min_features, self.min_novelty
        )
        rospy.loginfo("[SCORER] Weights: count=%.2f  entropy=%.2f  novelty=%.2f",
                      *self.weights)

    # ── Callback ──────────────────────────────────────────────────────────

    def _cb_candidate(self, msg):
        """
        Main scoring callback. Called for every FrameFeatures message.
        """
        self.n_received += 1
        now = time.time()

        n_kp = msg.n_keypoints

        # ── Reconstruct descriptors and keypoint angles ────────────────
        if len(msg.descriptors_flat) > 0:
            desc = np.frombuffer(msg.descriptors_flat,
                            dtype=np.uint8).reshape(-1, 32)
        else:
            desc = None

        kp_angles = list(msg.keypoint_angle)

        # ── Hard floor: reject immediately if too few features ─────────
        if n_kp < self.min_features or desc is None:
            self._reject(msg, 'too_few_features', n_kp,
                         0.0, 0.0, 0.0, 0.0, now)
            return

        # ── Signal 1: feature count ────────────────────────────────────
        s_count = score_feature_count(n_kp, self.max_features)

        # ── Signal 2: spatial entropy ──────────────────────────────────
        s_entropy = score_entropy(
            msg.keypoint_x, msg.keypoint_y,
            self.img_h, self.img_w
        )

        # ── Signal 3: novelty vs last saved frame ──────────────────────
        s_novelty, match_ratio = score_novelty(
            desc, self.prev_desc,
            kp_angles, self.prev_kp_angles,
            ratio=self.sim_ratio
        )

        # ── Combined score ─────────────────────────────────────────────
        score = combined_score(s_count, s_entropy, s_novelty, self.weights)

        # ── Decision logic ─────────────────────────────────────────────
        time_since_save = now - self.t_last_saved
        decision        = self._decide(score, s_novelty, time_since_save)

        if decision == 'save':
            self._save(msg, desc, kp_angles,
                       score, s_count, s_entropy, s_novelty,
                       match_ratio, now)

        elif decision == 'borderline_save':
            self._save(msg, desc, kp_angles,
                       score, s_count, s_entropy, s_novelty,
                       match_ratio, now, borderline=True)

        else:
            self._reject(msg, decision,
                         n_kp, score, s_count, s_entropy, s_novelty, now)

    # ── Decision ──────────────────────────────────────────────────────────

    def _decide(self, score, novelty, time_since_save):
        """
        Returns decision string:
            'save'             — clear pass
            'borderline_save'  — marginal but time safety net triggered
            'low_score'        — score below reject threshold
            'low_novelty'      — too similar to last saved frame
            'borderline_hold'  — borderline but not enough time elapsed
        """
        # Always reject below floor
        if score < self.reject_thresh:
            return 'low_score'

        # Reject if scene hasn't changed enough
        if novelty < self.min_novelty:
            return 'low_novelty'

        # Clear save
        if score >= self.save_thresh:
            return 'save'

        # Borderline zone (reject_thresh <= score < save_thresh)
        if time_since_save >= self.borderline_to:
            return 'borderline_save'
        else:
            return 'borderline_hold'

    # ── Save ──────────────────────────────────────────────────────────────

    def _save(self, msg, desc, kp_angles,
              score, s_count, s_entropy, s_novelty,
              match_ratio, now, borderline=False):
        """Publish frame to /keyframe/saved and update state."""

        # Republish the original FrameFeatures message downstream
        self.pub_saved.publish(msg)

        # Update state
        self.prev_desc      = desc
        self.prev_kp_angles = kp_angles
        self.t_last_saved   = now
        self.n_saved       += 1

        if borderline:
            self.n_borderline += 1

        label = 'borderline_save' if borderline else 'save'

        # Publish stats
        stats = {
            'decision'      : label,
            'frame_ts'      : msg.timestamp,
            'n_keypoints'   : msg.n_keypoints,
            'score'         : round(score,     3),
            's_count'       : round(s_count,   3),
            's_entropy'     : round(s_entropy, 3),
            's_novelty'     : round(s_novelty, 3),
            'match_ratio'   : round(match_ratio, 3),
            'n_saved_total' : self.n_saved,
            'save_rate'     : round(
                float(self.n_saved) / max(self.n_received, 1), 3
            ),
        }
        self.pub_stats.publish(String(data=json.dumps(stats)))

        if self.n_saved % 10 == 0:
            rospy.loginfo(
                "[SCORER] Saved #%d  score=%.2f (cnt=%.2f ent=%.2f nov=%.2f)"
                "  kp=%d  %s",
                self.n_saved, score,
                s_count, s_entropy, s_novelty,
                msg.n_keypoints,
                '(borderline)' if borderline else ''
            )

    # ── Reject ────────────────────────────────────────────────────────────

    def _reject(self, msg, reason,
                n_kp, score, s_count, s_entropy, s_novelty, now):
        """Log rejection and publish to /keyframe/rejected."""
        self.n_rejected += 1

        info = {
            'decision'    : 'reject',
            'reason'      : reason,
            'frame_ts'    : msg.timestamp,
            'n_keypoints' : n_kp,
            'score'       : round(score,     3),
            's_count'     : round(s_count,   3),
            's_entropy'   : round(s_entropy, 3),
            's_novelty'   : round(s_novelty, 3),
        }
        self.pub_rejected.publish(String(data=json.dumps(info)))

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rate = rospy.Rate(1)  # 1 Hz summary log
        while not rospy.is_shutdown():
            if self.n_received > 0:
                rospy.loginfo(
                    "[SCORER] received=%d  saved=%d  rejected=%d  "
                    "borderline=%d  save_rate=%.1f%%",
                    self.n_received, self.n_saved,
                    self.n_rejected, self.n_borderline,
                    100.0 * self.n_saved / self.n_received
                )
            rate.sleep()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = KeyframeScorerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
