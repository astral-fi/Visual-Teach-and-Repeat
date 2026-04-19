#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
lk_flow_node.py  —  Lucas-Kanade Optical Flow ROS Node
VT&R Project | VO-primary, VTR-assisted Architecture

ROS NODE NAME : /lk_flow_node
SUBSCRIBES    : /csi_cam_0/image_raw   (sensor_msgs/Image)
PUBLISHES     : /vtr/live_flow      (geometry_msgs/Vector3)
                    x = median horizontal displacement (dx) in pixels
                    y = median vertical displacement (dy) in pixels
                    z = estimated rotation (dtheta) in radians

WHAT THIS NODE DOES:
    Runs Shi-Tomasi corner detection + Lucas-Kanade pyramidal optical flow
    every camera frame to estimate ego-motion.  The median displacement of
    all tracked corners is published as a Vector3 to /vtr/live_flow.

    Corner re-detection fires automatically when the number of successfully
    tracked points drops below the re-detect threshold (default 20).

PARAMS:
    ~max_corners     (int,   default=200)    Shi-Tomasi max corners to detect
    ~quality_level   (float, default=0.01)   Shi-Tomasi minimum corner quality
    ~min_dist        (float, default=10.0)   Min pixel distance between corners
    ~win_size        (int,   default=21)     LK window side length (pixels)
    ~max_level       (int,   default=3)      LK pyramid levels
    ~redetect_thresh (int,   default=20)     Min tracked points before re-detect
=============================================================================
"""

import rospy
import cv2
import numpy as np

from sensor_msgs.msg    import Image
from geometry_msgs.msg  import Vector3
from cv_bridge          import CvBridge, CvBridgeError


class LKFlowNode(object):
    """
    Standalone Lucas-Kanade optical flow node.
    Tracks Shi-Tomasi corners across consecutive frames and publishes
    the median displacement as /vtr/live_flow (geometry_msgs/Vector3).
    """

    def __init__(self):
        rospy.init_node('lk_flow_node', anonymous=False)

        # ── Tuning params (all exposed for live rosparam tuning) ──────────
        self.max_corners     = rospy.get_param('~max_corners',     200)
        self.quality_level   = rospy.get_param('~quality_level',   0.01)
        self.min_dist        = rospy.get_param('~min_dist',        10.0)
        self.win_size        = rospy.get_param('~win_size',        21)
        self.max_level       = rospy.get_param('~max_level',       3)
        self.redetect_thresh = rospy.get_param('~redetect_thresh', 20)

        # ── Internal state ────────────────────────────────────────────────
        self.bridge          = CvBridge()
        self.prev_gray       = None   # previous greyscale frame
        self.prev_pts        = None   # tracked corner points from last frame

        # Pre-build the LK termination criteria once (avoid re-alloc per frame)
        # Stop when max 30 iterations OR movement < 0.01 px
        self.lk_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,      # max iterations
            0.01     # epsilon
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_flow = rospy.Publisher(
            '/vtr/live_flow', Vector3, queue_size=1)

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber('/csi_cam_0/image_raw', Image,
                         self._cb_image, queue_size=1,
                         buff_size=2**24)   # large buffer for raw images

        rospy.loginfo("[LK_FLOW] Node initialised.")
        rospy.loginfo(
            "[LK_FLOW] Params: max_corners=%d  quality=%.3f  min_dist=%.1f  "
            "win=%dx%d  levels=%d  redetect_below=%d",
            self.max_corners, self.quality_level, self.min_dist,
            self.win_size, self.win_size, self.max_level, self.redetect_thresh
        )

    # ── Camera callback ───────────────────────────────────────────────────

    def _cb_image(self, msg):
        """
        Main processing callback — fired at camera frame rate.
        Converts image → greyscale, tracks points with LK, publishes flow.
        """
        # Convert ROS Image to OpenCV greyscale
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError as e:
            rospy.logerr("[LK_FLOW] CvBridge error: %s", str(e))
            return

        # ── First frame or after re-detect: seed point set ────────────────
        if self.prev_gray is None or self.prev_pts is None or \
                len(self.prev_pts) < self.redetect_thresh:
            self._redetect_corners(frame)
            self.prev_gray = frame
            # On first frame there is no optical flow yet — publish zero
            self.pub_flow.publish(Vector3(x=0.0, y=0.0, z=0.0))
            return

        # ── Lucas-Kanade optical flow tracking ───────────────────────────
        # calcOpticalFlowPyrLK returns:
        #   next_pts  — predicted locations in current frame (N,1,2) float32
        #   status    — per-point success flag (N,1) uint8 (1=tracked, 0=lost)
        #   err       — per-point error value  (N,1) float32
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame,
            self.prev_pts,
            None,
            winSize  = (self.win_size, self.win_size),
            maxLevel = self.max_level,
            criteria = self.lk_criteria
        )

        # Keep only successfully tracked points
        if next_pts is not None and status is not None:
            good_mask  = status.ravel() == 1
            good_new   = next_pts[good_mask]   # shape (M,1,2)
            good_old   = self.prev_pts[good_mask]
        else:
            good_new = np.empty((0, 1, 2), dtype=np.float32)
            good_old = np.empty((0, 1, 2), dtype=np.float32)

        n_tracked = len(good_new)

        # ── Compute median displacement ────────────────────────────────────
        if n_tracked >= 2:
            # Reshape from (N,1,2) → (N,2) for arithmetic convenience
            new_2d = good_new.reshape(-1, 2)
            old_2d = good_old.reshape(-1, 2)

            # Per-point displacement
            disp = new_2d - old_2d          # shape (N, 2)

            # Median is robust to outlier tracks (e.g. mismatched features)
            dx = float(np.median(disp[:, 0]))   # horizontal (positive = right)
            dy = float(np.median(disp[:, 1]))   # vertical   (positive = down)

            # Estimate rotation: angular displacement around the image centre.
            # Use the cross-product of displacement vectors relative to the
            # image centroid as a rotation proxy (simplified planar model).
            cx = frame.shape[1] / 2.0
            cy = frame.shape[0] / 2.0
            # Vectors from centroid to old points
            rvec_old = old_2d - np.array([[cx, cy]])
            # Vectors from centroid to new points
            rvec_new = new_2d - np.array([[cx, cy]])
            # 2D cross product gives signed rotation per point
            cross = (rvec_old[:, 0] * rvec_new[:, 1] -
                     rvec_old[:, 1] * rvec_new[:, 0])
            dot   = (rvec_old[:, 0] * rvec_new[:, 0] +
                     rvec_old[:, 1] * rvec_new[:, 1])
            # atan2 of (cross, dot) gives rotation angle per point
            angles = np.arctan2(cross, dot + 1e-9)   # avoid div/0
            dtheta = float(np.median(angles))

        else:
            # Too few points to estimate reliably — publish zeros
            dx, dy, dtheta = 0.0, 0.0, 0.0

        # Publish the flow estimate
        self.pub_flow.publish(Vector3(x=dx, y=dy, z=dtheta))

        # ── Update state for next frame ────────────────────────────────────
        if n_tracked >= self.redetect_thresh:
            # Enough points — keep tracking them
            self.prev_pts  = good_new.reshape(-1, 1, 2)
        else:
            # Too few survived — re-detect on this frame before next callback
            rospy.logdebug(
                "[LK_FLOW] Only %d points tracked — re-detecting corners",
                n_tracked
            )
            self._redetect_corners(frame)

        self.prev_gray = frame

    # ── Corner re-detection ───────────────────────────────────────────────

    def _redetect_corners(self, gray_frame):
        """
        Run Shi-Tomasi corner detection on gray_frame and store results
        in self.prev_pts.  Called on the very first frame and whenever the
        tracked corner count drops below self.redetect_thresh.
        """
        # goodFeaturesToTrack returns array of shape (N,1,2) float32 or None
        pts = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners   = self.max_corners,
            qualityLevel = self.quality_level,
            minDistance  = self.min_dist
        )

        if pts is not None:
            self.prev_pts = pts
            rospy.logdebug("[LK_FLOW] Re-detected %d corners", len(pts))
        else:
            # Could not find any corners — set empty array so we keep trying
            self.prev_pts = np.empty((0, 1, 2), dtype=np.float32)
            rospy.logwarn("[LK_FLOW] goodFeaturesToTrack returned None "
                          "(textureless frame?)")

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rospy.spin()   # callbacks handle all work; no rate loop needed


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = LKFlowNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
