#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
fusion_node.py  —  VO-primary / ORB-checkpoint Fusion Node
VT&R Project | VO-primary, VTR-assisted Architecture

ROS NODE NAME : /fusion_node
SUBSCRIBES    : /vtr/orb_match_score  (std_msgs/Float32)   — from ORB matcher
                /vtr/live_flow        (geometry_msgs/Vector3) — from lk_flow_node
PUBLISHES     : /vtr/fusion_state     (std_msgs/String)
                    "vo_running"       — optical flow driving, no snap event
                    "checkpoint_snap"  — ORB confirmed a keyframe; reset drift

WHAT THIS NODE DOES:
    Acts as an arbitration layer between the LK optical flow backbone and
    the occasional ORB keyframe checkpoint matcher.

    Normal operation:
        Each frame the flow controller drives the robot using accumulated
        flow error (published as "vo_running").

    Checkpoint snap:
        When the incoming ORB match score exceeds ORB_THRESHOLD AND the
        waypoint index reported tracks the expected keyframe, the node
        publishes "checkpoint_snap".  The repeat controller reacts by
        resetting its accumulated drift accumulator and snapping position
        to the confirmed keyframe.

    The ORB_THRESHOLD is exposed as a live ROS param so it can be tuned
    with `rosparam set /fusion_node/orb_threshold 0.85` without restarting.

PARAMS:
    ~orb_threshold      (float, default=0.75)   Min match score to trigger snap
    ~expected_wp_index  (int,   default=-1)      Expected waypoint (-1 = any)
    ~snap_cooldown_s    (float, default=1.0)     Min seconds between snaps
=============================================================================
"""

import rospy
import time

from std_msgs.msg      import Float32, String
from geometry_msgs.msg import Vector3


class FusionNode(object):
    """
    Arbitrates between VO backbone and ORB checkpoint snapping.

    The ORB match score is compared against ORB_THRESHOLD each time a
    new score arrives.  If the score is high enough (and the cooldown has
    expired), a "checkpoint_snap" event is published; otherwise
    "vo_running" is published on every flow tick so the repeat controller
    knows the system is alive.
    """

    def __init__(self):
        rospy.init_node('fusion_node', anonymous=False)

        # ── Params ────────────────────────────────────────────────────────
        # Note: orb_threshold is read every time it's needed so live
        # rosparam changes take effect without a restart.
        self.orb_threshold   = rospy.get_param('~orb_threshold',    0.75)
        self.expected_wp_idx = rospy.get_param('~expected_wp_index', -1)
        self.snap_cooldown   = rospy.get_param('~snap_cooldown_s',   1.0)

        # ── State ─────────────────────────────────────────────────────────
        self.last_orb_score  = 0.0    # most recent ORB match score
        self.last_flow       = None   # most recent flow Vector3
        self.t_last_snap     = 0.0    # timestamp of last checkpoint_snap event
        self.drift_accum     = 0.0    # accumulated drift magnitude (for logging)
        self.n_snaps         = 0      # total checkpoint snap events this run

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_state = rospy.Publisher(
            '/vtr/fusion_state', String, queue_size=5)

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber('/vtr/orb_match_score', Float32,
                         self._cb_orb_score, queue_size=5)
        rospy.Subscriber('/vtr/live_flow', Vector3,
                         self._cb_live_flow, queue_size=1)

        rospy.loginfo("[FUSION] Node initialised.")
        rospy.loginfo(
            "[FUSION] orb_threshold=%.2f  expected_wp=%d  cooldown=%.1fs",
            self.orb_threshold, self.expected_wp_idx, self.snap_cooldown
        )
        rospy.loginfo(
            "[FUSION] Live-tune threshold: "
            "rosparam set /fusion_node/orb_threshold <value>"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_orb_score(self, msg):
        """
        Receive a new ORB match score from the ORB matcher node.
        Evaluate whether it constitutes a checkpoint snap event.
        """
        self.last_orb_score = msg.data

        # Re-read threshold live so rosparam tweaks are instant
        threshold = rospy.get_param('~orb_threshold', self.orb_threshold)

        now = time.time()
        cooldown_elapsed = (now - self.t_last_snap) >= self.snap_cooldown

        if self.last_orb_score >= threshold and cooldown_elapsed:
            # ── Checkpoint snap ──────────────────────────────────────────
            self.t_last_snap  = now
            self.n_snaps     += 1

            rospy.loginfo(
                "[FUSION] *** CHECKPOINT SNAP *** score=%.3f threshold=%.3f "
                "waypoint=%d  (total snaps: %d  drift_before_snap=%.2f)",
                self.last_orb_score,
                threshold,
                self.expected_wp_idx,
                self.n_snaps,
                self.drift_accum
            )

            # Reset drift accumulator after snap
            self.drift_accum = 0.0

            self.pub_state.publish(String(data='checkpoint_snap'))
        else:
            # ── Normal VO running ────────────────────────────────────────
            # Publish here too so the score topic drives state even without flow
            self.pub_state.publish(String(data='vo_running'))

    def _cb_live_flow(self, msg):
        """
        Receive live optical flow from lk_flow_node.
        Accumulate drift magnitude and publish 'vo_running' heartbeat.

        The drift accumulator tracks how far the robot has deviated since the
        last checkpoint snap — useful for logging and future threshold tuning.
        """
        self.last_flow = msg

        # Accumulate Euclidean magnitude of lateral + forward displacement
        displacement_mag = (msg.x ** 2 + msg.y ** 2) ** 0.5
        self.drift_accum += displacement_mag

        # Always publish 'vo_running' on each flow tick — this is the normal
        # heartbeat that lets the repeat controller know VO is active.
        # Do NOT publish 'checkpoint_snap' here; that is triggered only by ORB.
        self.pub_state.publish(String(data='vo_running'))

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rate = rospy.Rate(1)   # 1 Hz heartbeat log
        while not rospy.is_shutdown():
            # Periodic status log so operator can see fusion health
            rospy.logdebug(
                "[FUSION] orb_score=%.3f drift_accum=%.2f n_snaps=%d",
                self.last_orb_score, self.drift_accum, self.n_snaps
            )
            rate.sleep()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = FusionNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
