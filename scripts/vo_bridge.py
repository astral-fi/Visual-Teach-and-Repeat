#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step11_visual_odometry_bridge_node.py  —  Visual Odometry → EKF Bridge
VT&R Project | Phase 3

ROS NODE NAME : /visual_odometry_bridge
SUBSCRIBES    : /geometry/result    (std_msgs/String)  — from step7
PUBLISHES     : /visual_odom        (nav_msgs/Odometry) — for robot_localization

WHAT THIS NODE DOES:
    The robot_localization EKF node expects sensor data as standard
    ROS message types. The geometry engine publishes a custom JSON string.
    This bridge converts between them.

    From geometry/result JSON:
        lateral      → pose.pose.position.y   (lateral displacement)
        yaw          → pose.pose.orientation  (quaternion from yaw angle)
        confidence   → pose covariance diagonal (scales R matrix dynamically)

    Uncertainty-aware covariance:
        High confidence (>0.7) → small covariance → EKF trusts camera heavily
        Low  confidence (<0.4) → large covariance → EKF trusts IMU prediction

PARAMS:
    ~frame_id         (str,   default='odom')       parent frame
    ~child_frame_id   (str,   default='base_link')  robot frame
    ~base_covariance  (float, default=0.05)         baseline lateral covariance
=============================================================================
"""

import rospy
import json
import math

from std_msgs.msg  import String
from nav_msgs.msg  import Odometry
from geometry_msgs.msg import Quaternion


def yaw_to_quaternion(yaw):
    """Convert yaw angle (radians) to geometry_msgs/Quaternion."""
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


class VisualOdometryBridgeNode(object):

    def __init__(self):
        rospy.init_node('visual_odometry_bridge', anonymous=False)

        self.frame_id       = rospy.get_param('~frame_id',       'odom')
        self.child_frame_id = rospy.get_param('~child_frame_id', 'base_link')
        self.base_cov       = rospy.get_param('~base_covariance', 0.05)

        # Accumulated position estimate
        # We integrate lateral displacement over time
        # Forward position is estimated from forward speed
        self.pos_x = 0.0    # forward (along path)
        self.pos_y = 0.0    # lateral (cross-track error)
        self.yaw   = 0.0

        self.sub = rospy.Subscriber(
            '/geometry/result', String,
            self._cb_result, queue_size=1
        )
        self.pub = rospy.Publisher(
            '/visual_odom', Odometry, queue_size=10
        )

        rospy.loginfo("[VODOM] Bridge ready → /visual_odom")

    def _cb_result(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return

        if not data.get('success', False):
            return

        lateral     = data.get('lateral',    0.0)
        yaw         = data.get('yaw_deg',    0.0) * math.pi / 180.0
        confidence  = data.get('confidence', 0.0)
        method      = data.get('method',     'none')

        # Dynamic covariance — scales inversely with confidence
        if confidence > 0.7:
            cov_scale = 1.0
        elif confidence > 0.4:
            cov_scale = 5.0
        else:
            cov_scale = 50.0   # high uncertainty → EKF trusts IMU

        # LK fallback is less accurate than RANSAC
        if method == 'lk':
            cov_scale *= 3.0

        lateral_cov = self.base_cov * cov_scale
        yaw_cov     = 0.01  * cov_scale

        # Update integrated position
        self.pos_y = lateral   # lateral is absolute cross-track, not delta
        self.yaw   = yaw

        # Build Odometry message
        odom = Odometry()
        odom.header.stamp    = rospy.Time.now()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id  = self.child_frame_id

        # Pose
        odom.pose.pose.position.x  = self.pos_x
        odom.pose.pose.position.y  = self.pos_y
        odom.pose.pose.position.z  = 0.0
        odom.pose.pose.orientation = yaw_to_quaternion(self.yaw)

        # Pose covariance (6x6 row-major: x, y, z, roll, pitch, yaw)
        # Only y (lateral) and yaw are meaningful from vision
        # Set others to large values so EKF ignores them
        pc = [0.0] * 36
        pc[0]  = 999.0          # x — not observable from monocular
        pc[7]  = lateral_cov    # y — lateral displacement
        pc[14] = 999.0          # z
        pc[21] = 999.0          # roll
        pc[28] = 999.0          # pitch
        pc[35] = yaw_cov        # yaw
        odom.pose.covariance = pc

        # Twist (velocity) — not computed, set to zero with high covariance
        tc = [0.0] * 36
        for i in range(6):
            tc[i*7] = 999.0
        odom.twist.covariance = tc

        self.pub.publish(odom)

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = VisualOdometryBridgeNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass