#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
ekf_to_pid_bridge.py  —  EKF Odometry → PID Path Error Bridge
VT&R Project | Phase 3

Extracts lateral and yaw from /odometry/filtered (robot_localization output)
and publishes as a single Float32 path_error for the PID controller.

This is the final link in the Phase 3 chain:
    IMU + Camera → EKF → /odometry/filtered → this node → /ekf_path_error → PID
=============================================================================
"""

import rospy
import math

from nav_msgs.msg import Odometry
from std_msgs.msg import Float32


def quaternion_to_yaw(q):
    """Extract yaw from quaternion."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class EKFToPIDBridge(object):

    def __init__(self):
        rospy.init_node('ekf_pid_bridge', anonymous=False)

        self.w_lateral = rospy.get_param('~w_lateral', 0.7)
        self.w_yaw     = rospy.get_param('~w_yaw',     0.3)

        self.sub = rospy.Subscriber(
            '/odometry/filtered', Odometry,
            self._cb_odom, queue_size=1
        )
        self.pub = rospy.Publisher(
            '/ekf_path_error', Float32, queue_size=1
        )
        rospy.loginfo("[EKF→PID] Bridge ready  w_lat=%.2f  w_yaw=%.2f",
                      self.w_lateral, self.w_yaw)

    def _cb_odom(self, msg):
        lateral = msg.pose.pose.position.y
        yaw     = quaternion_to_yaw(msg.pose.pose.orientation)

        path_error = self.w_lateral * lateral + self.w_yaw * yaw
        self.pub.publish(Float32(data=path_error))

    def spin(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        node = EKFToPIDBridge()
        node.spin()
    except rospy.ROSInterruptException:
        pass