#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step6_teach_logger_node.py  —  ROS Melodic Teach Phase Logger
VT&R Project | Phase 1  (VO-primary, VTR-assisted extension)

ROS NODE NAME : /teach_logger
SUBSCRIBES    : /keyframe/saved       (vtr/FrameFeatures) — from step4
                /graph/node_added     (std_msgs/String)   — from step5
                /graph/status         (std_msgs/String)   — from step5
                /csi_cam_0/image_raw     (sensor_msgs/Image) — NEW: LK flow
PUBLISHES     : /teach/status         (std_msgs/String)   — JSON teach state
                /teach/hud            (sensor_msgs/Image) — live HUD overlay

SERVICES      : /teach/start          (std_srvs/Trigger)  — begin recording
                /teach/stop           (std_srvs/Trigger)  — end + save graph
                /teach/mark_junction  (std_srvs/Trigger)  — Method 2 junction
                /teach/mark_endpoint  (vtr/SetGoal)       — label last node

WHAT THIS NODE DOES:
    Orchestrates the entire Teach phase of the VT&R pipeline.

    NEW (VO-extension): Runs LK optical flow on every raw camera frame in
    parallel with ORB processing. On each keyframe save, accumulated flow
    (dx, dy, dtheta, magnitude) is bundled into a WaypointRecord and
    appended to self.waypoints. On shutdown the full list is pickled to
    waypoints_pkl_path so the Repeat phase can load it.

    1. Listens for gamepad START button (or rosservice call /teach/start)
       to begin a teach run.

    2. Monitors the pipeline, tracks nodes being added to the graph,
       logs teach statistics in real time.

    3. Provides junction marking — pressing the gamepad X button (or
       calling /teach/mark_junction) flags the most recent node as a
       junction for Method 2 multi-route branching.

    4. On STOP (gamepad SELECT or /teach/stop service):
       - Marks the last node as endpoint with the configured label
       - Calls /graph/save to persist the graph to disk
       - Saves waypoints.pkl with all accumulated VO data
       - Prints the full teach summary

    5. Publishes a live HUD image to /teach/hud.

GAMEPAD MAPPING:
    Button 7 (START)  — begin teach
    Button 6 (SELECT) — stop teach + save
    Button 2 (X)      — mark current node as junction
    Button 0 (A)      — mark current node as endpoint with label

LAUNCH:
    roslaunch vtr teach.launch route_id:=route_lab endpoint:="Lab door"

PARAMS:
    ~route_id           (str,  default='route_0')         passed to step5
    ~endpoint_label     (str,  default='')                destination label
    ~save_path          (str,  default='~/vtr_graph')
    ~waypoints_pkl_path (str,  default='~/waypoints.pkl') VO waypoint file
    ~joy_topic          (str,  default='/joy')
    ~hud_width          (int,  default=640)
    ~hud_height         (int,  default=120)
    ~append_mode        (bool, default=False)
    ~wp_save_every_n    (int,  default=1) save waypoint every N keyframes
=============================================================================
"""

import rospy
import numpy as np
import cv2
import json
import os
import time
import pickle                      # NEW — serialize waypoints on shutdown

from std_msgs.msg    import String
from std_srvs.srv    import Trigger, TriggerResponse
from sensor_msgs.msg import Image, Joy
from cv_bridge       import CvBridge, CvBridgeError

from vtr.msg import FrameFeatures
from vtr.srv import SetGoal, SetGoalResponse


# ── WaypointRecord ────────────────────────────────────────────────────────────
# Plain Python class used as a dataclass (Python 2.7 compatible).
# One WaypointRecord is saved per keyframe during the Teach phase.
# The Repeat controller loads the pkl and drives from these records.

class WaypointRecord(object):
    """
    Stores all data captured at a single teach waypoint.

    Fields
    ------
    index          : int   — waypoint index (0-based)
    node_id        : int   — corresponding graph node id from step5
    orb_count      : int   — number of ORB features in this frame
    keypoints_x    : list  — ORB keypoint x-coordinates
    keypoints_y    : list  — ORB keypoint y-coordinates
    descriptors    : ndarray — raw XFeat descriptors (N x 64 float32)
    flow_dx        : float — accumulated median horizontal LK displacement (px)
    flow_dy        : float — accumulated median vertical LK displacement (px)
    flow_dtheta    : float — accumulated median rotation estimate (radians)
    flow_magnitude : float — Euclidean magnitude of (flow_dx, flow_dy)
    steering       : float — recorded steering command at this waypoint
    throttle       : float — recorded throttle command at this waypoint
    """
    __slots__ = [
        'index', 'node_id', 'orb_count',
        'keypoints_x', 'keypoints_y', 'descriptors',
        'flow_dx', 'flow_dy', 'flow_dtheta', 'flow_magnitude',
        'steering', 'throttle'
    ]

    def __init__(self, index, node_id, orb_count,
                 keypoints_x, keypoints_y, descriptors,
                 flow_dx, flow_dy, flow_dtheta, flow_magnitude,
                 steering=0.0, throttle=0.0):
        self.index          = index
        self.node_id        = node_id
        self.orb_count      = orb_count
        self.keypoints_x    = keypoints_x
        self.keypoints_y    = keypoints_y
        self.descriptors    = descriptors
        self.flow_dx        = flow_dx
        self.flow_dy        = flow_dy
        self.flow_dtheta    = flow_dtheta
        self.flow_magnitude = flow_magnitude
        self.steering       = steering
        self.throttle       = throttle

    def __repr__(self):
        return (
            'WaypointRecord(idx={index} node={node_id} orb={orb_count} '
            'dx={flow_dx:.2f} dy={flow_dy:.2f} dth={flow_dtheta:.4f} '
            'mag={flow_magnitude:.2f} steer={steering:.3f})'
        ).format(**{s: getattr(self, s) for s in self.__slots__})


# ── Teach state machine ───────────────────────────────────────────────────────

IDLE      = 'IDLE'
RECORDING = 'RECORDING'
SAVED     = 'SAVED'


# ── HUD colours ───────────────────────────────────────────────────────────────

COL_GREEN  = (0,  200,  80)
COL_RED    = (0,   60, 220)
COL_AMBER  = (0,  160, 220)
COL_WHITE  = (220, 220, 220)
COL_DARK   = (30,   30,  30)
COL_GREY   = (120, 120, 120)


# ── ROS node ──────────────────────────────────────────────────────────────────

class TeachLoggerNode(object):
    """
    Orchestrates the Teach phase and provides the operator interface.
    """

    def __init__(self):
        rospy.init_node('teach_logger', anonymous=False)

        # ── Params ────────────────────────────────────────────────────────
        self.route_id        = rospy.get_param('~route_id',       'route_0')
        self.ep_label        = rospy.get_param('~endpoint_label', '')
        self.save_path       = os.path.expanduser(
            rospy.get_param('~save_path', '~/vtr_graph'))
        # NEW — output path for the pickled waypoints list
        self.waypoints_pkl   = os.path.expanduser(
            rospy.get_param('~waypoints_pkl_path', '~/waypoints.pkl'))
        self.joy_topic       = rospy.get_param('~joy_topic',      '/joy')
        self.hud_w           = rospy.get_param('~hud_width',       640)
        self.hud_h           = rospy.get_param('~hud_height',      120)
        self.append_mode     = rospy.get_param('~append_mode',     False)
        # NEW — save one waypoint every N approved keyframes (default: every one)
        self.wp_save_every_n = rospy.get_param('~wp_save_every_n', 1)

        # ── State ─────────────────────────────────────────────────────────
        self.state         = IDLE
        self.teach_start_t = None
        self.n_frames_recv = 0     # keyframes received from step4
        self.n_nodes_added = 0     # nodes confirmed added to graph (step5)
        self.last_edge     = None  # most recent edge info dict
        self.last_node_id  = -1
        self.junctions     = []    # node IDs marked as junctions
        self.endpoints     = []    # (node_id, label) tuples

        # Gamepad debounce — track last button states
        self._joy_prev       = {}

        # ── NEW — LK optical flow state ───────────────────────────────────
        self._lk_prev_gray   = None   # greyscale frame from previous tick
        self._lk_prev_pts    = None   # tracked corner points (N,1,2) float32

        # Accumulators for the inter-waypoint displacement bucket
        self._acc_dx         = 0.0   # sum of per-frame median x displacement
        self._acc_dy         = 0.0   # sum of per-frame median y displacement
        self._acc_dtheta     = 0.0   # sum of per-frame rotation estimate (rad)

        # LK tuning — same defaults as lk_flow_node to stay consistent
        self._lk_max_corners  = 200
        self._lk_quality      = 0.01
        self._lk_min_dist     = 10.0
        self._lk_win_size     = 21
        self._lk_max_level    = 3
        self._lk_redet_thresh = 20    # re-detect when fewer than this tracked
        self._lk_criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01
        )

        # ── NEW — Waypoint routing state ─────────────────────────────────
        self._last_ff        = None   # most recent FrameFeatures msg
        self.waypoints       = []     # list of WaypointRecord objects
        self._kf_since_wp    = 0     # keyframes elapsed since last waypoint save
        self._cur_steering   = 0.0   # latest joy steering value
        self._cur_throttle   = 0.0   # latest joy throttle value

        self.bridge          = CvBridge()

        # ── Subscribers ───────────────────────────────────────────────────
        # Pipeline feedback
        rospy.Subscriber('/keyframe/saved',  FrameFeatures,
                         self._cb_kf_saved,  queue_size=5)
        rospy.Subscriber('/graph/node_added', String,
                         self._cb_node_added, queue_size=10)
        rospy.Subscriber('/graph/status',     String,
                         self._cb_graph_status, queue_size=2)

        # NEW — raw camera images for LK optical flow tracking
        rospy.Subscriber('/csi_cam_0/image_raw', Image,
                         self._cb_image_raw, queue_size=1,
                         buff_size=2**24)   # large ring buffer for raw frames

        # Gamepad
        rospy.Subscriber(self.joy_topic, Joy,
                         self._cb_joy, queue_size=1)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_status = rospy.Publisher(
            '/teach/status', String, queue_size=5)
        self.pub_hud    = rospy.Publisher(
            '/teach/hud', Image, queue_size=1)

        # ── Services ──────────────────────────────────────────────────────
        rospy.Service('/teach/start',         Trigger,  self._srv_start)
        rospy.Service('/teach/stop',          Trigger,  self._srv_stop)
        rospy.Service('/teach/mark_junction', Trigger,  self._srv_mark_junction)
        rospy.Service('/teach/mark_endpoint', SetGoal,  self._srv_mark_endpoint)

        # ── Proxy to step5 services ───────────────────────────────────────
        rospy.loginfo("[TEACH] Waiting for /graph/save service...")
        rospy.wait_for_service('/graph/save',      timeout=10.0)
        rospy.wait_for_service('/graph/set_goal',  timeout=10.0)
        rospy.wait_for_service('/graph/plan_route',timeout=10.0)

        self._svc_graph_save  = rospy.ServiceProxy('/graph/save',      Trigger)
        self._svc_set_goal    = rospy.ServiceProxy('/graph/set_goal',   SetGoal)
        self._svc_plan_route  = rospy.ServiceProxy('/graph/plan_route', SetGoal)

        rospy.loginfo("[TEACH] Node ready.")
        rospy.loginfo("[TEACH] route=%s  endpoint='%s'  append=%s",
                      self.route_id, self.ep_label, self.append_mode)
        rospy.loginfo("[TEACH] Controls:")
        rospy.loginfo("[TEACH]   rosservice call /teach/start")
        rospy.loginfo("[TEACH]   rosservice call /teach/stop")
        rospy.loginfo("[TEACH]   rosservice call /teach/mark_junction")
        rospy.loginfo("[TEACH]   rosservice call /teach/mark_endpoint "
                      "\"goal_label: 'Lab door'\"")
        rospy.loginfo("[TEACH]   Gamepad: START=begin  SELECT=stop  "
                      "X=junction  A=endpoint")

    # ── NEW — LK optical flow helpers ────────────────────────────────────

    def _cb_image_raw(self, msg):
        """
        NEW — processes every raw camera frame.
        Runs LK optical flow and accumulates displacement into the
        per-waypoint accumulators (_acc_dx, _acc_dy, _acc_dtheta).
        Inactive unless state == RECORDING.
        """
        if self.state != RECORDING:
            return

        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        except CvBridgeError:
            return

        # Seed the point set on the first frame or after deficit
        if (self._lk_prev_gray is None or self._lk_prev_pts is None or
                len(self._lk_prev_pts) < self._lk_redet_thresh):
            self._lk_redetect(gray)
            self._lk_prev_gray = gray
            return

        # Lucas-Kanade pyramidal tracking
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._lk_prev_gray,
            gray,
            self._lk_prev_pts,
            None,
            winSize  = (self._lk_win_size, self._lk_win_size),
            maxLevel = self._lk_max_level,
            criteria = self._lk_criteria
        )

        # Filter to successfully tracked corners
        if next_pts is not None and status is not None:
            mask     = status.ravel() == 1
            good_new = next_pts[mask]
            good_old = self._lk_prev_pts[mask]
        else:
            good_new = np.empty((0, 1, 2), dtype=np.float32)
            good_old = np.empty((0, 1, 2), dtype=np.float32)

        n_tracked = len(good_new)

        if n_tracked >= 2:
            new2d = good_new.reshape(-1, 2)
            old2d = good_old.reshape(-1, 2)
            disp  = new2d - old2d

            # Median is robust to outlier tracks
            frame_dx = float(np.median(disp[:, 0]))
            frame_dy = float(np.median(disp[:, 1]))

            # Rotation estimate: angle between vectors from image centroid
            cx = gray.shape[1] / 2.0
            cy = gray.shape[0] / 2.0
            r_old  = old2d - np.array([[cx, cy]])
            r_new  = new2d - np.array([[cx, cy]])
            cross  = r_old[:, 0]*r_new[:, 1] - r_old[:, 1]*r_new[:, 0]
            dot    = r_old[:, 0]*r_new[:, 0] + r_old[:, 1]*r_new[:, 1]
            angles = np.arctan2(cross, dot + 1e-9)   # avoid div/0
            frame_dtheta = float(np.median(angles))
        else:
            frame_dx = frame_dy = frame_dtheta = 0.0

        # Accumulate into the inter-waypoint bucket
        self._acc_dx     += frame_dx
        self._acc_dy     += frame_dy
        self._acc_dtheta += frame_dtheta

        # Update tracking state for next tick
        if n_tracked >= self._lk_redet_thresh:
            self._lk_prev_pts = good_new.reshape(-1, 1, 2)
        else:
            self._lk_redetect(gray)   # too few survived — re-seed

        self._lk_prev_gray = gray

    def _lk_redetect(self, gray_frame):
        """
        NEW — Run Shi-Tomasi corner detection and reset self._lk_prev_pts.
        Called at the start of a new waypoint window and when tracked
        corner count drops below self._lk_redet_thresh.
        """
        pts = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners   = self._lk_max_corners,
            qualityLevel = self._lk_quality,
            minDistance  = self._lk_min_dist
        )
        self._lk_prev_pts = pts if pts is not None else \
            np.empty((0, 1, 2), dtype=np.float32)

    def _save_waypoint(self, ff_msg):
        """
        NEW — Snapshot the current accumulated flow + ORB data as a
        WaypointRecord and append it to self.waypoints.

        Resets the flow accumulators and triggers a corner re-detect so
        the next waypoint window starts with a clean feature set.

        Parameters
        ----------
        ff_msg : FrameFeatures  — approved keyframe from step4
        """
        idx = len(self.waypoints)

        # Euclidean magnitude of accumulated displacement
        flow_mag = (self._acc_dx**2 + self._acc_dy**2)**0.5

        # Unpack ORB data from the FrameFeatures message
        if ff_msg is not None:
            kp_x  = list(ff_msg.keypoint_x)
            kp_y  = list(ff_msg.keypoint_y)
            descs = (np.array(ff_msg.descriptors_flat, dtype=np.float32)
                     .reshape(-1, 64)   # XFeat: 64-dim float32 descriptor
                     if len(ff_msg.descriptors_flat) > 0
                     else np.empty((0, 64), dtype=np.float32))
            orb_n = int(ff_msg.n_keypoints)
        else:
            kp_x, kp_y   = [], []
            descs         = np.empty((0, 64), dtype=np.float32)
            orb_n         = 0

        wp = WaypointRecord(
            index          = idx,
            node_id        = self.last_node_id,
            orb_count      = orb_n,
            keypoints_x    = kp_x,
            keypoints_y    = kp_y,
            descriptors    = descs,
            flow_dx        = self._acc_dx,
            flow_dy        = self._acc_dy,
            flow_dtheta    = self._acc_dtheta,
            flow_magnitude = flow_mag,
            steering       = self._cur_steering,
            throttle       = self._cur_throttle,
        )
        self.waypoints.append(wp)

        # Log the saved waypoint for easy monitoring
        rospy.loginfo(
            '[TEACH] Waypoint %d saved: node=%d orb=%d '
            'dx=%.2f dy=%.2f dtheta=%.4f mag=%.2f steer=%.3f',
            idx, self.last_node_id, orb_n,
            self._acc_dx, self._acc_dy,
            self._acc_dtheta, flow_mag, self._cur_steering
        )

        # Reset accumulators for the next waypoint window
        self._acc_dx     = 0.0
        self._acc_dy     = 0.0
        self._acc_dtheta = 0.0

        # Re-detect corners so the next window starts with fresh features
        if self._lk_prev_gray is not None:
            self._lk_redetect(self._lk_prev_gray)

    def _dump_waypoints(self):
        """
        NEW — Serialize self.waypoints to disk as a pickle file.
        Registered as a rospy.on_shutdown handler so it fires on Ctrl-C
        or normal shutdown.
        """
        if not self.waypoints:
            rospy.logwarn('[TEACH] No waypoints recorded — skipping pickle.')
            return
        try:
            with open(self.waypoints_pkl, 'wb') as f:
                # protocol=2 ensures Python 2 compatibility
                pickle.dump(self.waypoints, f, protocol=2)
            rospy.loginfo(
                '[TEACH] Saved %d waypoints to %s',
                len(self.waypoints), self.waypoints_pkl
            )
        except Exception as exc:
            rospy.logerr('[TEACH] Failed to save waypoints: %s', str(exc))

    # ── Pipeline callbacks ────────────────────────────────────────────────

    def _cb_kf_saved(self, msg):
        """
        Count keyframes approved by step4 scorer.
        NEW — cache the FrameFeatures and trigger waypoint save logic.
        """
        if self.state != RECORDING:
            return

        self.n_frames_recv += 1
        self._last_ff       = msg   # cache for the next waypoint record

        # Save a WaypointRecord every wp_save_every_n approved keyframes
        self._kf_since_wp += 1
        if self._kf_since_wp >= self.wp_save_every_n:
            self._kf_since_wp = 0
            self._save_waypoint(self._last_ff)

    def _cb_node_added(self, msg):
        """Track nodes confirmed added to the graph by step5."""
        if self.state != RECORDING:
            return
        try:
            data = json.loads(msg.data)
        except ValueError:
            return

        self.n_nodes_added = data.get('total_nodes', self.n_nodes_added)
        self.last_node_id  = data.get('node_id', self.last_node_id)

        # Cache last edge info for HUD
        edge = data.get('edge')
        if edge is not None:
            self.last_edge = edge

        # Log every 10 nodes
        if self.n_nodes_added % 10 == 0 and self.n_nodes_added > 0:
            rospy.loginfo(
                "[TEACH] %d nodes  edge_inliers=%s  conf=%s",
                self.n_nodes_added,
                str(data.get('edge_inliers', 'N/A')),
                str(data.get('edge_conf',    'N/A'))
            )

    def _cb_graph_status(self, msg):
        """Receive graph heartbeat — no action needed, just confirms step5 alive."""
        pass

    # ── Gamepad ───────────────────────────────────────────────────────────

    def _cb_joy(self, msg):
        """
        Parse gamepad button events.
        Uses rising-edge detection to avoid repeated triggers.

        Button mapping (standard Bluetooth gamepad):
            0 = A        → mark endpoint
            2 = X        → mark junction
            6 = SELECT   → stop + save
            7 = START    → start teach
        """
        buttons = list(msg.buttons)

        def rising(idx):
            """True on button press (0→1 transition)."""
            if idx >= len(buttons):
                return False
            prev = self._joy_prev.get(idx, 0)
            curr = buttons[idx]
            return curr == 1 and prev == 0

        if rising(10):   # START
            self._do_start()
        if rising(11):   # SELECT
            self._do_stop()
        if rising(2):   # X — mark junction
            self._do_mark_junction()
        if rising(1):   # A — mark endpoint with configured label
            if self.ep_label:
                self._do_mark_endpoint(self.ep_label)

        # Update previous state
        for i, b in enumerate(buttons):
            self._joy_prev[i] = b

    # ── Core actions ──────────────────────────────────────────────────────

    def _do_start(self):
        if self.state == RECORDING:
            rospy.logwarn("[TEACH] Already recording.")
            return
        self.state         = RECORDING
        self.teach_start_t = time.time()
        self.n_frames_recv = 0
        self.n_nodes_added = 0
        self.last_edge     = None
        self.last_node_id  = -1
        self.junctions     = []
        self.endpoints     = []

        # NEW — reset VO state for this new recording run
        self._acc_dx       = 0.0
        self._acc_dy       = 0.0
        self._acc_dtheta   = 0.0
        self._kf_since_wp  = 0
        self.waypoints     = []
        self._lk_prev_gray = None
        self._lk_prev_pts  = None

        rospy.loginfo("[TEACH] *** RECORDING STARTED ***  route=%s",
                      self.route_id)

    def _do_stop(self):
        if self.state != RECORDING:
            rospy.logwarn("[TEACH] Not recording — nothing to stop.")
            return

        rospy.loginfo("[TEACH] Stopping teach run...")

        # Mark endpoint if label set and no explicit endpoint marked yet
        if self.ep_label and not self.endpoints:
            self._do_mark_endpoint(self.ep_label)

        # Call step5 graph save service
        try:
            resp = self._svc_graph_save()
            if resp.success:
                rospy.loginfo("[TEACH] Graph saved: %s", resp.message)
            else:
                rospy.logwarn("[TEACH] Save failed: %s", resp.message)
        except rospy.ServiceException as e:
            rospy.logerr("[TEACH] /graph/save error: %s", str(e))

        self.state = SAVED

        # NEW — persist VO waypoints before printing summary
        self._dump_waypoints()

        self._print_teach_summary()

    def _do_mark_junction(self):
        """
        Mark the most recently added node as a junction (Method 2).
        Publishes a SetGoal call to step5 — step5 handles the flag.
        In practice step5 is listening for /teach/mark_junction and
        we call it via the node_added callback index.
        """
        if self.state != RECORDING:
            rospy.logwarn("[TEACH] Not recording — cannot mark junction.")
            return
        if self.last_node_id < 0:
            rospy.logwarn("[TEACH] No node added yet.")
            return

        # We publish the junction request via a dedicated topic
        # step5 reads /graph/status to know which node to flag
        # The actual flagging is done by calling the step5 internal
        # mark_junction method — here we log and track it
        self.junctions.append(self.last_node_id)
        rospy.loginfo("[TEACH] Junction marked at node %d", self.last_node_id)

        # Publish to /graph/junction_event so step5 can react
        # (step5 also exposes a mark_junction service)
        try:
            svc = rospy.ServiceProxy('/graph/mark_junction', Trigger)
            svc()
        except Exception:
            pass  # Service optional — step5 may not expose it

    def _do_mark_endpoint(self, label):
        if self.state != RECORDING:
            rospy.logwarn("[TEACH] Not recording — cannot mark endpoint.")
            return
        if self.last_node_id < 0:
            rospy.logwarn("[TEACH] No node to mark as endpoint.")
            return

        self.endpoints.append((self.last_node_id, label))
        rospy.loginfo("[TEACH] Endpoint '%s' marked at node %d",
                      label, self.last_node_id)

        try:
            resp = self._svc_set_goal(label)
            if not resp.success:
                rospy.logwarn("[TEACH] set_goal failed: %s", resp.message)
        except rospy.ServiceException as e:
            rospy.logwarn("[TEACH] /graph/set_goal error: %s", str(e))

    # ── Services ──────────────────────────────────────────────────────────

    def _srv_start(self, req):
        self._do_start()
        return TriggerResponse(
            success=True,
            message='Recording started  route=%s' % self.route_id
        )

    def _srv_stop(self, req):
        if self.state != RECORDING:
            return TriggerResponse(success=False,
                                   message='Not recording')
        self._do_stop()
        return TriggerResponse(
            success=True,
            message='Teach complete  nodes=%d' % self.n_nodes_added
        )

    def _srv_mark_junction(self, req):
        self._do_mark_junction()
        return TriggerResponse(
            success=True,
            message='Junction at node %d' % self.last_node_id
        )

    def _srv_mark_endpoint(self, req):
        self._do_mark_endpoint(req.goal_label)
        return SetGoalResponse(
            success=True,
            message='Endpoint %s at node %d' % (
                req.goal_label, self.last_node_id)
        )

    # ── HUD image ─────────────────────────────────────────────────────────

    def _build_hud(self):
        """
        Build a HUD image showing teach state.
        Published to /teach/hud — view with rqt_image_view.

        Layout:
            [STATE badge] [nodes: N] [frames: N] [duration: Xs]
            [edge: inliers=N conf=0.XX] [junctions: N] [route: name]
        """
        hud = np.zeros((self.hud_h, self.hud_w, 3), dtype=np.uint8)
        hud[:] = COL_DARK

        # State badge
        if self.state == IDLE:
            badge_col  = COL_GREY
            badge_text = ' IDLE '
        elif self.state == RECORDING:
            badge_col  = COL_GREEN
            badge_text = ' REC  '
        else:
            badge_col  = COL_AMBER
            badge_text = ' SAVED'

        cv2.rectangle(hud, (8, 8), (100, 44), badge_col, -1)
        cv2.putText(hud, badge_text, (12, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_DARK, 2)

        # Duration
        if self.teach_start_t is not None and self.state == RECORDING:
            dur = time.time() - self.teach_start_t
            dur_str = '%ds' % int(dur)
        else:
            dur_str = '--'

        # Row 1 — main stats
        stats1 = 'nodes: %d    frames: %d    duration: %s    route: %s' % (
            self.n_nodes_added, self.n_frames_recv, dur_str, self.route_id
        )
        cv2.putText(hud, stats1, (112, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_WHITE, 1)

        # Row 2 — edge quality
        if self.last_edge is not None:
            inliers = self.last_edge.get('edge_inliers', 0)
            conf    = self.last_edge.get('edge_conf',    0.0)
            lat     = self.last_edge.get('lateral',      0.0)
            yaw     = self.last_edge.get('yaw_deg',      0.0)
            conf_col = COL_GREEN if conf > 0.4 else COL_RED
            edge_str = 'last edge: inliers=%d  conf=%.2f  lat=%+.3f  yaw=%+.1fdeg' % (
                inliers, conf, lat, yaw
            )
            cv2.putText(hud, edge_str, (12, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, conf_col, 1)
        else:
            cv2.putText(hud, 'last edge: --', (12, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_GREY, 1)

        # Row 3 — junctions and endpoint
        junc_str = 'junctions: %d    endpoint: %s' % (
            len(self.junctions),
            ('"%s" @ node %d' % self.endpoints[-1][1::-1])
            if self.endpoints else '--'
        )
        cv2.putText(hud, junc_str, (12, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COL_AMBER, 1)

        # Controls reminder at right edge
        ctrl = 'START=rec  SELECT=stop  X=junction'
        cv2.putText(hud, ctrl, (self.hud_w - 330, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_GREY, 1)

        return hud

    def _publish_hud(self):
        if self.pub_hud.get_num_connections() == 0:
            return
        hud = self._build_hud()
        try:
            msg = self.bridge.cv2_to_imgmsg(hud, 'bgr8')
            self.pub_hud.publish(msg)
        except CvBridgeError:
            pass

    # ── Summary ───────────────────────────────────────────────────────────

    def _print_teach_summary(self):
        dur = (time.time() - self.teach_start_t) if self.teach_start_t else 0
        rospy.loginfo("=" * 52)
        rospy.loginfo("[TEACH] TEACH RUN COMPLETE")
        rospy.loginfo("  route          : %s", self.route_id)
        rospy.loginfo("  duration       : %.1f s", dur)
        rospy.loginfo("  nodes saved    : %d", self.n_nodes_added)
        rospy.loginfo("  frames recv    : %d", self.n_frames_recv)
        rospy.loginfo("  save rate      : %.1f%%",
                      100.0 * self.n_nodes_added / max(self.n_frames_recv, 1))
        rospy.loginfo("  junctions      : %d  %s",
                      len(self.junctions), str(self.junctions))
        rospy.loginfo("  endpoints      : %s", str(self.endpoints))
        rospy.loginfo("  graph path     : %s", self.save_path)
        rospy.loginfo("=" * 52)
        rospy.loginfo("[TEACH] To start Repeat phase:")
        rospy.loginfo("  roslaunch vtr repeat.launch "
                      "goal:='%s'", self.ep_label or 'your_destination')

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        # NEW — register shutdown hook to save waypoints on Ctrl-C or kill
        rospy.on_shutdown(self._dump_waypoints)

        rate = rospy.Rate(10)   # 10 Hz — enough for HUD and status
        while not rospy.is_shutdown():
            # Publish status JSON
            status = {
                'state'        : self.state,
                'route_id'     : self.route_id,
                'n_nodes'      : self.n_nodes_added,
                'n_frames'     : self.n_frames_recv,
                'last_node_id' : self.last_node_id,
                'n_junctions'  : len(self.junctions),
                'endpoints'    : self.endpoints,
                'append_mode'  : self.append_mode,
                'n_waypoints'  : len(self.waypoints),   # NEW — show in status
            }
            self.pub_status.publish(String(data=json.dumps(status)))

            # Publish HUD
            self._publish_hud()

            rate.sleep()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = TeachLoggerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass