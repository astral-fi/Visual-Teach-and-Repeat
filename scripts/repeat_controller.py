#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step9_repeat_controller_node.py  —  ROS Melodic Repeat Phase Controller
VT&R Project | Phase 2

ROS NODE NAME : /repeat_controller
SUBSCRIBES    : /geometry/result       (std_msgs/String)   — inlier counts
PUBLISHES     : /graph/current_node    (std_msgs/String)   — target for geo engine
                /graph/next_node       (std_msgs/String)   — i+1 for Goal List
                /repeat/state          (std_msgs/String)   — state for PID
                /repeat/stats          (std_msgs/String)   — run statistics

SERVICES      : /repeat/start          (std_srvs/Trigger)  — begin repeat run
                /repeat/stop           (std_srvs/Trigger)  — emergency stop

WHAT THIS NODE DOES:
    Brain of the Repeat phase. Manages the node pointer across the
    pre-planned route from the memory graph.

    On startup:
        Loads graph.pkl from disk.
        Runs BFS to plan the route to the configured goal label.
        Publishes node 0 as the current target → geometry engine starts matching.

    Every geometry result received:
        Reads inlier count for current node i.
        Reads inlier count for next node i+1 (Goal List — Paper 2).
        If inliers(i+1) > inliers(i) for 2 consecutive frames → advance pointer.
        If inliers(i) < MIN_INLIERS for FAILURE_TIMEOUT seconds → failure path.

    Failure path (from Paper 1 dynamic pruning):
        Mark node i as pruned.
        Jump pointer to i+1, attempt matching.
        If still failing, jump to i+2 (max LOOKAHEAD jumps).
        If all lookahead nodes fail → publish FAILURE state → PID stops motors.

    Endpoint detection:
        When pointer reaches a node with is_endpoint=True AND inliers
        stay above MIN_INLIERS for ENDPOINT_HOLD seconds → COMPLETE.

PARAMS:
    ~graph_path        (str,   default='~/vtr_graph')   directory with graph.pkl
    ~goal_label        (str,   default='')              destination endpoint label
    ~min_inliers       (int,   default=15)              floor for reliable match
    ~advance_votes     (int,   default=2)               consecutive frames to advance
    ~failure_timeout   (float, default=3.0)             seconds before pruning
    ~lookahead         (int,   default=3)               max jump attempts on failure
    ~endpoint_hold     (float, default=1.0)             seconds to confirm endpoint
    ~publish_rate      (float, default=20.0)            Hz for state/node publishing
=============================================================================
"""

import rospy
import numpy as np
import json
import os
import pickle
import time

from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from memory_graph import TopologicalMemoryGraph, Edge, KeyframeNode

# ── State machine constants ───────────────────────────────────────────────────

STATE_IDLE       = 'IDLE'
STATE_RUNNING    = 'RUNNING'
STATE_RECOVERING = 'RECOVERING'
STATE_COMPLETE   = 'COMPLETE'
STATE_FAILURE    = 'FAILURE'
STATE_STOPPED    = 'STOPPED'


# ── Graph loader ──────────────────────────────────────────────────────────────

def load_graph(graph_path):
    """
    Load TopologicalMemoryGraph from graph.pkl.
    Returns the graph object or raises IOError.
    """
    pkl_path = os.path.join(os.path.expanduser(graph_path), 'graph.pkl')
    if not os.path.exists(pkl_path):
        raise IOError("graph.pkl not found at: %s" % pkl_path)
    with open(pkl_path, 'rb') as f:
        graph = pickle.load(f)
    return graph


def node_to_json(node):
    """
    Serialise a KeyframeNode to JSON string for publishing.
    Includes descriptors_flat (N*32 uint8 values) and keypoint arrays.
    """
    if node is None:
        return json.dumps({'node_id': -1})

    desc_flat = []
    if node.descriptors is not None:
        desc_flat = node.descriptors.flatten().tolist()

    return json.dumps({
        'node_id'         : node.id,
        'route_id'        : node.route_id,
        'is_junction'     : node.is_junction,
        'is_endpoint'     : node.is_endpoint,
        'endpoint_label'  : node.endpoint_label,
        'descriptors_flat': desc_flat,
        'keypoint_x'      : list(node.keypoints_x),
        'keypoint_y'      : list(node.keypoints_y),
        'keypoint_angle'  : list(node.keypoints_angle),
        'keypoint_size'   : list(node.keypoints_size),
        'keypoint_octave' : list(node.keypoints_octave),
        'orb_count'       : node.orb_count,
        'quality_score'   : round(node.quality_score, 3),
    })


# ── ROS node ──────────────────────────────────────────────────────────────────

class RepeatControllerNode(object):
    """
    Manages the node pointer during the Repeat phase.
    """

    def __init__(self):
        rospy.init_node('repeat_controller', anonymous=False)

        # ── Params ────────────────────────────────────────────────────────
        self.graph_path      = rospy.get_param('~graph_path',      '~/vtr_graph')
        self.goal_label      = rospy.get_param('~goal_label',      '')
        self.min_inliers     = rospy.get_param('~min_inliers',     15)
        self.advance_votes   = rospy.get_param('~advance_votes',   2)
        self.failure_timeout = rospy.get_param('~failure_timeout', 3.0)
        self.lookahead       = rospy.get_param('~lookahead',       3)
        self.endpoint_hold   = rospy.get_param('~endpoint_hold',   1.0)
        self.pub_rate        = rospy.get_param('~publish_rate',    20.0)

        # ── Load graph ────────────────────────────────────────────────────
        self.graph = None
        while self.graph is None and not rospy.is_shutdown():
            try:
                self.graph = load_graph(self.graph_path)
                rospy.loginfo("[REPEAT] Loaded graph: %d nodes  routes=%s",
                              len(self.graph.nodes),
                              list(self.graph.routes.keys()))
            except IOError as e:
                rospy.logwarn("[REPEAT] %s", str(e))
                rospy.logwarn("[REPEAT] Waiting for graph.pkl — "
                              "run teach phase first. Retrying in 5s...")
                rospy.sleep(5.0)

        if rospy.is_shutdown():
            return

        # ── Plan route ────────────────────────────────────────────────────
        self.route = []
        if self.goal_label:
            self.route = self.graph.plan_route(0, self.goal_label)
            if not self.route:
                rospy.logwarn("[REPEAT] No route to '%s' — using all nodes",
                              self.goal_label)
        else:
            rospy.logwarn("[REPEAT] No goal_label set. "
                          "Pass goal:=<label> to repeat.launch")
            rospy.logwarn("[REPEAT] Available endpoints: %s",
                          str(self.graph.list_endpoints()))

        if not self.route:
            # Default: traverse all non-pruned nodes in order
            self.route = [n.id for n in self.graph.nodes if not n.pruned]

        rospy.loginfo("[REPEAT] Route: %d nodes  goal='%s'",
                      len(self.route), self.goal_label)
        rospy.loginfo("[REPEAT] Node sequence: %s%s",
                      str(self.route[:10]),
                      '...' if len(self.route) > 10 else '')

        # ── State ─────────────────────────────────────────────────────────
        self.state             = STATE_IDLE
        self.pointer           = 0         # index into self.route
        self.advance_count     = 0         # consecutive frames next > curr
        self.t_failure_start   = None      # when current failure began
        self.t_endpoint_start  = None      # when endpoint hold began
        self.n_pruned          = 0
        self.consec_failures   = 0
        self.recovery_attempts = 0

        # Last geometry result cache
        self.last_inliers_curr = 0
        self.last_inliers_next = 0
        self.last_confidence   = 0.0
        self.last_method       = 'none'
        self.t_last_result     = 0.0

        # Run statistics
        self.t_run_start       = None
        self.n_frames          = 0
        self.total_inliers     = 0

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber('/geometry/result', String,
                         self._cb_geo_result, queue_size=1)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_curr  = rospy.Publisher(
            '/graph/current_node', String, queue_size=1)
        self.pub_next  = rospy.Publisher(
            '/graph/next_node',    String, queue_size=1)
        self.pub_state = rospy.Publisher(
            '/repeat/state', String, queue_size=5)
        self.pub_stats = rospy.Publisher(
            '/repeat/stats', String, queue_size=5)

        # ── Services ──────────────────────────────────────────────────────
        rospy.Service('/repeat/start', Trigger, self._srv_start)
        rospy.Service('/repeat/stop',  Trigger, self._srv_stop)

        rospy.loginfo("[REPEAT] Ready. Call /repeat/start to begin.")
        rospy.loginfo("[REPEAT] Available endpoints: %s",
                      str(self.graph.list_endpoints()))

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def current_node(self):
        """The node currently being tracked."""
        if self.pointer < len(self.route):
            node_id = self.route[self.pointer]
            return self.graph.nodes[node_id]
        return None

    @property
    def next_node(self):
        """The node after the current one (for Goal List scoring)."""
        if self.pointer + 1 < len(self.route):
            node_id = self.route[self.pointer + 1]
            return self.graph.nodes[node_id]
        return None

    @property
    def progress_pct(self):
        if not self.route:
            return 0.0
        return 100.0 * self.pointer / max(len(self.route) - 1, 1)

    # ── Geometry result callback ──────────────────────────────────────────

    def _cb_geo_result(self, msg):
        """
        Receive geometry engine result.
        Update pointer advance logic and failure detection.
        Only active when state == RUNNING or RECOVERING.
        """
        if self.state not in (STATE_RUNNING, STATE_RECOVERING):
            return

        try:
            data = json.loads(msg.data)
        except ValueError:
            return

        self.t_last_result     = time.time()
        self.n_frames         += 1
        inliers                = data.get('inlier_count', 0)
        self.last_inliers_curr = inliers
        self.last_confidence   = data.get('confidence', 0.0)
        self.last_method       = data.get('method', 'none')
        self.last_forward      = data.get('forward', 1.0)
        self.total_inliers    += inliers

        # ── Endpoint detection ─────────────────────────────────────────
        curr = self.current_node
        if curr is not None and curr.is_endpoint:
            if inliers >= self.min_inliers:
                if self.t_endpoint_start is None:
                    self.t_endpoint_start = time.time()
                elif (time.time() - self.t_endpoint_start) >= self.endpoint_hold:
                    self._do_complete()
                    return
            else:
                self.t_endpoint_start = None

        # ── Failure detection ──────────────────────────────────────────
        if inliers < self.min_inliers:
            if self.t_failure_start is None:
                self.t_failure_start = time.time()
            elif (time.time() - self.t_failure_start) >= self.failure_timeout:
                self._do_failure()
                return
            # Within failure window — do not advance, wait
            return
        else:
            # Good inliers — reset failure timer
            self.t_failure_start   = None
            self.consec_failures   = 0
            self.recovery_attempts = 0
            if self.state == STATE_RECOVERING:
                rospy.loginfo("[REPEAT] Recovered at node %d",
                              self.current_node.id if self.current_node else -1)
                self.state = STATE_RUNNING

        # ── Goal List pointer advance ──────────────────────────────────
        # Advance when the camera physically reaches the target node
        # forward < 0.2 indicates the target is very close or passed
        #
        if getattr(self, 'last_forward', 1.0) < 0.2:
            self._try_advance()

    # ── Pointer advance ───────────────────────────────────────────────────

    def _try_advance(self):
        """
        Attempt to advance pointer to the next node.
        Called when physical distance drops below threshold.
        """
        self.advance_count = 0

        if self.pointer + 1 >= len(self.route):
            # At the last node — no advance possible
            return

        next_n = self.next_node
        if next_n is None or next_n.pruned:
            # Next node is pruned — skip it
            self.pointer += 1
            self._skip_pruned()
            return

        # Advance pointer unconditionally since we physically reached it
        self._advance_pointer()

    def _advance_pointer(self):
        """Move pointer to next node and publish new targets."""
        old_id = self.route[self.pointer] if self.pointer < len(self.route) else -1
        self.pointer      += 1
        self.advance_count = 0

        new_node = self.current_node
        if new_node is None:
            return

        rospy.loginfo(
            "[REPEAT] Pointer advance: node %d → %d  (%.1f%% complete)",
            old_id, new_node.id, self.progress_pct
        )

        # Publish updated targets immediately
        self._publish_targets()

    def _skip_pruned(self):
        """Skip over any consecutive pruned nodes."""
        while self.pointer < len(self.route):
            node_id = self.route[self.pointer]
            if not self.graph.nodes[node_id].pruned:
                break
            self.pointer += 1
            rospy.logdebug("[REPEAT] Skipped pruned node %d", node_id)
        self._publish_targets()

    # ── Failure and recovery ──────────────────────────────────────────────

    def _do_failure(self):
        """
        Localisation failure at current node.
        Prune it and attempt recovery by jumping forward.
        From Paper 1: dynamic graph maintenance.
        """
        curr = self.current_node
        if curr is None:
            self._do_stop()
            return

        rospy.logwarn("[REPEAT] Failure at node %d after %.1fs  attempts=%d",
                      curr.id, self.failure_timeout, self.recovery_attempts)

        # Prune current node
        curr.pruned    = True
        self.n_pruned += 1
        self.t_failure_start = None
        self.state           = STATE_RECOVERING
        self.recovery_attempts += 1

        # Check max recovery attempts
        if self.recovery_attempts > self.lookahead:
            rospy.logerr(
                "[REPEAT] Exceeded %d recovery attempts — FAILURE",
                self.lookahead
            )
            self._do_stop(failure=True)
            return

        # Jump to next node
        self.pointer      += 1
        self.advance_count = 0

        # Skip any additional pruned nodes
        self._skip_pruned()

        new_node = self.current_node
        if new_node is None:
            rospy.logerr("[REPEAT] No more nodes after recovery jump — FAILURE")
            self._do_stop(failure=True)
            return

        rospy.loginfo("[REPEAT] Recovery jump → node %d", new_node.id)
        self._publish_targets()

    # ── Complete and stop ─────────────────────────────────────────────────

    def _do_complete(self):
        """Endpoint reached successfully."""
        self.state = STATE_COMPLETE
        rospy.loginfo("[REPEAT] *** ENDPOINT REACHED: '%s' ***",
                      self.goal_label)
        self._publish_state()
        self._print_run_stats()

    def _do_stop(self, failure=False):
        """Stop the repeat run."""
        self.state = STATE_FAILURE if failure else STATE_STOPPED
        self._publish_state()
        if failure:
            rospy.logerr("[REPEAT] Run ended with FAILURE")
        else:
            rospy.loginfo("[REPEAT] Run stopped")
        self._print_run_stats()

    # ── Services ──────────────────────────────────────────────────────────

    def _srv_start(self, req):
        if self.state == STATE_RUNNING:
            return TriggerResponse(success=False, message='Already running')

        if not self.route:
            return TriggerResponse(success=False, message='No route planned')

        # Reset state
        self.pointer           = 0
        self.advance_count     = 0
        self.t_failure_start   = None
        self.t_endpoint_start  = None
        self.n_pruned          = 0
        self.consec_failures   = 0
        self.recovery_attempts = 0
        self.n_frames          = 0
        self.total_inliers     = 0
        self.t_run_start       = time.time()
        self.state             = STATE_RUNNING

        # Publish first node immediately
        self._publish_targets()

        rospy.loginfo("[REPEAT] *** RUN STARTED ***  goal='%s'  nodes=%d",
                      self.goal_label, len(self.route))
        return TriggerResponse(
            success=True,
            message='Running  goal=%s  nodes=%d' % (
                self.goal_label, len(self.route))
        )

    def _srv_stop(self, req):
        self._do_stop()
        return TriggerResponse(
            success=True,
            message='Stopped at node %d' % (
                self.route[self.pointer]
                if self.pointer < len(self.route) else -1)
        )

    # ── Publishing ────────────────────────────────────────────────────────

    def _publish_targets(self):
        """
        Publish current and next node data to the geometry engine.
        Called immediately on pointer advance.
        """
        curr = self.current_node
        nxt  = self.next_node

        self.pub_curr.publish(String(data=node_to_json(curr)))
        self.pub_next.publish(String(data=node_to_json(nxt)))

    def _publish_state(self):
        """Publish repeat controller state to PID and monitoring."""
        curr_id = (self.route[self.pointer]
                   if self.pointer < len(self.route) else -1)

        state_msg = {
            'state'          : self.state,
            'node_id'        : curr_id,
            'pointer'        : self.pointer,
            'route_length'   : len(self.route),
            'progress_pct'   : round(self.progress_pct, 1),
            'n_pruned'       : self.n_pruned,
            'last_inliers'   : self.last_inliers_curr,
            'last_confidence': round(self.last_confidence, 3),
            'last_method'    : self.last_method,
            'recovery_attempts': self.recovery_attempts,
        }
        self.pub_state.publish(String(data=json.dumps(state_msg)))

    # ── Statistics ────────────────────────────────────────────────────────

    def _print_run_stats(self):
        dur = (time.time() - self.t_run_start) if self.t_run_start else 0
        mean_inliers = (
            float(self.total_inliers) / max(self.n_frames, 1)
        )
        rospy.loginfo("=" * 52)
        rospy.loginfo("[REPEAT] RUN SUMMARY")
        rospy.loginfo("  state          : %s", self.state)
        rospy.loginfo("  goal           : %s", self.goal_label)
        rospy.loginfo("  duration       : %.1f s", dur)
        rospy.loginfo("  nodes traversed: %d / %d",
                      self.pointer, len(self.route))
        rospy.loginfo("  progress       : %.1f%%", self.progress_pct)
        rospy.loginfo("  nodes pruned   : %d", self.n_pruned)
        rospy.loginfo("  frames         : %d", self.n_frames)
        rospy.loginfo("  mean inliers   : %.1f", mean_inliers)
        rospy.loginfo("=" * 52)

        # Publish final stats
        stats = {
            'state'           : self.state,
            'goal'            : self.goal_label,
            'duration_s'      : round(dur, 1),
            'nodes_traversed' : self.pointer,
            'route_length'    : len(self.route),
            'progress_pct'    : round(self.progress_pct, 1),
            'n_pruned'        : self.n_pruned,
            'n_frames'        : self.n_frames,
            'mean_inliers'    : round(mean_inliers, 1),
        }
        self.pub_stats.publish(String(data=json.dumps(stats)))

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rate = rospy.Rate(self.pub_rate)
        rospy.loginfo("[REPEAT] Spinning at %.0f Hz", self.pub_rate)

        while not rospy.is_shutdown():
            # Publish state every cycle — PID needs it to stay active
            self._publish_state()

            # Re-publish current targets at rate (geometry engine may restart)
            if self.state in (STATE_RUNNING, STATE_RECOVERING):
                self._publish_targets()

            rate.sleep()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = RepeatControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass