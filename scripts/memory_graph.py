#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
ROS NODE NAME : /memory_graph
SUBSCRIBES    : /keyframe/saved         (vtr/FrameFeatures)  — from step4
PUBLISHES     : /graph/status           (std_msgs/String)    — JSON status
                /graph/node_added       (std_msgs/String)    — JSON per node
                /graph/junction_event   (std_msgs/String)    — branch events

SERVICES      : /graph/save             (std_srvs/Trigger)   — save to disk
                /graph/set_goal         (vtr/SetGoal)        — set repeat dest
                /graph/plan_route       (vtr/PlanRoute)      — BFS route plan

CUSTOM SERVICES (place in vtr/srv/):

    SetGoal.srv:
        string goal_label
        ---
        bool   success
        string message

    PlanRoute.srv:
        string goal_label
        ---
        bool    success
        int32[] node_sequence
        string  message

PARAMS:
    ~route_id          (str,   default='route_0')  route name for teach run
    ~endpoint_label    (str,   default='')         destination label
    ~append_mode       (bool,  default=False)      Method 3 append teach
    ~overlap_threshold (int,   default=40)         RANSAC inliers to confirm
    ~overlap_votes     (int,   default=3)          consecutive frames needed
    ~save_path         (str,   default='~/vtr_graph') output directory
    ~calib_path        (str,   default='')         calibration.yaml path
    ~min_edge_inliers  (int,   default=10)         min inliers for valid edge
=============================================================================
"""

import rospy
import numpy as np
import cv2
import json
import os
import pickle
import time
import yaml
from collections import deque

from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse
from vtr.msg      import FrameFeatures
from vtr.srv      import SetGoal,    SetGoalResponse
from vtr.srv      import PlanRoute,  PlanRouteResponse


# ── Data structures ───────────────────────────────────────────────────────────

class KeyframeNode(object):
    """
    One node in the topological memory graph.
    Stores visual descriptors + graph metadata.
    Raw pixel data is never stored.
    Typical size: ~16 KB per node (500 keypoints x 32 bytes descriptors).
    """

    def __init__(self, node_id, route_id, timestamp,
                 keypoints_x, keypoints_y,
                 keypoints_angle, keypoints_size, keypoints_octave,
                 descriptors, quality_score, orb_count, entropy):

        self.id               = node_id
        self.route_id         = route_id
        self.timestamp        = timestamp

        # Visual features — only these are needed for matching
        self.keypoints_x      = keypoints_x       # list[float]
        self.keypoints_y      = keypoints_y
        self.keypoints_angle  = keypoints_angle
        self.keypoints_size   = keypoints_size
        self.keypoints_octave = keypoints_octave
        self.descriptors      = descriptors        # np.ndarray (N,32) uint8

        # Quality metadata
        self.quality_score    = quality_score
        self.orb_count        = orb_count
        self.entropy          = entropy

        # Graph structure
        self.next_nodes       = []      # list[int] — supports branching
        self.is_junction      = False
        self.is_endpoint      = False
        self.endpoint_label   = ''
        self.pruned           = False   # marked bad during Repeat phase

    def to_cv_keypoints(self):
        """Reconstruct list of cv2.KeyPoint from stored arrays."""
        kps = []
        for x, y, ang, sz, oct_ in zip(
                self.keypoints_x,     self.keypoints_y,
                self.keypoints_angle, self.keypoints_size,
                self.keypoints_octave):
            kps.append(cv2.KeyPoint(
                x=float(x), y=float(y),
                _size=float(sz), _angle=float(ang),
                _octave=int(oct_)
            ))
        return kps

    def to_debug_dict(self):
        """Serialisable summary — no descriptor bytes."""
        return {
            'id'            : self.id,
            'route_id'      : self.route_id,
            'timestamp'     : round(self.timestamp,    3),
            'orb_count'     : self.orb_count,
            'quality_score' : round(self.quality_score, 3),
            'entropy'       : round(self.entropy,       3),
            'next_nodes'    : self.next_nodes,
            'is_junction'   : self.is_junction,
            'is_endpoint'   : self.is_endpoint,
            'endpoint_label': self.endpoint_label,
            'pruned'        : self.pruned,
        }


class Edge(object):
    """
    Directed edge: node[src] → node[dst].
    Stores the relative pose (R, t) from the Essential Matrix.
    """

    def __init__(self, src_id, dst_id, R, t, match_count, confidence):
        self.src_id      = src_id
        self.dst_id      = dst_id
        self.R           = R              # np.ndarray (3,3)
        self.t           = t              # np.ndarray (3,1) unit vector
        self.match_count = match_count    # RANSAC inlier count
        self.confidence  = confidence     # 1/cond(E) clipped [0,1]

    def to_debug_dict(self):
        return {
            'src'         : self.src_id,
            'dst'         : self.dst_id,
            'match_count' : self.match_count,
            'confidence'  : round(self.confidence, 3),
            'lateral'     : round(float(self.t[0][0]), 4),
            'yaw_deg'     : round(float(np.degrees(
                np.arctan2(self.R[1, 0], self.R[0, 0]))), 2),
        }


class TopologicalMemoryGraph(object):
    """
    Core VT&R data structure.

    Ordered list of KeyframeNodes connected by Edges.
    Supports single route, multiple routes with branching,
    append-teach overlap merging, BFS route planning, and disk I/O.
    """

    def __init__(self):
        self.nodes    = []    # list[KeyframeNode]
        self.edges    = []    # list[Edge|None], parallel to nodes
        self.routes   = {}    # route_id -> list[node_id]
        self.metadata = {
            'created_at'     : time.time(),
            'total_nodes'    : 0,
            'teach_duration' : 0.0,
            'routes'         : [],
        }

    # ── Node management ───────────────────────────────────────────────────

    def add_node(self, node, edge=None):
        """
        Append a KeyframeNode to the graph.
        Optionally attach an Edge from the previous node.
        Automatically links previous node's next_nodes to this one.
        Returns the assigned node_id.
        """
        node_id  = len(self.nodes)
        node.id  = node_id
        self.nodes.append(node)

        # Link previous node
        if node_id > 0:
            prev = self.nodes[node_id - 1]
            if node_id not in prev.next_nodes:
                prev.next_nodes.append(node_id)

        self.edges.append(edge)

        # Track route membership
        if node.route_id not in self.routes:
            self.routes[node.route_id] = []
        self.routes[node.route_id].append(node_id)

        self.metadata['total_nodes'] = len(self.nodes)
        return node_id

    def mark_junction(self, node_id):
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id].is_junction = True
            rospy.loginfo("[GRAPH] Node %d marked as junction", node_id)

    def mark_endpoint(self, node_id, label):
        if 0 <= node_id < len(self.nodes):
            self.nodes[node_id].is_endpoint    = True
            self.nodes[node_id].endpoint_label = label
            rospy.loginfo("[GRAPH] Node %d marked as endpoint '%s'",
                          node_id, label)

    # ── BFS route planning ────────────────────────────────────────────────

    def plan_route(self, start_id, goal_label):
        """
        BFS from start_id to the node with endpoint_label == goal_label.
        Skips pruned nodes.
        Returns list of node IDs start..goal, or [] if not found.
        """
        goal_id = None
        for node in self.nodes:
            if node.endpoint_label == goal_label and not node.pruned:
                goal_id = node.id
                break

        if goal_id is None:
            rospy.logwarn("[GRAPH] No endpoint '%s'", goal_label)
            return []

        visited = set()
        queue   = deque([[start_id]])
        while queue:
            path = queue.popleft()
            curr = path[-1]
            if curr == goal_id:
                return path
            if curr in visited:
                continue
            visited.add(curr)
            for nxt in self.nodes[curr].next_nodes:
                if nxt not in visited and not self.nodes[nxt].pruned:
                    queue.append(path + [nxt])

        rospy.logwarn("[GRAPH] No path from %d to '%s'", start_id, goal_label)
        return []

    def list_endpoints(self):
        return [(n.id, n.endpoint_label)
                for n in self.nodes if n.is_endpoint]

    # ── Validation ────────────────────────────────────────────────────────

    def validate(self):
        """Check graph for structural issues. Returns (bool, [warnings])."""
        warnings = []
        for i, node in enumerate(self.nodes):
            if (not node.is_endpoint
                    and len(node.next_nodes) == 0
                    and i < len(self.nodes) - 1):
                warnings.append('Node %d: no next_nodes, not endpoint' % i)
            for nxt in node.next_nodes:
                if nxt >= len(self.nodes):
                    warnings.append('Node %d: next_node %d out of range' % (i, nxt))
        if len(self.edges) != len(self.nodes):
            warnings.append('edges(%d) != nodes(%d)' % (
                len(self.edges), len(self.nodes)))
        return len(warnings) == 0, warnings

    # ── Disk I/O ──────────────────────────────────────────────────────────

    def save(self, save_dir):
        """
        Save to graph.pkl (pickle) and graph_debug.json.
        Returns (pkl_path, json_path).
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(self.nodes) > 0:
            self.metadata['teach_duration'] = (
                self.nodes[-1].timestamp - self.nodes[0].timestamp
            )
        self.metadata['routes']      = list(self.routes.keys())
        self.metadata['total_nodes'] = len(self.nodes)
        self.metadata['saved_at']    = time.time()

        # Pickle — full graph with descriptor arrays
        pkl_path = os.path.join(save_dir, 'graph.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self, f, protocol=2)

        # JSON — human-readable, no descriptor bytes
        debug = {
            'metadata' : self.metadata,
            'nodes'    : [n.to_debug_dict() for n in self.nodes],
            'edges'    : [
                e.to_debug_dict() if e is not None else None
                for e in self.edges
            ],
        }
        json_path = os.path.join(save_dir, 'graph_debug.json')
        with open(json_path, 'w') as f:
            json.dump(debug, f, indent=2)

        rospy.loginfo("[GRAPH] Saved %d nodes to %s", len(self.nodes), save_dir)
        return pkl_path, json_path

    @staticmethod
    def load(pkl_path):
        with open(pkl_path, 'rb') as f:
            graph = pickle.load(f)
        rospy.loginfo("[GRAPH] Loaded %d nodes from %s",
                      len(graph.nodes), pkl_path)
        return graph


# ── Edge computer ─────────────────────────────────────────────────────────────

class EdgeComputer(object):
    """
    Computes Essential Matrix between two consecutive KeyframeNodes.
    Pipeline: BFMatcher → ratio test → orientation filter → RANSAC → recoverPose.
    Returns an Edge object, or None if matching fails.
    """

    def __init__(self, K, min_inliers=10):
        self.K           = K
        self.min_inliers = min_inliers
        self.matcher     = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def compute(self, node_prev, node_curr):
        desc_a = node_prev.descriptors
        desc_b = node_curr.descriptors
        kp_a   = node_prev.to_cv_keypoints()
        kp_b   = node_curr.to_cv_keypoints()

        if desc_a is None or desc_b is None:
            return None
        if len(desc_a) < 8 or len(desc_b) < 8:
            return None

        # Stage 1: ratio test
        try:
            pairs = self.matcher.knnMatch(desc_a, desc_b, k=2)
        except cv2.error:
            return None

        good = [m for pair in pairs
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 0.75 * n.distance]

        if len(good) < self.min_inliers:
            return None

        # Stage 2: orientation histogram filter
        if len(good) > 8:
            deltas  = np.array([kp_a[m.queryIdx].angle - kp_b[m.trainIdx].angle
                                for m in good])
            hist, _ = np.histogram(deltas, bins=30, range=(-180.0, 180.0))
            top_idx = set(np.argsort(hist)[-3:].tolist())
            bin_w   = 360.0 / 30
            filtered = [m for m, d in zip(good, deltas)
                        if max(0, min(int((d + 180.0) / bin_w), 29)) in top_idx]
            if len(filtered) >= 8:
                good = filtered

        if len(good) < self.min_inliers:
            return None

        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])

        # Stage 3: RANSAC Essential Matrix
        try:
            E, mask = cv2.findEssentialMat(
                pts_a, pts_b, self.K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0
            )
        except cv2.error:
            return None

        if E is None or mask is None:
            return None

        inlier_count = int(np.sum(mask))
        if inlier_count < self.min_inliers:
            return None

        confidence = float(np.clip(1.0 / np.linalg.cond(E), 0.0, 1.0))
        _, R, t, _ = cv2.recoverPose(E, pts_a, pts_b, self.K, mask=mask)

        return Edge(
            src_id      = node_prev.id,
            dst_id      = node_curr.id,
            R           = R,
            t           = t,
            match_count = inlier_count,
            confidence  = confidence,
        )


# ── Overlap detector (Method 3 — Append Teach) ───────────────────────────────

class OverlapDetector(object):
    """
    During append teach, checks each new frame against all existing nodes.
    Two-tier: fast ratio pre-filter, then RANSAC confirmation.
    Requires overlap_votes consecutive matching frames before declaring junction.
    """

    def __init__(self, existing_graph, K,
                 overlap_threshold=40, overlap_votes=3):
        self.graph      = existing_graph
        self.K          = K
        self.threshold  = overlap_threshold
        self.votes_needed = overlap_votes
        self.vote_counts  = {}
        self.matcher    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def check(self, desc_new, kp_new):
        """
        Returns existing node_id if junction confirmed, else None.
        """
        if desc_new is None or len(desc_new) < 8:
            return None

        # Tier 1 — fast pre-filter
        candidates = []
        for node in self.graph.nodes:
            if node.pruned or node.descriptors is None:
                continue
            if len(node.descriptors) < 8:
                continue
            try:
                pairs = self.matcher.knnMatch(desc_new, node.descriptors, k=2)
            except cv2.error:
                continue
            n_good = sum(1 for p in pairs
                         if len(p) == 2 and p[0].distance < 0.75 * p[1].distance)
            if float(n_good) / max(len(desc_new), 1) > 0.35:
                candidates.append(node)

        if not candidates:
            self.vote_counts = {}
            return None

        # Tier 2 — RANSAC confirmation
        confirmed = None
        for node in candidates:
            kp_node = node.to_cv_keypoints()
            try:
                all_matches = self.matcher.match(desc_new, node.descriptors)
            except cv2.error:
                continue

            if len(all_matches) < 8:
                continue

            pts_new  = np.float32([kp_new[m.queryIdx].pt
                                   for m in all_matches])
            pts_node = np.float32([kp_node[m.trainIdx].pt
                                   for m in all_matches])
            try:
                _, mask = cv2.findEssentialMat(
                    pts_new, pts_node, self.K,
                    method=cv2.RANSAC, prob=0.999, threshold=1.0
                )
            except cv2.error:
                continue

            if mask is None:
                continue

            inliers = int(np.sum(mask))
            if inliers >= self.threshold:
                self.vote_counts[node.id] = \
                    self.vote_counts.get(node.id, 0) + 1
                if self.vote_counts[node.id] >= self.votes_needed:
                    confirmed = node.id
                    break
            else:
                self.vote_counts[node.id] = 0

        return confirmed


# ── ROS node ──────────────────────────────────────────────────────────────────

class MemoryGraphNode(object):

    def __init__(self):
        rospy.init_node('memory_graph', anonymous=False)

        # ── Params ────────────────────────────────────────────────────────
        self.route_id    = rospy.get_param('~route_id',         'route_0')
        self.ep_label    = rospy.get_param('~endpoint_label',   '')
        self.append_mode = rospy.get_param('~append_mode',      False)
        self.save_path   = os.path.expanduser(
            rospy.get_param('~save_path', '~/vtr_graph'))
        calib_path       = rospy.get_param('~calib_path',       '')
        self.min_inliers = rospy.get_param('~min_edge_inliers', 10)
        overlap_thresh   = rospy.get_param('~overlap_threshold', 40)
        overlap_votes    = rospy.get_param('~overlap_votes',     3)

        # ── Calibration ───────────────────────────────────────────────────
        self.K = self._load_K(calib_path)

        # ── Graph + tools ─────────────────────────────────────────────────
        self.graph     = TopologicalMemoryGraph()
        self.edge_comp = EdgeComputer(self.K, self.min_inliers)
        self.prev_node = None

        # ── Append mode ───────────────────────────────────────────────────
        self.overlap_det = None
        if self.append_mode:
            pkl = os.path.join(self.save_path, 'graph.pkl')
            if os.path.exists(pkl):
                self.graph       = TopologicalMemoryGraph.load(pkl)
                self.overlap_det = OverlapDetector(
                    self.graph, self.K,
                    overlap_threshold=overlap_thresh,
                    overlap_votes=overlap_votes
                )
                rospy.loginfo("[GRAPH] Append mode: %d existing nodes",
                              len(self.graph.nodes))
            else:
                rospy.logwarn("[GRAPH] Append mode but no graph.pkl — starting fresh")
                self.append_mode = False

        # ── State ─────────────────────────────────────────────────────────
        self.n_received = 0
        self.n_added    = 0

        # ── Subscribers ───────────────────────────────────────────────────
        self.sub = rospy.Subscriber(
            '/keyframe/saved', FrameFeatures,
            self._cb_keyframe, queue_size=10
        )

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_status   = rospy.Publisher('/graph/status',
                                            String, queue_size=5)
        self.pub_added    = rospy.Publisher('/graph/node_added',
                                            String, queue_size=10)
        self.pub_junction = rospy.Publisher('/graph/junction_event',
                                            String, queue_size=5)

        # ── Services ──────────────────────────────────────────────────────
        rospy.Service('/graph/save',       Trigger,   self._srv_save)
        rospy.Service('/graph/set_goal',   SetGoal,   self._srv_set_goal)
        rospy.Service('/graph/plan_route', PlanRoute, self._srv_plan_route)

        rospy.loginfo("[GRAPH] Ready  route=%s  append=%s  save=%s",
                      self.route_id, self.append_mode, self.save_path)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _load_K(self, calib_path):
        if not calib_path or not os.path.exists(calib_path):
            rospy.logwarn("[GRAPH] No calibration — using identity K")
            return np.eye(3, dtype=np.float64)
        with open(calib_path, 'r') as f:
            data = yaml.safe_load(f)
        K = np.array(data['camera_matrix']['data'],
                     dtype=np.float64).reshape(3, 3)
        rospy.loginfo("[GRAPH] K loaded from %s", calib_path)
        return K

    # ── Keyframe callback ─────────────────────────────────────────────────

    def _cb_keyframe(self, msg):
        self.n_received += 1

        if len(msg.descriptors_flat) == 0:
            rospy.logwarn("[GRAPH] Empty descriptors — skip")
            return

        desc = np.array(msg.descriptors_flat,
                        dtype=np.uint8).reshape(-1, 32)

        # ── Append overlap check ───────────────────────────────────────
        if self.overlap_det is not None:
            kp_cv = [cv2.KeyPoint(float(x), float(y), float(sz), float(ang))
                     for x, y, sz, ang in zip(
                         msg.keypoint_x, msg.keypoint_y,
                         msg.keypoint_size, msg.keypoint_angle)]
            junc_id = self.overlap_det.check(desc, kp_cv)
            if junc_id is not None:
                rospy.loginfo("[GRAPH] Junction confirmed at node %d", junc_id)
                self.graph.mark_junction(junc_id)
                if self.prev_node is not None and \
                        junc_id not in self.prev_node.next_nodes:
                    self.prev_node.next_nodes.append(junc_id)
                self.pub_junction.publish(String(data=json.dumps({
                    'event'      : 'overlap_confirmed',
                    'junction_id': junc_id,
                    'route_id'   : self.route_id,
                })))
                self.overlap_det = None
                return

        # ── Build node ─────────────────────────────────────────────────
        node = KeyframeNode(
            node_id         = len(self.graph.nodes),
            route_id        = self.route_id,
            timestamp       = msg.timestamp,
            keypoints_x     = list(msg.keypoint_x),
            keypoints_y     = list(msg.keypoint_y),
            keypoints_angle = list(msg.keypoint_angle),
            keypoints_size  = list(msg.keypoint_size),
            keypoints_octave= list(msg.keypoint_octave),
            descriptors     = desc,
            quality_score   = float(msg.quality_hint),
            orb_count       = msg.n_keypoints,
            entropy         = 0.0,
        )

        # ── Compute edge ───────────────────────────────────────────────
        edge = None
        if self.prev_node is not None:
            edge = self.edge_comp.compute(self.prev_node, node)
            if edge is None:
                rospy.logwarn("[GRAPH] Edge failed node %d→%d",
                              self.prev_node.id, node.id)

        # ── Add to graph ───────────────────────────────────────────────
        node_id        = self.graph.add_node(node, edge)
        self.prev_node = node
        self.n_added  += 1

        # ── Publish event ──────────────────────────────────────────────
        self.pub_added.publish(String(data=json.dumps({
            'node_id'      : node_id,
            'route_id'     : self.route_id,
            'orb_count'    : node.orb_count,
            'quality_score': round(node.quality_score, 3),
            'edge_inliers' : edge.match_count if edge else 0,
            'edge_conf'    : round(edge.confidence, 3) if edge else 0.0,
            'total_nodes'  : len(self.graph.nodes),
        })))

        if self.n_added % 20 == 0:
            rospy.loginfo("[GRAPH] %d nodes  edge_inliers=%s  conf=%s",
                          self.n_added,
                          str(edge.match_count) if edge else 'N/A',
                          str(round(edge.confidence, 2)) if edge else 'N/A')

    # ── Services ──────────────────────────────────────────────────────────

    def _srv_save(self, req):
        if len(self.graph.nodes) == 0:
            return TriggerResponse(success=False, message='Graph is empty')

        if self.ep_label:
            last = len(self.graph.nodes) - 1
            self.graph.mark_endpoint(last, self.ep_label)

        is_valid, warnings = self.graph.validate()
        for w in warnings:
            rospy.logwarn("[GRAPH] %s", w)

        self._print_stats()
        pkl_path, _ = self.graph.save(self.save_path)
        return TriggerResponse(
            success=True,
            message='Saved %d nodes to %s' % (len(self.graph.nodes),
                                               self.save_path)
        )

    def _srv_set_goal(self, req):
        labels = [ep[1] for ep in self.graph.list_endpoints()]
        if req.goal_label not in labels:
            return SetGoalResponse(
                success=False,
                message='Unknown goal: %s  available: %s' % (
                    req.goal_label, str(labels))
            )
        return SetGoalResponse(success=True,
                               message='Goal set: %s' % req.goal_label)

    def _srv_plan_route(self, req):
        path = self.graph.plan_route(0, req.goal_label)
        if not path:
            return PlanRouteResponse(success=False, node_sequence=[],
                                     message='No route to: ' + req.goal_label)
        return PlanRouteResponse(
            success=True, node_sequence=path,
            message='Route length: %d nodes' % len(path)
        )

    # ── Stats ─────────────────────────────────────────────────────────────

    def _print_stats(self):
        nodes = self.graph.nodes
        edges = [e for e in self.graph.edges if e is not None]
        if not nodes:
            return
        rospy.loginfo("=" * 52)
        rospy.loginfo("[GRAPH] TEACH SUMMARY")
        rospy.loginfo("  total nodes      : %d", len(nodes))
        rospy.loginfo("  routes           : %s",
                      list(self.graph.routes.keys()))
        rospy.loginfo("  teach duration   : %.1f s",
                      nodes[-1].timestamp - nodes[0].timestamp)
        rospy.loginfo("  mean quality     : %.3f",
                      np.mean([n.quality_score for n in nodes]))
        rospy.loginfo("  mean ORB count   : %.0f",
                      np.mean([n.orb_count for n in nodes]))
        rospy.loginfo("  mean edge inliers: %.0f",
                      np.mean([e.match_count for e in edges]) if edges else 0)
        rospy.loginfo("  mean edge conf   : %.3f",
                      np.mean([e.confidence for e in edges]) if edges else 0)
        rospy.loginfo("  junctions        : %d",
                      sum(1 for n in nodes if n.is_junction))
        rospy.loginfo("  endpoints        : %s",
                      [(n.id, n.endpoint_label)
                       for n in nodes if n.is_endpoint])
        rospy.loginfo("=" * 52)

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            self.pub_status.publish(String(data=json.dumps({
                'total_nodes' : len(self.graph.nodes),
                'route_id'    : self.route_id,
                'append_mode' : self.append_mode,
                'n_received'  : self.n_received,
                'n_added'     : self.n_added,
            })))
            rate.sleep()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = MemoryGraphNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
