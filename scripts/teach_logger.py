#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
teach_logger_node.py  —  ROS Melodic  (step 6 of Phase 1)
═══════════════════════════════════════════════════════════

This ROS node drives the full Teach pipeline end-to-end during a
manual drive.  It wires together Steps 2–5 and adds:
  • ROS image subscription (sensor_msgs/Image via cv_bridge)
  • Gamepad start / stop control (sensor_msgs/Joy)
  • Live preview publisher so you can watch on rqt_image_view
  • Status publisher for dashboard / terminal monitoring
  • Graceful save-on-shutdown with rospy shutdown hook

TOPIC MAP
─────────────────────────────────────────────────────────────────────
  SUBSCRIBED
  /camera/image_raw          sensor_msgs/Image   raw camera frames
  /joy                       sensor_msgs/Joy     gamepad (btn 0 = teach toggle)

  PUBLISHED
  /teach/preview             sensor_msgs/Image   annotated debug frame
  /teach/status              std_msgs/String     JSON status string

  SERVICES
  /teach/start               std_srvs/Trigger    start teaching
  /teach/stop                std_srvs/Trigger    stop + save graph

PARAMETER SERVER  (set in launch file or rosparam)
  ~clip_limit        float   CLAHE clip limit        default 2.0
  ~tile_size         int     CLAHE tile size          default 8
  ~n_features        int     ORB feature count        default 1000
  ~score_threshold   float   Keyframe score gate      default 0.40
  ~save_dir          str     Output directory         default ~/teach_memory
  ~graph_filename    str     Graph pickle filename    default graph.pkl
  ~joy_button        int     Gamepad button index     default 0
  ~image_transport   str     raw / compressed         default raw

RUN
  roslaunch vtr_jetracer teach.launch
  — OR —
  rosrun vtr_jetracer teach_logger_node.py

DEPENDENCIES  (all in the same ROS package src/ folder)
  step2_clahe.py
  step3_orb.py
  step4_keyframe_scorer.py
  step5_memory_graph.py
"""

import os
import sys
import time
import json
import threading
import pickle

import rospy
import cv2
import numpy as np

from sensor_msgs.msg import Image, Joy
from std_msgs.msg    import String
from std_srvs.srv    import Trigger, TriggerResponse
from cv_bridge       import CvBridge, CvBridgeError

# ── Import our pipeline steps ─────────────────────────────────────────────────
# All four files must live in the same directory as this node.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from orb_node            import ORBExtractor, ClaheProcessor
from vtr.msg import FrameFeatures
from step4_keyframe_scorer import KeyframeScorer
from step5_memory_graph   import MemoryGraph


# ─────────────────────────────────────────────────────────────────────────────
# TEACH LOGGER NODE
# ─────────────────────────────────────────────────────────────────────────────

class TeachLoggerNode(object):
    """
    ROS node that records a Teach phase and builds the topological graph.

    State machine
    ─────────────
      IDLE  ──[start srv / joy btn]──►  TEACHING  ──[stop srv / joy btn]──►  SAVING
                                                                                 │
                                                                              IDLE

    Thread safety
    ─────────────
    The image callback runs in a ROS spinner thread.
    The save / build operation runs in a separate thread so the ROS
    spinner never blocks.  self._lock protects shared state.
    """

    # ── States ────────────────────────────────────────────────────────────────
    STATE_IDLE     = "IDLE"
    STATE_TEACHING = "TEACHING"
    STATE_SAVING   = "SAVING"

    def __init__(self):
        rospy.init_node("teach_logger", anonymous=False)
        rospy.loginfo("TeachLoggerNode: initialising …")

        # ── Read ROS parameters ───────────────────────────────────────────────
        self._clip_limit      = rospy.get_param("~clip_limit",     2.0)
        self._tile_size       = rospy.get_param("~tile_size",       8)
        self._n_features      = rospy.get_param("~n_features",   1000)
        self._score_threshold = rospy.get_param("~score_threshold", 0.40)
        self._save_dir        = os.path.expanduser(
                                    rospy.get_param("~save_dir",
                                                    "~/teach_memory"))
        self._graph_filename  = rospy.get_param("~graph_filename", "graph.pkl")
        self._joy_button      = rospy.get_param("~joy_button",      0)

        # Camera topic — allows remapping in launch file
        cam_topic = rospy.get_param("~camera_topic", "/camera/image_raw")

        os.makedirs(self._save_dir, exist_ok=True)

        # ── Build pipeline objects ─────────────────────────────────────────────
        self._clahe = ClaheProcessor(
            K=K, D=D,
            width=CAM_W, height=CAM_H,
            clip_limit=self._clip_limit,
            tile_size=(self._tile_size, self._tile_size)
        )
        self._orb    = ORBExtractor(n_features=self._n_features)
        self._scorer = KeyframeScorer(self._orb,
                                      score_threshold=self._score_threshold)
        self._graph  = MemoryGraph(self._orb)

        # ── Shared state ───────────────────────────────────────────────────────
        self._lock           = threading.Lock()
        self._state          = self.STATE_IDLE
        self._collected      = []          # list of accepted FrameData
        self._frames_seen    = 0           # total frames processed
        self._last_joy_press = 0.0         # debounce timestamp
        self._teach_start_t  = 0.0

        # ── cv_bridge ──────────────────────────────────────────────────────────
        self._bridge = CvBridge()

        # ── Publishers ─────────────────────────────────────────────────────────
        self._pub_preview = rospy.Publisher(
            "/teach/preview", Image, queue_size=2)
        self._pub_status = rospy.Publisher(
            "/teach/status", String, queue_size=5)

        # ── Subscribers ────────────────────────────────────────────────────────
        rospy.Subscriber(cam_topic, Image,
                         self._cb_image, queue_size=1,
                         buff_size=2**24)   # large buffer for 1280×720
        rospy.Subscriber("/joy", Joy,
                         self._cb_joy, queue_size=10)

        # ── Services ───────────────────────────────────────────────────────────
        rospy.Service("/teach/start", Trigger, self._srv_start)
        rospy.Service("/teach/stop",  Trigger, self._srv_stop)

        # ── Shutdown hook ─────────────────────────────────────────────────────
        rospy.on_shutdown(self._on_shutdown)

        # ── Status timer (publishes /teach/status at 2 Hz) ────────────────────
        rospy.Timer(rospy.Duration(0.5), self._cb_status_timer)

        rospy.loginfo("TeachLoggerNode: ready.")
        rospy.loginfo(f"  Camera topic : {cam_topic}")
        rospy.loginfo(f"  Save dir     : {self._save_dir}")
        rospy.loginfo(f"  n_features   : {self._n_features}")
        rospy.loginfo(f"  score thresh : {self._score_threshold}")
        rospy.loginfo(f"  Joy button   : {self._joy_button}  (press to toggle teaching)")
        rospy.loginfo("  Services: /teach/start  /teach/stop")
        rospy.loginfo("  Preview : /teach/preview  (view with rqt_image_view)")

    # ─────────────────────────────────────────────────────────────────────────
    # IMAGE CALLBACK — runs in ROS spinner thread
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_image(self, msg):
        """
        Called for every camera frame.

        Pipeline per frame:
          1. cv_bridge  → numpy BGR array
          2. ClaheProcessor.process()  → gray_clahe, bgr_undist
          3. ORBExtractor.extract()    → FrameData (keypoints + descriptors)
          4. KeyframeScorer.score()    → ScoredFrame (accept/reject decision)
          5. If accepted AND teaching  → append to self._collected
          6. Build annotated preview image
          7. Publish preview on /teach/preview

        The pipeline runs even in IDLE state so the preview is always live.
        Only step 5 is gated on state == TEACHING.
        """
        with self._lock:
            state = self._state

        # ── Step 1: decode ROS image → numpy BGR ─────────────────────────────
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, f"cv_bridge error: {e}")
            return

        # ── Step 2: undistort + CLAHE ─────────────────────────────────────────
        gray_clahe, bgr_undist = self._clahe.process(bgr)

        # ── Step 3: ORB extraction ────────────────────────────────────────────
        ts = msg.header.stamp.to_sec()
        fd = self._orb.extract(gray_clahe, ts, bgr=bgr_undist)

        # ── Step 4: keyframe scoring ──────────────────────────────────────────
        sf = self._scorer.score(fd)

        with self._lock:
            self._frames_seen += 1

            # ── Step 5: save if teaching ──────────────────────────────────────
            if state == self.STATE_TEACHING and sf.accepted:
                self._collected.append(fd)

        # ── Step 6 + 7: build and publish preview ────────────────────────────
        preview = self._make_preview(bgr_undist, fd, sf, state)
        try:
            preview_msg = self._bridge.cv2_to_imgmsg(preview, encoding="bgr8")
            preview_msg.header = msg.header
            self._pub_preview.publish(preview_msg)
        except CvBridgeError as e:
            rospy.logwarn_throttle(5.0, f"preview publish error: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # JOYSTICK CALLBACK
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_joy(self, msg):
        """
        Toggle teaching on/off with a single gamepad button press.

        Uses a 0.5-second debounce to prevent double-triggers from
        button bounce on the F710 / PS4 controllers.

        Button index is set by ~joy_button parameter (default 0 = A/X button).
        """
        if self._joy_button >= len(msg.buttons):
            return

        if msg.buttons[self._joy_button] == 1:
            now = time.time()
            if now - self._last_joy_press < 0.5:
                return   # debounce
            self._last_joy_press = now
            self._toggle_teaching()

    def _toggle_teaching(self):
        with self._lock:
            state = self._state

        if state == self.STATE_IDLE:
            self._do_start()
        elif state == self.STATE_TEACHING:
            self._do_stop()
        # In SAVING state: ignore button press

    # ─────────────────────────────────────────────────────────────────────────
    # SERVICES
    # ─────────────────────────────────────────────────────────────────────────

    def _srv_start(self, req):
        """rosservice call /teach/start"""
        with self._lock:
            state = self._state
        if state != self.STATE_IDLE:
            return TriggerResponse(
                success=False,
                message=f"Cannot start: currently in state {state}")
        self._do_start()
        return TriggerResponse(success=True, message="Teaching started")

    def _srv_stop(self, req):
        """rosservice call /teach/stop"""
        with self._lock:
            state = self._state
        if state != self.STATE_TEACHING:
            return TriggerResponse(
                success=False,
                message=f"Cannot stop: currently in state {state}")
        self._do_stop()
        return TriggerResponse(success=True,
                               message="Teaching stopped — saving graph …")

    # ─────────────────────────────────────────────────────────────────────────
    # STATE TRANSITIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _do_start(self):
        """Transition IDLE → TEACHING."""
        with self._lock:
            self._state         = self.STATE_TEACHING
            self._collected     = []
            self._teach_start_t = time.time()
            # Reset scorer so last-accepted-frame comparison starts fresh
            self._scorer        = KeyframeScorer(
                                      self._orb,
                                      score_threshold=self._score_threshold)
            self._graph         = MemoryGraph(self._orb)

        rospy.loginfo("━━━ TEACHING STARTED ━━━  Drive the robot now.")
        rospy.loginfo(f"  Press Joy button {self._joy_button} again to stop.")

    def _do_stop(self):
        """
        Transition TEACHING → SAVING.
        Launches graph building in a background thread so the ROS
        spinner is never blocked.
        """
        with self._lock:
            self._state = self.STATE_SAVING
            frames      = list(self._collected)   # snapshot

        duration = time.time() - self._teach_start_t
        n_frames  = len(frames)
        rospy.loginfo(f"━━━ TEACHING STOPPED ━━━  {n_frames} keyframes "
                      f"in {duration:.1f}s  →  building graph …")

        t = threading.Thread(target=self._build_and_save,
                             args=(frames,),
                             daemon=True,
                             name="GraphBuilder")
        t.start()

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH BUILD + SAVE  (background thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_and_save(self, frames):
        """
        Runs Algorithm 1 (MemoryGraph.build_from_frames) then saves:
          teach_memory/graph.pkl          ← the MemoryGraph object
          teach_memory/teach_frames.pkl   ← raw FrameData list (for ablation)
          teach_memory/metadata.json      ← run stats (human-readable)
        """
        t0 = time.time()

        if len(frames) < 2:
            rospy.logwarn("Too few frames to build graph (need ≥ 2). "
                          "Did you drive long enough?")
            with self._lock:
                self._state = self.STATE_IDLE
            return

        # ── Algorithm 1 ───────────────────────────────────────────────────────
        rospy.loginfo(f"Building graph from {len(frames)} keyframes …")
        self._graph.build_from_frames(frames, verbose=False)
        build_time = time.time() - t0

        # ── Save graph.pkl ────────────────────────────────────────────────────
        graph_path = os.path.join(self._save_dir, self._graph_filename)
        self._graph.save(graph_path)

        # ── Save raw frames (needed for ablation study and VO trail) ──────────
        frames_path = os.path.join(self._save_dir, "teach_frames.pkl")
        with open(frames_path, "wb") as f:
            pickle.dump(frames, f, protocol=pickle.HIGHEST_PROTOCOL)
        frames_kb = os.path.getsize(frames_path) / 1024

        # ── Save human-readable metadata ─────────────────────────────────────
        stats = self._graph.stats() if hasattr(self._graph, "stats") else {}
        metadata = {
            "timestamp":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_teach_frames":  len(frames),
            "n_graph_nodes":   len(self._graph.nodes),
            "n_graph_edges":   sum(len(n.edges)
                                   for n in self._graph.nodes.values()),
            "build_time_s":    round(build_time, 2),
            "teach_duration_s": round(time.time() - self._teach_start_t, 1),
            "acceptance_rate": round(self._scorer.acceptance_rate, 3),
            "frames_seen":     self._frames_seen,
            "n_features":      self._n_features,
            "score_threshold": self._score_threshold,
            "clip_limit":      self._clip_limit,
        }
        meta_path = os.path.join(self._save_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        rospy.loginfo("━━━ GRAPH SAVED ━━━")
        rospy.loginfo(f"  Nodes      : {len(self._graph.nodes)}")
        rospy.loginfo(f"  Dir edges  : {metadata['n_graph_edges']}")
        rospy.loginfo(f"  Build time : {build_time:.1f}s")
        rospy.loginfo(f"  graph.pkl  : {graph_path}")
        rospy.loginfo(f"  frames.pkl : {frames_path}  ({frames_kb:.0f} KB)")
        rospy.loginfo(f"  metadata   : {meta_path}")

        with self._lock:
            self._state = self.STATE_IDLE

        rospy.loginfo("Ready for next teach run.  Press Joy button to start.")

    # ────────────────────────────────────────────────────────────────────────
    # PREVIEW BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def _make_preview(self, bgr_undist, fd, sf, state):
        """
        Annotated preview frame published on /teach/preview.

        Annotations:
          • ORB keypoints coloured by accept (green) / reject (red)
          • 4×4 grid lines (spatial entropy reference)
          • Top-left HUD: state, keyframe count, score, reason
          • Top-right: flashing red REC indicator when teaching
          • Bottom: score bars (count | entropy | combined)
        """
        out = bgr_undist.copy()
        h, w = out.shape[:2]

        # ── Keypoints ─────────────────────────────────────────────────────────
        kp_col = (0, 200, 0) if sf.accepted else (0, 60, 220)
        for kp in fd.keypoints:
            cv2.circle(out, (int(kp.pt[0]), int(kp.pt[1])), 3, kp_col, -1)

        # ── 4×4 grid (entropy reference) ──────────────────────────────────────
        for r in range(1, 4):
            cv2.line(out, (0, r * h // 4), (w, r * h // 4), (40, 40, 40), 1)
        for c in range(1, 4):
            cv2.line(out, (c * w // 4, 0), (c * w // 4, h), (40, 40, 40), 1)

        # ── HUD box ───────────────────────────────────────────────────────────
        with self._lock:
            n_collected = len(self._collected)
            accept_rate = self._scorer.acceptance_rate

        lines = [
            f"STATE: {state}",
            f"KF saved: {n_collected}  ({accept_rate:.0%} accepted)",
            f"Features: {fd.n_features}",
            f"Score: {sf.combined_score:.2f}  "
            f"(cnt={sf.count_score:.2f} ent={sf.entropy_score:.2f})",
        ]
        if not sf.accepted:
            lines.append(f"REJECT: {sf.reject_reason[:35]}")

        box_h = len(lines) * 18 + 8
        cv2.rectangle(out, (4, 4), (380, box_h), (0, 0, 0), -1)
        for i, line in enumerate(lines):
            cv2.putText(out, line, (8, 20 + i * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

        # ── REC indicator (flashing at ~1 Hz) ────────────────────────────────
        if state == self.STATE_TEACHING:
            if int(time.time() * 2) % 2 == 0:   # blink every 0.5 s
                cv2.circle(out, (w - 20, 20), 10, (0, 0, 220), -1)
                cv2.putText(out, "REC", (w - 55, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 2)
        elif state == self.STATE_SAVING:
            cv2.putText(out, "SAVING…", (w - 100, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        # ── Score bars (bottom strip) ─────────────────────────────────────────
        bar_y  = h - 20
        bar_h  = 10
        bar_w  = 120

        def draw_bar(x, val, colour, label):
            cv2.rectangle(out, (x, bar_y), (x + bar_w, bar_y + bar_h),
                          (40, 40, 40), -1)
            fill = int(bar_w * max(0.0, min(val, 1.0)))
            if fill > 0:
                cv2.rectangle(out, (x, bar_y),
                              (x + fill, bar_y + bar_h), colour, -1)
            cv2.putText(out, f"{label} {val:.2f}",
                        (x, bar_y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        draw_bar(4,   sf.count_score,   (0, 220, 255), "cnt")
        draw_bar(132, sf.entropy_score, (255, 200, 0), "ent")
        draw_bar(260, sf.combined_score,(80, 255, 80), "comb")

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # STATUS TIMER  (publishes JSON to /teach/status at 2 Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _cb_status_timer(self, event):
        """
        Publishes a JSON string on /teach/status.
        Useful for a ROS dashboard or custom monitoring node.

        JSON fields:
          state, n_collected, n_nodes, frames_seen,
          acceptance_rate, teach_duration_s
        """
        with self._lock:
            state       = self._state
            n_collected = len(self._collected)
            n_nodes     = len(self._graph.nodes)
            frames_seen = self._frames_seen
            accept_rate = self._scorer.acceptance_rate

        duration = (time.time() - self._teach_start_t
                    if state == self.STATE_TEACHING else 0.0)

        status = {
            "state":            state,
            "n_collected":      n_collected,
            "n_nodes":          n_nodes,
            "frames_seen":      frames_seen,
            "acceptance_rate":  round(accept_rate, 3),
            "teach_duration_s": round(duration, 1),
            "stamp":            time.time(),
        }
        self._pub_status.publish(String(data=json.dumps(status)))

    # ─────────────────────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────────────────────

    def _on_shutdown(self):
        """
        rospy calls this on Ctrl-C / rosnode kill.
        If teaching is active, save whatever we have.
        """
        with self._lock:
            state   = self._state
            frames  = list(self._collected)

        if state == self.STATE_TEACHING and len(frames) >= 2:
            rospy.logwarn("Shutdown during teaching — saving partial graph …")
            self._do_stop()
            time.sleep(2.0)   # give background thread time to finish
        else:
            rospy.loginfo("TeachLoggerNode: clean shutdown.")

    def spin(self):
        rospy.spin()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        node = TeachLoggerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
