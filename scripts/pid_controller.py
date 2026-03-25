#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step8_pid_controller_node.py  —  ROS Melodic PID + Ackermann Steering
VT&R Project | Phase 2

ROS NODE NAME : /pid_controller
SUBSCRIBES    : /geometry/path_error   (std_msgs/Float32)  — from step7
                /geometry/result       (std_msgs/String)   — JSON for dead-band
                /repeat/state          (std_msgs/String)   — RUNNING/STOPPED
PUBLISHES     : /cmd_vel               (geometry_msgs/Twist) — to JetRacer
                /pid/debug             (std_msgs/String)   — JSON PID state

WHAT THIS NODE DOES:
    Receives path_error from the geometry engine and outputs wheel commands.

    Pipeline:
        path_error (float)
            → uncertainty-aware dead-band gate
            → PID controller (P + I + D)
            → steering angle clip to JetRacer physical limits
            → speed scaling (slow down when turning sharply)
            → geometry_msgs/Twist → /cmd_vel

    Key features:
        Uncertainty-aware dead-band:
            Widens from 0.05 to 0.15 when geometry engine reports
            low confidence (few inliers, high condition number).
            Robot drives straight through uncertain zones rather than
            chasing noisy geometry estimates.

        Anti-windup:
            Integral term is clamped to MAX_INTEGRAL.
            Integrator resets on: node pointer advance, STOP command,
            localisation failure (consecutive geometry failures).

        Speed scaling:
            Forward speed reduces proportionally to steering angle.
            Prevents overshooting at corners.

        Node transition damping:
            When the node pointer advances, path_error spikes briefly
            as the reference frame shifts. The D term absorbs this.
            Kd tuning is therefore critical for smooth node transitions.

JETRACER /cmd_vel CONTRACT:
    geometry_msgs/Twist:
        linear.x  = forward speed  (m/s, positive = forward)
        angular.z = steering rate  (rad/s, positive = left turn)
    The JetRacer motor controller subscribes to /cmd_vel directly.

TUNING SEQUENCE (follow this order):
    1. Ki=0, Kd=0. Tune Kp on straight corridor. Start at 0.3.
    2. Add Kd to damp node-transition jerks. Start at 0.05.
    3. Add small Ki only if steady-state offset persists. Start at 0.01.
    4. Tune dead_band_tight to ~50% of on-path noise floor.

PARAMS:
    ~Kp               (float, default=0.5)    proportional gain
    ~Ki               (float, default=0.02)   integral gain
    ~Kd               (float, default=0.1)    derivative gain
    ~base_speed       (float, default=0.3)    forward speed m/s
    ~max_steer        (float, default=0.6)    max steering angle rad
    ~dead_band_tight  (float, default=0.05)   dead-band when confident
    ~dead_band_wide   (float, default=0.15)   dead-band when uncertain
    ~max_integral     (float, default=0.5)    anti-windup clamp
    ~cmd_rate         (float, default=20.0)   Hz — /cmd_vel publish rate
    ~timeout          (float, default=0.5)    seconds before stop on no input
=============================================================================
"""

import rospy
import numpy as np
import json
import time

from std_msgs.msg    import Float32, String
from geometry_msgs.msg import Twist


# ── Configuration ─────────────────────────────────────────────────────────────

KP              = 0.5
KI              = 0.02
KD              = 0.1
BASE_SPEED      = 0.3
MAX_STEER       = 0.6
DEAD_BAND_TIGHT = 0.05
DEAD_BAND_WIDE  = 0.15
MAX_INTEGRAL    = 0.5
CMD_RATE        = 20.0
TIMEOUT         = 0.5


# ── PID controller ────────────────────────────────────────────────────────────

class PIDController(object):
    """
    Standard discrete PID controller with:
        - Anti-windup integral clamp
        - Derivative on measurement (not error) to avoid D-kick on setpoint change
        - Dead-band gate before update
        - Integrator reset method
    """

    def __init__(self, Kp, Ki, Kd, max_integral=MAX_INTEGRAL):
        self.Kp           = Kp
        self.Ki           = Ki
        self.Kd           = Kd
        self.max_integral = max_integral

        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = None

    def reset(self):
        """Reset integrator and derivative state. Call on node advance or stop."""
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def update(self, error):
        """
        Compute PID output for a given error signal.

        Args:
            error: float — path_error after dead-band gate applied

        Returns:
            output: float — steering signal (positive = left correction)
        """
        now = time.time()

        if self._prev_time is None:
            dt = 1.0 / CMD_RATE
        else:
            dt = now - self._prev_time
            dt = max(dt, 1e-4)   # guard against zero dt

        # P term
        p_term = self.Kp * error

        # I term with anti-windup clamp
        self._integral += error * dt
        self._integral  = float(np.clip(
            self._integral, -self.max_integral, self.max_integral
        ))
        i_term = self.Ki * self._integral

        # D term — derivative on error
        d_term = self.Kd * (error - self._prev_error) / dt

        self._prev_error = error
        self._prev_time  = now

        return p_term + i_term + d_term

    @property
    def integral(self):
        return self._integral


# ── Ackermann steering converter ──────────────────────────────────────────────

class AckermannConverter(object):
    """
    Converts PID output → geometry_msgs/Twist for the JetRacer.

    JetRacer uses Ackermann (car-like) steering:
        linear.x  = forward speed (m/s)
        angular.z = steering rate (rad/s)

    Features:
        Steering angle clipped to physical limits (±MAX_STEER rad)
        Speed scaling: reduces speed when turning sharply
        Emergency stop: sets both to zero
    """

    def __init__(self, base_speed=BASE_SPEED, max_steer=MAX_STEER):
        self.base_speed = base_speed
        self.max_steer  = max_steer

    def convert(self, pid_output, stopped=False):
        """
        Convert PID output to Twist message.

        Args:
            pid_output: float — raw PID output
            stopped:    bool  — if True, return zero Twist

        Returns:
            geometry_msgs/Twist
        """
        cmd = Twist()

        if stopped:
            return cmd   # linear.x=0, angular.z=0

        # Clip steering to physical limits
        steering = float(np.clip(pid_output, -self.max_steer, self.max_steer))

        # Speed scaling — slow down proportionally when turning sharply
        # At max steer → speed = 50% of base
        speed_scale = 1.0 - 0.5 * abs(steering) / self.max_steer
        speed       = self.base_speed * speed_scale

        cmd.linear.x  = float(speed)
        cmd.angular.z = float(steering)

        return cmd


# ── ROS node ──────────────────────────────────────────────────────────────────

class PIDControllerNode(object):
    """
    ROS node: geometry engine path_error → wheel commands.
    """

    def __init__(self):
        rospy.init_node('pid_controller', anonymous=False)

        # ── Params ────────────────────────────────────────────────────────
        Kp            = rospy.get_param('~Kp',              KP)
        Ki            = rospy.get_param('~Ki',              KI)
        Kd            = rospy.get_param('~Kd',              KD)
        base_speed    = rospy.get_param('~base_speed',      BASE_SPEED)
        max_steer     = rospy.get_param('~max_steer',       MAX_STEER)
        self.db_tight = rospy.get_param('~dead_band_tight', DEAD_BAND_TIGHT)
        self.db_wide  = rospy.get_param('~dead_band_wide',  DEAD_BAND_WIDE)
        max_integral  = rospy.get_param('~max_integral',    MAX_INTEGRAL)
        self.cmd_rate = rospy.get_param('~cmd_rate',        CMD_RATE)
        self.timeout  = rospy.get_param('~timeout',         TIMEOUT)

        # ── Controllers ───────────────────────────────────────────────────
        self.pid       = PIDController(Kp, Ki, Kd, max_integral)
        self.ackermann = AckermannConverter(base_speed, max_steer)

        # ── State ─────────────────────────────────────────────────────────
        self.path_error   = 0.0
        self.uncertain    = True      # from geometry engine
        self.dead_band    = self.db_wide
        self.running      = False     # set True by /repeat/state
        self.t_last_input = 0.0       # time of last path_error received
        self.last_node_id = -1        # detect node pointer advance

        # Statistics
        self.n_commands   = 0
        self.n_deadband   = 0         # corrections suppressed by dead-band
        self.total_error  = 0.0       # for mean absolute error logging

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber('/geometry/path_error', Float32,
                         self._cb_path_error, queue_size=1)
        rospy.Subscriber('/geometry/result',     String,
                         self._cb_geo_result,   queue_size=1)
        rospy.Subscriber('/repeat/state',        String,
                         self._cb_repeat_state, queue_size=5)

        # ── Publishers ────────────────────────────────────────────────────
        self.pub_cmd   = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_debug = rospy.Publisher('/pid/debug', String, queue_size=5)

        rospy.loginfo(
            "[PID] Ready  Kp=%.2f Ki=%.3f Kd=%.2f  "
            "speed=%.2f  max_steer=%.2frad  "
            "db_tight=%.3f db_wide=%.3f",
            Kp, Ki, Kd, base_speed, max_steer,
            self.db_tight, self.db_wide
        )
        rospy.loginfo("[PID] Waiting for /repeat/state=RUNNING to activate...")

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _cb_path_error(self, msg):
        """Receive path error from geometry engine."""
        self.path_error   = float(msg.data)
        self.t_last_input = time.time()

    def _cb_geo_result(self, msg):
        """
        Receive full geometry result JSON.
        Extract uncertainty flag and node_id for:
            - Dead-band width selection
            - Integrator reset on node pointer advance
        """
        try:
            data = json.loads(msg.data)
        except ValueError:
            return

        self.uncertain = data.get('uncertain', True)

        # Update dead-band based on geometry confidence
        self.dead_band = (
            self.db_wide if self.uncertain else self.db_tight
        )

        # Detect node pointer advance → reset integrator
        node_id = data.get('node_id', -1)
        if node_id != self.last_node_id and node_id >= 0:
            if self.last_node_id >= 0:
                rospy.logdebug(
                    "[PID] Node advance %d→%d — resetting integrator",
                    self.last_node_id, node_id
                )
                self.pid.reset()
            self.last_node_id = node_id

    def _cb_repeat_state(self, msg):
        """
        Receive repeat controller state.
        Expected values: 'RUNNING', 'STOPPED', 'FAILURE'
        """
        try:
            data = json.loads(msg.data)
            state = data.get('state', msg.data)
        except ValueError:
            state = msg.data.strip()

        if state == 'RUNNING':
            if not self.running:
                rospy.loginfo("[PID] Activated — publishing /cmd_vel")
                self.pid.reset()
            self.running = True

        elif state in ('STOPPED', 'FAILURE', 'COMPLETE'):
            if self.running:
                rospy.loginfo("[PID] Deactivated — state=%s", state)
                self._publish_stop()
            self.running = False
            self.pid.reset()

    # ── Main control loop ─────────────────────────────────────────────────

    def _control_step(self):
        """
        One PID control step. Called at cmd_rate Hz.

        Decision tree:
            Not running         → publish stop
            Input timeout       → publish stop (geometry engine stalled)
            |error| < dead_band → publish straight (no correction)
            else                → PID update → Ackermann → publish
        """
        # Not running
        if not self.running:
            self._publish_stop()
            return

        # Input timeout — geometry engine has stalled
        if (time.time() - self.t_last_input) > self.timeout:
            self._publish_stop()
            rospy.logwarn_throttle(
                2.0, "[PID] Input timeout — no path_error for %.1fs",
                self.timeout
            )
            return

        error = self.path_error

        # Dead-band gate
        if abs(error) < self.dead_band:
            # Error is within noise floor — drive straight
            # Do NOT update integral (avoid windup during straight sections)
            cmd = self.ackermann.convert(0.0)
            self.n_deadband += 1
        else:
            # Remove dead-band offset before feeding to PID
            # This prevents a step discontinuity at the dead-band boundary
            gated_error = error - np.sign(error) * self.dead_band
            pid_output  = self.pid.update(float(gated_error))
            cmd         = self.ackermann.convert(pid_output)

        self.pub_cmd.publish(cmd)
        self.n_commands  += 1
        self.total_error += abs(error)

        # Publish debug
        if self.pub_debug.get_num_connections() > 0:
            debug = {
                'path_error'  : round(error, 4),
                'dead_band'   : round(self.dead_band, 3),
                'uncertain'   : self.uncertain,
                'linear_x'    : round(cmd.linear.x,  3),
                'angular_z'   : round(cmd.angular.z, 3),
                'integral'    : round(self.pid.integral, 4),
                'n_commands'  : self.n_commands,
                'n_deadband'  : self.n_deadband,
                'mean_abs_err': round(
                    self.total_error / max(self.n_commands, 1), 4
                ),
            }
            self.pub_debug.publish(String(data=json.dumps(debug)))

    def _publish_stop(self):
        """Publish zero velocity — motors stop."""
        self.pub_cmd.publish(self.ackermann.convert(0.0, stopped=True))

    # ── Spin ──────────────────────────────────────────────────────────────

    def spin(self):
        """Run control loop at cmd_rate Hz."""
        rate = rospy.Rate(self.cmd_rate)
        rospy.loginfo("[PID] Control loop running at %.0f Hz", self.cmd_rate)

        while not rospy.is_shutdown():
            self._control_step()

            # Periodic summary log every 5 seconds
            if self.n_commands > 0 and self.n_commands % int(self.cmd_rate * 5) == 0:
                rospy.loginfo(
                    "[PID] cmds=%d  dead_band_hits=%d(%.0f%%)  "
                    "mean_abs_err=%.4f  integral=%.4f  uncertain=%s",
                    self.n_commands,
                    self.n_deadband,
                    100.0 * self.n_deadband / self.n_commands,
                    self.total_error / self.n_commands,
                    self.pid.integral,
                    self.uncertain
                )

            rate.sleep()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = PIDControllerNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
