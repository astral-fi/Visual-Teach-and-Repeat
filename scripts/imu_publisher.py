#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step10_imu_node.py  —  Waveshare JetRacer RP2040 IMU (Binary Protocol)
VT&R Project | Phase 3

PROTOCOL:
    The RP2040 expansion board streams WitMotion/Waveshare binary packets.
    Each packet is exactly 11 bytes:

        Byte 0    : 0x55  (header, always)
        Byte 1    : packet type
                      0x51 = accelerometer
                      0x52 = gyroscope
                      0x53 = angle (euler)
                      0x54 = magnetometer (not used)
        Bytes 2-9 : 4 x signed 16-bit little-endian values
        Byte 10   : checksum = (sum of bytes 0-9) & 0xFF

    Accelerometer packet (type 0x51):
        ax_raw = int16(bytes 2-3)  →  ax = ax_raw / 32768.0 * 16 * 9.8 m/s²
        ay_raw = int16(bytes 4-5)  →  ay = ay_raw / 32768.0 * 16 * 9.8 m/s²
        az_raw = int16(bytes 6-7)  →  az = az_raw / 32768.0 * 16 * 9.8 m/s²
        temp   = int16(bytes 8-9)  →  not used

    Gyroscope packet (type 0x52):
        gx_raw = int16(bytes 2-3)  →  gx = gx_raw / 32768.0 * 2000 deg/s
        gy_raw = int16(bytes 4-5)  →  gy = gy_raw / 32768.0 * 2000 deg/s
        gz_raw = int16(bytes 6-7)  →  gz = gz_raw / 32768.0 * 2000 deg/s
        temp   = int16(bytes 8-9)  →  not used

ROS NODE NAME : /imu_publisher
PUBLISHES     : /imu/data  (sensor_msgs/Imu)

SETUP:
    sudo usermod -a -G dialout $USER
    logout and login
    pip install pyserial --break-system-packages

PARAMS:
    ~serial_port   (str,   default='/dev/ttyACM0')
    ~serial_baud   (int,   default=9600)    WitMotion default is 9600
    ~publish_rate  (float, default=100.0)
    ~frame_id      (str,   default='imu')
    ~verbose       (bool,  default=False)   print each parsed packet
=============================================================================
"""

import rospy
import math
import time
import struct

from sensor_msgs.msg import Imu

GRAVITY = 9.80665

# WitMotion packet constants
HEADER      = 0x55
PKT_ACCEL   = 0x51
PKT_GYRO    = 0x52
PKT_ANGLE   = 0x53
PKT_LEN     = 11

# Scale factors (WitMotion default full-scale ranges)
ACCEL_SCALE = 32768.0 / (16.0 * GRAVITY)   # raw → m/s²  (±16g range)
GYRO_SCALE  = 32768.0 / 2000.0             # raw → deg/s (±2000 dps range)


# ── Binary packet parser ──────────────────────────────────────────────────────

class WitMotionParser(object):
    """
    Stateful parser for WitMotion binary serial stream.
    Handles partial packets, byte alignment, and checksum validation.
    """

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data):
        """
        Feed raw bytes into the parser.
        Returns list of parsed packets as dicts.
        Each dict: {'type': 0x51, 'values': [v0, v1, v2, v3]}
        """
        self._buf.extend(bytearray(data))
        packets = []

        while len(self._buf) >= PKT_LEN:
            # Find next 0x55 header
            if self._buf[0] != HEADER:
                self._buf = self._buf[1:]
                continue

            # Check if we have a full packet
            if len(self._buf) < PKT_LEN:
                break

            pkt = self._buf[:PKT_LEN]

            # Validate checksum
            expected = sum(pkt[:10]) & 0xFF
            actual   = pkt[10]

            if expected != actual:
                # Bad checksum — not a real packet start, skip byte
                self._buf = self._buf[1:]
                continue

            # Valid packet
            pkt_type = pkt[1]
            v = []
            for i in range(4):
                raw = struct.unpack_from('<h', bytes(pkt), 2 + i*2)[0]
                v.append(raw)

            packets.append({'type': pkt_type, 'values': v})
            self._buf = self._buf[PKT_LEN:]

        return packets


# ── IMU state (accumulates accel + gyro across separate packets) ──────────────

class IMUState(object):
    """
    The WitMotion protocol sends accel and gyro in separate packets.
    This class accumulates both and provides a complete reading.
    """

    def __init__(self):
        self.ax = 0.0
        self.ay = 0.0
        self.az = GRAVITY
        self.gx = 0.0
        self.gy = 0.0
        self.gz = 0.0
        self.accel_ready = False
        self.gyro_ready  = False
        self.t_last      = 0.0

    def update(self, pkt):
        """Process one parsed packet."""
        vals = pkt['values']
        t    = pkt['type']

        if t == PKT_ACCEL:
            # ax, ay, az in raw units → m/s²
            # Scale: raw / 32768 * 16g
            self.ax = vals[0] / 32768.0 * 16.0 * GRAVITY
            self.ay = vals[1] / 32768.0 * 16.0 * GRAVITY
            self.az = vals[2] / 32768.0 * 16.0 * GRAVITY
            self.accel_ready = True

        elif t == PKT_GYRO:
            # gx, gy, gz in raw units → rad/s
            # Scale: raw / 32768 * 2000 deg/s → rad/s
            self.gx = math.radians(vals[0] / 32768.0 * 2000.0)
            self.gy = math.radians(vals[1] / 32768.0 * 2000.0)
            self.gz = math.radians(vals[2] / 32768.0 * 2000.0)
            self.gyro_ready = True

    @property
    def ready(self):
        """True when both accel and gyro have been updated at least once."""
        return self.accel_ready and self.gyro_ready

    def get(self):
        """Return (ax, ay, az, gx, gy, gz) in SI units."""
        return self.ax, self.ay, self.az, self.gx, self.gy, self.gz


# ── ROS node ──────────────────────────────────────────────────────────────────

class IMUNode(object):

    def __init__(self):
        rospy.init_node('imu_publisher', anonymous=False)

        self.port     = rospy.get_param('~serial_port',  '/dev/ttyACM0')
        self.baud     = rospy.get_param('~serial_baud',  9600)
        self.rate_hz  = rospy.get_param('~publish_rate', 100.0)
        self.frame    = rospy.get_param('~frame_id',     'imu')
        self.verbose  = rospy.get_param('~verbose',      False)

        self.ser      = None
        self.parser   = WitMotionParser()
        self.state    = IMUState()

        self._open_serial()

        self.pub = rospy.Publisher('/imu/data', Imu, queue_size=10)

        # Covariance matrices for MPU9250 via WitMotion firmware
        self.accel_cov  = [0.04,   0, 0,  0, 0.04,   0,  0, 0, 0.04]
        self.gyro_cov   = [0.0002, 0, 0,  0, 0.0002, 0,  0, 0, 0.0002]
        self.orient_cov = [-1.0] + [0.0] * 8   # orientation unknown

        rospy.loginfo("[IMU] WitMotion binary parser ready")
        rospy.loginfo("[IMU] port=%s  baud=%d  rate=%.0fHz",
                      self.port, self.baud, self.rate_hz)
        rospy.loginfo("[IMU] Publishing to /imu/data")
        rospy.loginfo("[IMU] Verify: rostopic echo /imu/data -n 5")

    def _open_serial(self):
        try:
            import serial
            self.ser = serial.Serial(
                self.port,
                self.baud,
                timeout=0.01,     # 10ms read timeout — non-blocking
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self.ser.flushInput()
            rospy.loginfo("[IMU] Opened %s at %d baud", self.port, self.baud)
        except ImportError:
            rospy.logerr("[IMU] pyserial missing.")
            rospy.logerr("[IMU]   pip install pyserial --break-system-packages")
            rospy.logerr("[IMU] Running in simulation mode.")
        except Exception as e:
            rospy.logerr("[IMU] Cannot open %s: %s", self.port, str(e))
            rospy.logerr("[IMU] Check: ls -la /dev/ttyACM*")
            rospy.logerr("[IMU] Check: sudo usermod -a -G dialout $USER")
            rospy.logerr("[IMU] Running in simulation mode.")

    def _sim_read(self):
        """Simulated IMU for testing without hardware."""
        t = time.time()
        return (
            0.01 * math.sin(t),
            0.01 * math.cos(t),
            GRAVITY,
            0.001 * math.sin(2*t),
            0.001 * math.cos(2*t),
            0.0,
        )

    def spin(self):
        rate   = rospy.Rate(self.rate_hz)
        seq    = 0
        n_pkts = 0
        n_bad  = 0

        rospy.loginfo("[IMU] Spinning...")

        while not rospy.is_shutdown():

            # Read available bytes from serial
            if self.ser is not None and self.ser.is_open:
                try:
                    waiting = self.ser.in_waiting
                    if waiting > 0:
                        raw = self.ser.read(waiting)
                        pkts = self.parser.feed(raw)
                        for pkt in pkts:
                            self.state.update(pkt)
                            n_pkts += 1
                            if self.verbose:
                                rospy.logdebug(
                                    "[IMU] pkt type=0x%02X vals=%s",
                                    pkt['type'], str(pkt['values'])
                                )
                except Exception as e:
                    rospy.logwarn_throttle(5.0, "[IMU] Serial read error: %s", str(e))

            # Get current IMU values
            if self.state.ready and self.ser is not None:
                ax, ay, az, gx, gy, gz = self.state.get()
            else:
                ax, ay, az, gx, gy, gz = self._sim_read()

            # Build and publish Imu message
            msg                    = Imu()
            msg.header.stamp       = rospy.Time.now()
            msg.header.frame_id    = self.frame
            msg.header.seq         = seq

            msg.orientation_covariance         = self.orient_cov
            msg.angular_velocity.x             = gx
            msg.angular_velocity.y             = gy
            msg.angular_velocity.z             = gz
            msg.angular_velocity_covariance    = self.gyro_cov
            msg.linear_acceleration.x          = ax
            msg.linear_acceleration.y          = ay
            msg.linear_acceleration.z          = az
            msg.linear_acceleration_covariance = self.accel_cov

            self.pub.publish(msg)
            seq += 1

            # Periodic status log
            if seq % int(self.rate_hz * 10) == 0 and seq > 0:
                rospy.loginfo(
                    "[IMU] packets=%d  ax=%.3f ay=%.3f az=%.3f  "
                    "gx=%.4f gy=%.4f gz=%.4f",
                    n_pkts, ax, ay, az, gx, gy, gz
                )

            rate.sleep()


if __name__ == '__main__':
    try:
        node = IMUNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass
