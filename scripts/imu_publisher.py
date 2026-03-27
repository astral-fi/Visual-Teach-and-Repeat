#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
=============================================================================
step10_imu_node.py  —  ROS Melodic MPU9250 IMU Publisher
VT&R Project | Phase 3

ROS NODE NAME : /imu_publisher
PUBLISHES     : /imu/data    (sensor_msgs/Imu)  — raw IMU for EKF

WHAT THIS NODE DOES:
    Reads the MPU9250 IMU over I2C from the Jetson Nano and publishes
    sensor_msgs/Imu messages for the robot_localization EKF node.

    The MPU9250 provides:
        Accelerometer : linear acceleration (m/s²) — 3 axes
        Gyroscope     : angular velocity (rad/s)   — 3 axes
        Magnetometer  : not used (compass unreliable indoors)

    For VT&R we primarily use:
        angular_velocity.z  — yaw rate (robot turning)
        linear_acceleration.x — forward acceleration (speed changes)

I2C WIRING on Jetson Nano:
    MPU9250 VCC  → 3.3V (Pin 1)
    MPU9250 GND  → GND  (Pin 6)
    MPU9250 SCL  → SCL  (Pin 5 / GPIO3)
    MPU9250 SDA  → SDA  (Pin 3 / GPIO2)
    I2C address  : 0x68 (AD0 low) or 0x69 (AD0 high)

DEPENDENCIES:
    pip install smbus2 --break-system-packages
    sudo usermod -a -G i2c $USER  (then logout/login)
    sudo modprobe i2c-dev

PARAMS:
    ~i2c_bus        (int,   default=1)     I2C bus number (/dev/i2c-1)
    ~i2c_addr       (int,   default=0x68)  MPU9250 I2C address
    ~publish_rate   (int,   default=100)   Hz (MPU9250 supports up to 1000)
    ~frame_id       (str,   default='imu') TF frame ID
    ~accel_scale    (int,   default=2)     ±2g, 4g, 8g, or 16g
    ~gyro_scale     (int,   default=250)   ±250, 500, 1000, or 2000 dps
=============================================================================
"""

import rospy
import struct
import math
import time

from sensor_msgs.msg import Imu
from std_msgs.msg    import Header

try:
    import smbus2 as smbus
    SMBUS_AVAILABLE = True
except ImportError:
    SMBUS_AVAILABLE = False
    rospy.logwarn_once("[IMU] smbus2 not installed — running in simulation mode")


# ── MPU9250 register map ──────────────────────────────────────────────────────

MPU9250_ADDR      = 0x68
PWR_MGMT_1        = 0x6B
ACCEL_CONFIG      = 0x1C
GYRO_CONFIG       = 0x1B
ACCEL_XOUT_H      = 0x3B
GYRO_XOUT_H       = 0x43
WHO_AM_I          = 0x75
WHO_AM_I_RESPONSE = 0x71   # MPU9250 returns 0x71

# Sensitivity scales
ACCEL_SCALE_MAP = {2: (0x00, 16384.0),
                   4: (0x08,  8192.0),
                   8: (0x10,  4096.0),
                   16:(0x18,  2048.0)}

GYRO_SCALE_MAP  = {250:  (0x00, 131.0),
                   500:  (0x08,  65.5),
                   1000: (0x10,  32.8),
                   2000: (0x18,  16.4)}

GRAVITY = 9.80665  # m/s²


# ── MPU9250 driver ────────────────────────────────────────────────────────────

class MPU9250Driver(object):
    """
    Minimal MPU9250 I2C driver.
    Reads raw accelerometer and gyroscope values and converts to SI units.
    """

    def __init__(self, bus_num=1, addr=MPU9250_ADDR,
                 accel_scale=2, gyro_scale=250):
        self.addr         = addr
        self.accel_scale  = accel_scale
        self.gyro_scale   = gyro_scale

        if not SMBUS_AVAILABLE:
            rospy.logwarn("[IMU] smbus2 unavailable — publishing simulated data")
            self.bus = None
            return

        try:
            self.bus = smbus.SMBus(bus_num)
        except Exception as e:
            rospy.logerr("[IMU] Cannot open I2C bus %d: %s", bus_num, str(e))
            self.bus = None
            return

        self._init_device()

    def _write(self, reg, val):
        try:
            self.bus.write_byte_data(self.addr, reg, val)
        except Exception as e:
            rospy.logerr("[IMU] I2C write error reg=0x%02X: %s", reg, str(e))

    def _read_bytes(self, reg, length):
        try:
            return self.bus.read_i2c_block_data(self.addr, reg, length)
        except Exception as e:
            rospy.logerr("[IMU] I2C read error reg=0x%02X: %s", reg, str(e))
            return [0] * length

    def _init_device(self):
        """Wake up MPU9250 and configure scales."""
        # Wake up — clear sleep bit
        self._write(PWR_MGMT_1, 0x00)
        time.sleep(0.1)

        # Verify WHO_AM_I
        who = self._read_bytes(WHO_AM_I, 1)[0]
        if who != WHO_AM_I_RESPONSE:
            rospy.logwarn("[IMU] WHO_AM_I=0x%02X expected 0x71 — check wiring",
                          who)
        else:
            rospy.loginfo("[IMU] MPU9250 detected (WHO_AM_I=0x71)")

        # Set accelerometer scale
        accel_reg, self._accel_lsb = ACCEL_SCALE_MAP.get(
            self.accel_scale, ACCEL_SCALE_MAP[2])
        self._write(ACCEL_CONFIG, accel_reg)

        # Set gyroscope scale
        gyro_reg, self._gyro_lsb = GYRO_SCALE_MAP.get(
            self.gyro_scale, GYRO_SCALE_MAP[250])
        self._write(GYRO_CONFIG, gyro_reg)

        rospy.loginfo("[IMU] accel=±%dg  gyro=±%ddps", self.accel_scale, self.gyro_scale)

    def _to_signed16(self, high, low):
        """Convert two bytes to signed 16-bit integer."""
        val = (high << 8) | low
        if val >= 0x8000:
            val -= 0x10000
        return val

    def read(self):
        """
        Read accelerometer and gyroscope.
        Returns (ax, ay, az, gx, gy, gz) in SI units (m/s² and rad/s).
        Returns zeros if bus unavailable.
        """
        if self.bus is None:
            # Simulation mode — return small noise
            t = time.time()
            return (
                0.01 * math.sin(t),   # ax
                0.01 * math.cos(t),   # ay
                GRAVITY,              # az (gravity)
                0.001 * math.sin(2*t),# gx
                0.001 * math.cos(2*t),# gy
                0.0,                  # gz
            )

        raw = self._read_bytes(ACCEL_XOUT_H, 14)

        ax_raw = self._to_signed16(raw[0],  raw[1])
        ay_raw = self._to_signed16(raw[2],  raw[3])
        az_raw = self._to_signed16(raw[4],  raw[5])
        # raw[6], raw[7] = temperature (skip)
        gx_raw = self._to_signed16(raw[8],  raw[9])
        gy_raw = self._to_signed16(raw[10], raw[11])
        gz_raw = self._to_signed16(raw[12], raw[13])

        # Convert to SI units
        ax = ax_raw / self._accel_lsb * GRAVITY
        ay = ay_raw / self._accel_lsb * GRAVITY
        az = az_raw / self._accel_lsb * GRAVITY
        gx = math.radians(gx_raw / self._gyro_lsb)
        gy = math.radians(gy_raw / self._gyro_lsb)
        gz = math.radians(gz_raw / self._gyro_lsb)

        return ax, ay, az, gx, gy, gz


# ── ROS node ──────────────────────────────────────────────────────────────────

class IMUNode(object):

    def __init__(self):
        rospy.init_node('imu_publisher', anonymous=False)

        bus_num      = rospy.get_param('~i2c_bus',      1)
        addr         = rospy.get_param('~i2c_addr',     0x68)
        self.rate_hz = rospy.get_param('~publish_rate', 100)
        self.frame   = rospy.get_param('~frame_id',     'imu')
        accel_scale  = rospy.get_param('~accel_scale',  2)
        gyro_scale   = rospy.get_param('~gyro_scale',   250)

        self.imu = MPU9250Driver(
            bus_num     = bus_num,
            addr        = addr,
            accel_scale = accel_scale,
            gyro_scale  = gyro_scale,
        )

        self.pub = rospy.Publisher('/imu/data', Imu, queue_size=10)

        # Covariance matrices (diagonal — independent axes)
        # Tuned for MPU9250 typical noise specs
        self.accel_cov = [
            0.04, 0, 0,
            0, 0.04, 0,
            0, 0, 0.04,
        ]
        self.gyro_cov = [
            0.0002, 0, 0,
            0, 0.0002, 0,
            0, 0, 0.0002,
        ]
        # Orientation unknown from gyro alone
        self.orient_cov = [-1.0] + [0.0] * 8

        rospy.loginfo("[IMU] Publishing /imu/data at %d Hz", self.rate_hz)

    def spin(self):
        rate = rospy.Rate(self.rate_hz)
        seq  = 0

        while not rospy.is_shutdown():
            ax, ay, az, gx, gy, gz = self.imu.read()

            msg = Imu()
            msg.header.stamp    = rospy.Time.now()
            msg.header.frame_id = self.frame
            msg.header.seq      = seq

            # Orientation unknown (no magnetometer fusion)
            msg.orientation_covariance = self.orient_cov

            # Angular velocity (rad/s)
            msg.angular_velocity.x = gx
            msg.angular_velocity.y = gy
            msg.angular_velocity.z = gz
            msg.angular_velocity_covariance = self.gyro_cov

            # Linear acceleration (m/s²)
            msg.linear_acceleration.x = ax
            msg.linear_acceleration.y = ay
            msg.linear_acceleration.z = az
            msg.linear_acceleration_covariance = self.accel_cov

            self.pub.publish(msg)
            seq  += 1
            rate.sleep()


if __name__ == '__main__':
    try:
        node = IMUNode()
        node.spin()
    except rospy.ROSInterruptException:
        pass