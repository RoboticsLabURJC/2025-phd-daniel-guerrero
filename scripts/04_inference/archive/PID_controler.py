#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from carla_msgs.msg import CarlaEgoVehicleControl


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


class PID:
    def __init__(self, kp, ki, kd, i_min=-1e9, i_max=1e9):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i_min = i_min
        self.i_max = i_max
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def step(self, error, dt):
        if dt <= 1e-6:
            return 0.0

        # Integral with clamp (anti-windup simple)
        self.integral += error * dt
        self.integral = clamp(self.integral, self.i_min, self.i_max)

        # Derivative
        if self.prev_error is None:
            deriv = 0.0
        else:
            deriv = (error - self.prev_error) / dt
        self.prev_error = error

        return self.kp * error + self.ki * self.integral + self.kd * deriv


class TwistPIDController(Node):
    def __init__(self):
        super().__init__("twist_pid_controller")

        # ----- Parameters (ajustables en runtime si quieres) -----
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/carla/ego_vehicle/odometry")
        self.declare_parameter("control_topic", "/carla/ego_vehicle/vehicle_control_cmd")

        # Vehicle / mapping params
        self.declare_parameter("wheelbase", 2.7)         # [m] aprox, calibra si quieres
        self.declare_parameter("v_min", 0.8)             # [m/s] para evitar w/v infinito
        self.declare_parameter("delta_max", 0.6)         # [rad] aprox (steer=1.0)
        self.declare_parameter("max_throttle", 0.75)     # [0..1]
        self.declare_parameter("max_brake", 0.8)         # [0..1]

        # PID speed
        self.declare_parameter("kp_v", 0.45)
        self.declare_parameter("ki_v", 0.10)
        self.declare_parameter("kd_v", 0.02)
        self.declare_parameter("i_v_min", -3.0)
        self.declare_parameter("i_v_max",  3.0)

        # Smoothing / limits
        self.declare_parameter("cmd_timeout", 0.5)       # [s] si no llega cmd, frena suave
        self.declare_parameter("steer_rate_limit", 1.2)  # [unit/s] en steer normalizado
        self.declare_parameter("throttle_rate_limit", 1.0)  # [unit/s]
        self.declare_parameter("brake_rate_limit", 2.0)     # [unit/s]
        self.declare_parameter("ema_alpha", 0.3)         # 0..1 (filtro EMA para v_ref,w_ref)

        # Read params
        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self.odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.control_topic = self.get_parameter("control_topic").get_parameter_value().string_value

        self.L = float(self.get_parameter("wheelbase").value)
        self.v_min = float(self.get_parameter("v_min").value)
        self.delta_max = float(self.get_parameter("delta_max").value)
        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_brake = float(self.get_parameter("max_brake").value)

        kp_v = float(self.get_parameter("kp_v").value)
        ki_v = float(self.get_parameter("ki_v").value)
        kd_v = float(self.get_parameter("kd_v").value)
        i_v_min = float(self.get_parameter("i_v_min").value)
        i_v_max = float(self.get_parameter("i_v_max").value)
        self.pid_v = PID(kp_v, ki_v, kd_v, i_min=i_v_min, i_max=i_v_max)

        self.cmd_timeout = float(self.get_parameter("cmd_timeout").value)
        self.steer_rate = float(self.get_parameter("steer_rate_limit").value)
        self.thr_rate = float(self.get_parameter("throttle_rate_limit").value)
        self.brk_rate = float(self.get_parameter("brake_rate_limit").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)

        # ----- State -----
        self.v_meas = 0.0
        self.v_ref = 0.0
        self.w_ref = 0.0
        self.v_ref_f = 0.0
        self.w_ref_f = 0.0
        self.last_cmd_time = None

        self.prev_time = self.get_clock().now()
        self.steer_prev = 0.0
        self.throttle_prev = 0.0
        self.brake_prev = 0.0

        # ----- ROS interfaces -----
        self.sub_cmd = self.create_subscription(Twist, self.cmd_topic, self.on_cmd, 10)
        self.sub_odom = self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.pub_ctrl = self.create_publisher(CarlaEgoVehicleControl, self.control_topic, 10)

        # Control loop timer (50 Hz)
        self.timer = self.create_timer(0.02, self.on_timer)

        self.get_logger().info(f"Listening cmd: {self.cmd_topic}")
        self.get_logger().info(f"Listening odom: {self.odom_topic}")
        self.get_logger().info(f"Publishing control: {self.control_topic}")

    def on_cmd(self, msg: Twist):
        self.v_ref = float(msg.linear.x)
        self.w_ref = float(msg.angular.z)
        self.last_cmd_time = self.get_clock().now()

        # EMA filter
        a = self.ema_alpha
        self.v_ref_f = a * self.v_ref + (1.0 - a) * self.v_ref_f
        self.w_ref_f = a * self.w_ref + (1.0 - a) * self.w_ref_f

    def on_odom(self, msg: Odometry):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.v_meas = float(math.sqrt(vx * vx + vy * vy))

    def rate_limit(self, target, prev, rate, dt):
        # limit |target-prev| <= rate*dt
        if dt <= 1e-6:
            return prev
        max_step = rate * dt
        delta = clamp(target - prev, -max_step, max_step)
        return prev + delta

    def on_timer(self):
        now = self.get_clock().now()
        dt = (now - self.prev_time).nanoseconds * 1e-9
        self.prev_time = now

        # Safety: if cmd is stale, reduce speed reference and brake gently
        cmd_stale = True
        if self.last_cmd_time is not None:
            age = (now - self.last_cmd_time).nanoseconds * 1e-9
            cmd_stale = age > self.cmd_timeout

        v_ref = self.v_ref_f
        w_ref = self.w_ref_f

        if cmd_stale:
            v_ref = 0.0
            w_ref = 0.0

        # ----- Lateral: (v, w) -> steer -----
        # curvature kappa = w/v; delta = atan(L*kappa)
        v_for_curv = max(self.v_meas, self.v_min)
        kappa = w_ref / v_for_curv
        delta = math.atan(self.L * kappa)  # [rad]
        steer = clamp(delta / self.delta_max, -1.0, 1.0)

        # Rate limit steering
        steer = self.rate_limit(steer, self.steer_prev, self.steer_rate, dt)
        self.steer_prev = steer

        # ----- Longitudinal: PID on speed -> accel_cmd -----
        # For CARLA, we convert accel_cmd to throttle/brake
        e_v = v_ref - self.v_meas
        accel_cmd = self.pid_v.step(e_v, dt)

        # Simple mapping: accel_cmd > 0 => throttle, else brake
        throttle = 0.0
        brake = 0.0

        # Aceleración deseada -> [0..1] (calibra el gain)
        # Gain “a_to_throttle” implícito: aquí lo tratamos 1:1 y saturamos.
        if accel_cmd >= 0.0:
            throttle = clamp(accel_cmd, 0.0, self.max_throttle)
            brake = 0.0
        else:
            brake = clamp(-accel_cmd, 0.0, self.max_brake)
            throttle = 0.0

        # Si cmd está stale, frena suave (por seguridad)
        if cmd_stale:
            throttle = 0.0
            brake = max(brake, 0.2)

        # Rate limit throttle/brake
        throttle = self.rate_limit(throttle, self.throttle_prev, self.thr_rate, dt)
        brake = self.rate_limit(brake, self.brake_prev, self.brk_rate, dt)
        self.throttle_prev = throttle
        self.brake_prev = brake

        # ----- Publish CARLA control -----
        ctrl = CarlaEgoVehicleControl()
        ctrl.throttle = float(throttle)
        ctrl.steer = float(steer)
        ctrl.brake = float(brake)
        ctrl.hand_brake = False
        ctrl.reverse = False
        ctrl.manual_gear_shift = False

        self.pub_ctrl.publish(ctrl)


def main():
    rclpy.init()
    node = TwistPIDController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
