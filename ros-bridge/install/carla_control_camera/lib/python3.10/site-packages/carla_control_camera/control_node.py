import rclpy
from rclpy.node import Node
from carla_msgs.msg import CarlaEgoVehicleControl

class ControlNode(Node):
    def __init__(self):
        super().__init__('carla_control_node')
        self.publisher_ = self.create_publisher(CarlaEgoVehicleControl, '/carla/hero/vehicle_control_cmd', 10)
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('Control node initialized')

    def timer_callback(self):
        msg = CarlaEgoVehicleControl()
        msg.throttle = 0.5
        msg.steer = 0.0
        msg.brake = 0.0
        msg.hand_brake = False
        msg.reverse = False
        self.publisher_.publish(msg)
        self.get_logger().info('Published control command')

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
