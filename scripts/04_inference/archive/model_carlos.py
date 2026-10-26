import os
import argparse
from threading import Lock
import time
import random
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus
from std_msgs.msg import Bool, Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
# ----- CARGA DE MODELOS EN PYTORCH -----
from utils.ModifiedDeepestLSTM import ModifiedDeepestLSTM
monolithic_model_path = "/home/canveo/Projects/deepestLSTM/bubble_monolitico_burbuja_deepest_2.pth"
monolithic_model = ModifiedDeepestLSTM(image_shape=(66, 200, 3), num_labels=2)
monolithic_model.load_state_dict(torch.load(monolithic_model_path, map_location=torch.device('cpu')))
monolithic_model.eval()
def process_image_rgb(image_seg):
    calzada_color = [128, 64, 128]
    mask = cv2.inRange(image_seg, np.array(calzada_color), np.array(calzada_color))
    image_seg_masked = np.zeros_like(image_seg)
    image_seg_masked[mask > 0] = [255, 255, 255]
    image_seg_rgb = cv2.resize(image_seg_masked[200:-1, :], (200, 66))
    image_seg_rgb = cv2.cvtColor(image_seg_rgb, cv2.COLOR_BGR2GRAY)
    image_seg_rgb = cv2.merge([image_seg_rgb, image_seg_rgb, image_seg_rgb])
    return image_seg_rgb
def predict_controls_pytorch(model, image_seg):
    image_seg_processed = process_image_rgb(image_seg)
    input_tensor = np.expand_dims(image_seg_processed, axis=0)
    input_tensor = torch.from_numpy(input_tensor).float()
    input_tensor = input_tensor.permute(0, 3, 1, 2).to(torch.device('cpu'))
    with torch.no_grad():
        prediction = model(input_tensor)
    steer = prediction[0].item()
    throttle = prediction[1].item()
    return steer, throttle
class DummyControl(Node):
    def __init__(self, mode, monolithic):
        super().__init__("carla_dummy")
        self.bridge = CvBridge()
        self.camera_image = None
        self.segmented_image = None
        self.lock = Lock()
        self.mode = mode
        self.monolithic = monolithic
        cv2.namedWindow("Road Control")
        cv2.resizeWindow("Road Control", 1024, 512)
        self.publisher_control = self.create_publisher(
            CarlaEgoVehicleControl,
            "/carla/ego_vehicle/vehicle_control_cmd_manual",
            10,
        )
        self.publisher_control_manual_override = self.create_publisher(
            Bool,
            "/carla/ego_vehicle/vehicle_control_manual_override",
            qos_profile=rclpy.qos.qos_profile_system_default,
        )
        self.publisher_autopilot = self.create_publisher(
            Bool, "/carla/ego_vehicle/enable_autopilot", 10
        )
        self.status_subscriber = self.create_subscription(
            CarlaEgoVehicleStatus,
            "/carla/ego_vehicle/vehicle_status",
            self.status_callback_speed,
            10,
        )
        self.image_subscriber = self.create_subscription(
            Image, "/carla_custom/rgb_front/image", self.vehicle_image_callback, 10
        )
        self.segmentation_image_subscriber = self.create_subscription(
            Image,
            "/carla_custom/semantic_segmentation_front/image",
            self.segmentation_image_callback,
            10,
        )
        self.speed = 0.0
        self.control_msg = CarlaEgoVehicleControl()
        self.reset_control_msg()
        self.timer = self.create_timer(1.0 / 40, self.control_vehicle)
        self.manual_mode = False
        self.manual_timer_end = 0
        self.next_manual_mode_time = time.monotonic() + 5
        self.previous_throttle = 0.0
        self.values = deque(maxlen=10)
    def filter(self, new_value):
        self.values.append(new_value)
        return sum(self.values) / len(self.values)
    def set_control_manual_override(self):
        self.publisher_control_manual_override.publish(Bool(data=True))
    def set_autopilot(self):
        self.publisher_autopilot.publish(Bool(data=False))
    def status_callback_speed(self, msg):
        self.speed = msg.velocity * 3.6
    def reset_control_msg(self):
        self.control_msg = CarlaEgoVehicleControl(
            header=Header(stamp=self.get_clock().now().to_msg()),
            throttle=0.0,
            steer=0.0,
            brake=0.0,
            hand_brake=False,
            reverse=False,
            gear=1,
            manual_gear_shift=False,
        )
    def vehicle_image_callback(self, image):
        self.camera_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        self.update_display()
    def segmentation_image_callback(self, image):
        self.segmented_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
    def predict_controls(self):
        return predict_controls_pytorch(monolithic_model, self.segmented_image)
    def control_vehicle(self):
        self.set_autopilot()
        self.set_control_manual_override()
        if self.mode == "dagger":
            if time.monotonic() >= self.next_manual_mode_time:
                self.manual_mode = True
                self.manual_timer_end = time.monotonic() + 1.0
                self.next_manual_mode_time = self.manual_timer_end + 5
            if self.manual_mode and time.monotonic() > self.manual_timer_end:
                self.manual_mode = False
            if self.manual_mode:
                self.control_msg.throttle = 0.3
                self.control_msg.steer = random.choice([-0.5, 0.5])
                self.control_msg.brake = 0.0
            else:
                if self.segmented_image is not None:
                    predicted_steer, predicted_throttle = self.predict_controls()
                    self.control_msg.steer = float(predicted_steer)
                    self.control_msg.throttle = predicted_throttle
                    self.control_msg.brake = 0.0
        else:  # modo normal
            if self.segmented_image is not None:
                start_time = time.time()
                steer, throttle = self.predict_controls()
                denormalized_steer = np.interp(steer, (0, 1), (-1, 1))
                smoothed_throttle = 0.6 * throttle + 0.4 * self.previous_throttle
                self.get_logger().info(f"Inference time: {time.time() - start_time:.4f} seconds")
                self.get_logger().info(f"Predicted - Steer: {denormalized_steer}, Throttle: {smoothed_throttle}")
                self.control_msg.throttle = smoothed_throttle
                self.control_msg.brake = 0.0
                self.control_msg.steer = float(denormalized_steer)
                self.previous_throttle = smoothed_throttle
        self.control_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_control.publish(self.control_msg)
    def update_display(self):
        if self.camera_image is not None:
            img_rgb = self.camera_image.copy()
            x1, y1 = 35, 30
            x2, y2 = 200, 130
            alpha = 0.5
            overlay = img_rgb.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0, img_rgb)
            cv2.putText(img_rgb, f"Speed: {self.speed:.2f} km/h", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_rgb, f"Steer: {self.control_msg.steer:.2f}", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_rgb, f"Throttle: {self.control_msg.throttle:.2f}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_rgb, f"Brake: {self.control_msg.brake:.2f}", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(img_rgb, f"Modo: {'DAGGER' if self.mode == 'dagger' else 'Normal'}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Road Control", cv2.resize(img_rgb, (1024, 512)))
            cv2.waitKey(1)
def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['dagger', 'normal'], default='normal', help='Modo de funcionamiento')
    parser.add_argument('--monolithic', action='store_true', help='(Obsoleto) - Siempre monolítico')
    parsed_args = parser.parse_args()
    rclpy.init(args=args)
    node = DummyControl(mode=parsed_args.mode, monolithic=True)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == "__main__":
    main()