import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/daniel/code/2025-phd-daniel-guerrero/ros-bridge/install/carla_waypoint_publisher'
