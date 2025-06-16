import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/daniel/code/2025-phd-daniel-guerrero/robot_ws/install/carla_manual_control'
