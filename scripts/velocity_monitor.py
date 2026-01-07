import carla
import time
import math

HOST = "127.0.0.1"
PORT = 2000
HZ = 20  # igual que tu CAM_FPS, pero puede ser 5/10/20

def find_ego(world, timeout_s=30.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for v in world.get_actors().filter("vehicle.*"):
            if v.attributes.get("role_name", "") == "ego":
                return v
        time.sleep(0.2)
    return None

def main():
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)
    world = client.get_world()

    ego = find_ego(world)
    if ego is None:
        print("[ERROR] No encontré vehicle con role_name='ego'.")
        print("Corre primero tu script control_carla_simple_axes.py")
        return

    print(f"[OK] Ego encontrado: id={ego.id}")
    print("[INFO] Monitor en tiempo real (Ctrl+C para salir)\n")

    dt = 1.0 / HZ

    try:
        while True:
            tf = ego.get_transform()
            vel = ego.get_velocity()
            ang = ego.get_angular_velocity()
            ctl = ego.get_control()

            # Velocidad total (magnitud)
            speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

            # Velocidad longitudinal (hacia adelante del coche)
            fwd = tf.get_forward_vector()
            v_forward = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z

            print(
                f"speed={speed_mps:6.2f} m/s ({speed_mps*3.6:6.1f} km/h) | "
                f"v_fwd={v_forward:6.2f} m/s | "
                f"yaw_rate={ang.z:7.3f} rad/s | "
                f"thr={ctl.throttle:4.2f} brk={ctl.brake:4.2f} str={ctl.steer:5.2f} rev={int(ctl.reverse)}"
            )

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[INFO] Monitor detenido.")

if __name__ == "__main__":
    main()
