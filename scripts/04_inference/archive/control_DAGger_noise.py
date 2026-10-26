# control_carla_dagger_continuous_noise_csv.py
"""
CARLA manual driving + DAgger-style continuous action-noise dataset collection (CSV).

Core idea (DAgger-style, "continuous noise"):
- You drive the vehicle manually (steering wheel/pedals if available, otherwise keyboard).
- The dataset label is ALWAYS the human "expert" command (clean, noise-free intent).
- The control applied to the vehicle is expert + small Gaussian noise each tick (optional).
  This forces the human to constantly correct small disturbances, generating robust data.

Output:
- Images saved to: dagger_runs/<run_id>/rgb/<step>.jpg
- Labels saved to: dagger_runs/<run_id>/labels.csv
  Each row includes expert controls, applied controls, and noise configuration.
"""

import os
import time
import csv
from datetime import datetime
from queue import Queue, Empty

import cv2
import numpy as np
import pygame
import carla

# -------------------- Connection / Simulation Parameters --------------------
HOST = "127.0.0.1"
PORT = 2000

# Camera resolution and synchronous simulation tick rate
IMG_W, IMG_H = 1280, 920
CAM_FPS = 20

# Keyboard driving parameters (how fast steering changes, throttle/brake increments)
STEER_RATE = 2.5
THROTTLE_STEP = 0.02
BRAKE_STEP = 0.05


# -------------------- Steering Wheel / Pedal Mapping --------------------
# Axis indices depend on your specific wheel/pedal device.
AXIS_STEER = 0
AXIS_THROTTLE = 1
AXIS_BRAKE = 2

# Deadzones to avoid jitter near the neutral position
DEADZONE_STEER = 0.02
DEADZONE_PEDAL = 0.01


# -------------------- DAgger Continuous Noise Parameters --------------------
# "Continuous noise" means: on every tick, applied_action = expert_action + N(0, std).
# The label remains the expert action (clean).
NOISE_STD_STEER = 0.05         # typical 0.02–0.10
NOISE_STD_THROTTLE = 0.01      # optional small noise
NOISE_STD_BRAKE = 0.00         # usually keep 0 (brake noise can be harsh)
NOISE_ENABLED_DEFAULT = True   # start with noise enabled

SAVE_JPEG_QUALITY = 90
# ------------------------------------------------------------------------


def to_bgr(image: carla.Image) -> np.ndarray:
    """
    Convert a CARLA BGRA image buffer into a BGR numpy array for OpenCV.

    CARLA provides raw_data as BGRA (4 channels). OpenCV commonly uses BGR (3 channels),
    so we drop the alpha channel.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]  # BGR


def pedal_inverted_to_01(v: float, deadzone: float = 0.0) -> float:
    """
    Convert an inverted pedal axis from [1..-1] to [0..1].

    Many wheels/pedals report:
      1.0  -> not pressed
     -1.0  -> fully pressed

    This maps:
      1.0  -> 0.0
     -1.0  -> 1.0

    deadzone:
      If v is very close to 1.0 (unpressed), snap to 1.0 to remove tiny noise.
    """
    v = float(np.clip(v, -1.0, 1.0))
    if v > 1.0 - deadzone:
        v = 1.0
    return (1.0 - v) / 2.0


def axis_with_deadzone(v: float, dz: float) -> float:
    """
    Apply a symmetric deadzone to an axis value.

    If abs(v) < dz, return 0.0, otherwise return v.
    """
    v = float(np.clip(v, -1.0, 1.0))
    return 0.0 if abs(v) < dz else v


def save_image(out_dir: str, step: int, frame_bgr: np.ndarray) -> str:
    """
    Save one camera frame to disk as a JPEG.

    Returns:
      Relative image path (e.g., "rgb/000123.jpg") to store in the CSV label file.
    """
    img_rel = f"rgb/{step:06d}.jpg"
    cv2.imwrite(
        os.path.join(out_dir, img_rel),
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(SAVE_JPEG_QUALITY)],
    )
    return img_rel


def main():
    # -------------------- Connect to CARLA --------------------
    client = carla.Client(HOST, PORT)
    client.set_timeout(5.0)

    # Load desired map if needed
    DESIRED_MAP = "Town01"
    current_map = client.get_world().get_map().name
    if DESIRED_MAP not in current_map:
        print(f"[INFO] Loading map {DESIRED_MAP}...")
        world = client.load_world(DESIRED_MAP)
    else:
        print(f"[INFO] Already in {DESIRED_MAP}")
        world = client.get_world()

    # Save original settings so we can restore them on exit
    original_settings = world.get_settings()

    vehicle = None
    camera = None
    image_queue = Queue()

    # -------------------- Initialize pygame input --------------------
    pygame.init()
    pygame.joystick.init()

    joystick = None
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(
            f"[INFO] Joystick: {joystick.get_name()} | "
            f"Axes: {joystick.get_numaxes()} | Buttons: {joystick.get_numbuttons()}"
        )
    else:
        print("[INFO] No joystick/wheel found. Keyboard controls: W/S/A/D, SPACE (brake), R (reverse), Q (quit)")

    # -------------------- Output directory and CSV setup --------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("dagger_runs", run_id)
    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)

    log_path = os.path.join(out_dir, "labels.csv")
    print(f"[INFO] Saving dataset (continuous noise) to: {out_dir}")

    # Runtime state
    step = 0
    record_enabled = True
    noise_enabled = bool(NOISE_ENABLED_DEFAULT)

    # Write CSV header
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "step",
            "image",
            "expert_steer", "expert_throttle", "expert_brake", "expert_reverse",
            "applied_steer", "applied_throttle", "applied_brake", "applied_reverse",
            "mode",
            "noise_enabled",
            "noise_std_steer",
            "noise_std_throttle",
            "noise_std_brake",
        ])

    try:
        # -------------------- Enable synchronous mode --------------------
        # We run the simulator with fixed time steps for consistent data capture.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / CAM_FPS
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()

        # -------------------- Spawn vehicle --------------------
        vehicle_bp = bp_lib.find("vehicle.tesla.model3")
        if vehicle_bp.has_attribute("role_name"):
            vehicle_bp.set_attribute("role_name", "ego")

        spawn_points = world.get_map().get_spawn_points() or [carla.Transform()]
        for sp in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle:
                break
        if vehicle is None:
            raise RuntimeError("Could not spawn the vehicle (no free spawn points).")

        # -------------------- Attach RGB camera sensor --------------------
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", str(1.0 / CAM_FPS))

        cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(image_queue.put)

        cv2.namedWindow("CAM", cv2.WINDOW_AUTOSIZE)

        # -------------------- Control variables --------------------
        steer = 0.0
        throttle = 0.0
        brake = 0.0
        reverse = False
        hand_brake = False

        control = carla.VehicleControl()

        # Prime the world once
        world.tick()

        print("[INFO] Keys: Q quit | R reverse | G toggle recording | N toggle noise")

        running = True
        while running:
            # --------------- Advance simulation ---------------
            world.tick()

            # --------------- Get latest camera frame ---------------
            img = None
            while True:
                try:
                    img = image_queue.get_nowait()
                except Empty:
                    break

            frame = None
            if img is not None:
                frame = to_bgr(img)

                # Debug overlay: shows current recording/noise state
                overlay = frame.copy()
                status = [
                    f"record={record_enabled}",
                    f"noise={noise_enabled}",
                    f"std_steer={NOISE_STD_STEER}",
                ]
                cv2.putText(
                    overlay,
                    " | ".join(status),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                cv2.imshow("CAM", overlay)
                cv2.waitKey(1)

            # --------------- Pygame events ---------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            # Toggle recording (simple debounce via sleep)
            if keys[pygame.K_g]:
                record_enabled = not record_enabled
                print(f"[INFO] record_enabled -> {record_enabled}")
                time.sleep(0.2)

            # Toggle noise injection (simple debounce via sleep)
            if keys[pygame.K_n]:
                noise_enabled = not noise_enabled
                print(f"[INFO] noise_enabled -> {noise_enabled}")
                time.sleep(0.2)

            # Quit
            if keys[pygame.K_q]:
                running = False

            # -------------------- Read human "expert" input --------------------
            if joystick is None:
                # ---- Keyboard steering ----
                if keys[pygame.K_a]:
                    steer -= STEER_RATE * settings.fixed_delta_seconds
                elif keys[pygame.K_d]:
                    steer += STEER_RATE * settings.fixed_delta_seconds
                else:
                    steer *= 0.9

                # ---- Keyboard throttle / brake ----
                if keys[pygame.K_w]:
                    throttle = min(1.0, throttle + THROTTLE_STEP)
                    brake = 0.0
                elif keys[pygame.K_s]:
                    brake = min(1.0, brake + BRAKE_STEP)
                    throttle = max(0.0, throttle - THROTTLE_STEP)
                else:
                    throttle *= 0.98
                    brake *= 0.9

                if keys[pygame.K_SPACE]:
                    brake = 1.0

                # Reverse toggle: hold R to set reverse; release when stopped
                if keys[pygame.K_r]:
                    reverse = True
                if not keys[pygame.K_r] and reverse and throttle == 0.0:
                    reverse = False

            else:
                # ---- Wheel steering axis ----
                try:
                    steer_val = joystick.get_axis(AXIS_STEER)
                except Exception:
                    steer_val = 0.0
                steer = axis_with_deadzone(steer_val, DEADZONE_STEER)

                # ---- Throttle pedal ----
                try:
                    accel_val = joystick.get_axis(AXIS_THROTTLE)
                except Exception:
                    accel_val = 1.0
                throttle = float(np.clip(pedal_inverted_to_01(accel_val, DEADZONE_PEDAL), 0.0, 1.0))

                # ---- Brake pedal ----
                try:
                    brake_val = joystick.get_axis(AXIS_BRAKE)
                except Exception:
                    brake_val = 1.0
                brake = float(np.clip(pedal_inverted_to_01(brake_val, DEADZONE_PEDAL), 0.0, 1.0))

                # Reverse with keyboard R
                if keys[pygame.K_r]:
                    reverse = True
                if not keys[pygame.K_r] and reverse and throttle == 0.0:
                    reverse = False

            # Clamp expert commands to valid ranges
            steer = float(np.clip(steer, -1.0, 1.0))
            throttle = float(np.clip(throttle, 0.0, 1.0)) / 4.0  # keep your original scaling
            brake = float(np.clip(brake, 0.0, 1.0))

            # Expert label (clean intent)
            expert = {
                "steer": steer,
                "throttle": throttle,
                "brake": brake,
                "reverse": bool(reverse),
            }

            # ------------------------------------------------------------------
            # Continuous noise injection:
            # - Applied action is expert + Gaussian noise (if noise_enabled).
            # - Labels remain the expert action.
            # ------------------------------------------------------------------
            applied = dict(expert)

            if noise_enabled:
                applied["steer"] = float(np.clip(
                    applied["steer"] + np.random.normal(0.0, NOISE_STD_STEER),
                    -1.0, 1.0
                ))
                applied["throttle"] = float(np.clip(
                    applied["throttle"] + np.random.normal(0.0, NOISE_STD_THROTTLE),
                    0.0, 1.0
                ))
                applied["brake"] = float(np.clip(
                    applied["brake"] + np.random.normal(0.0, NOISE_STD_BRAKE),
                    0.0, 1.0
                ))

            # -------------------- Apply control to CARLA --------------------
            control.steer = float(np.clip(applied["steer"], -1.0, 1.0))
            control.throttle = float(np.clip(applied["throttle"], 0.0, 1.0))
            control.brake = float(np.clip(applied["brake"], 0.0, 1.0))
            control.hand_brake = hand_brake
            control.reverse = bool(applied["reverse"])
            vehicle.apply_control(control)

            # -------------------- Save sample (image + CSV row) --------------------
            # Note: label == expert, applied == what was actually executed.
            if record_enabled and (frame is not None):
                img_rel = save_image(out_dir, step, frame)

                mode = "dagger_continuous_noise"
                with open(log_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        step,
                        img_rel,

                        expert["steer"],
                        expert["throttle"],
                        expert["brake"],
                        int(expert["reverse"]),

                        applied["steer"],
                        applied["throttle"],
                        applied["brake"],
                        int(applied["reverse"]),

                        mode,
                        int(noise_enabled),
                        NOISE_STD_STEER,
                        NOISE_STD_THROTTLE,
                        NOISE_STD_BRAKE,
                    ])

                step += 1

        cv2.destroyAllWindows()

    finally:
        # -------------------- Cleanup / restore settings --------------------
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        try:
            if camera is not None:
                camera.destroy()
        except Exception:
            pass

        try:
            if vehicle is not None:
                vehicle.destroy()
        except Exception:
            pass

        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        pygame.quit()


if __name__ == "__main__":
    main()
