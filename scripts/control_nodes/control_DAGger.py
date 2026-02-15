# control_carla_dagger_recovery_only.py
"""
CARLA manual driving + DAgger-style "recovery-only" dataset collection.

Idea:
- You drive with steering wheel/pedals (or keyboard).
- Every NOISE_INTERVAL_SEC seconds, the script injects a short steering "push" (noise)
  for NOISE_TICKS simulation ticks. Those noisy ticks are NOT recorded.
- Immediately after the noise ends, the script records RECOVERY_SEC seconds of frames
  while you recover the vehicle back to normal driving. Only these recovery frames
  are saved to disk, producing a dataset focused on corrective actions.

Output format:
- Images are saved to: dagger_runs/<run_id>/rgb/<step>.jpg
- Labels are appended to: dagger_runs/<run_id>/labels.jsonl
  Each JSONL row contains:
    - step: integer index
    - image: relative path to jpg
    - expert: the human's raw control intent (steer/throttle/brake/reverse)
    - applied: the control actually applied to the vehicle (expert + noise if active)
    - meta: metadata about the run and noise/recovery state
"""

import os
import json
import time
from datetime import datetime
from queue import Queue, Empty

import cv2
import numpy as np
import pygame
import carla


# -------------------- Connection / Simulation Parameters --------------------
HOST = "127.0.0.1"
PORT = 2000

# Camera resolution and simulation FPS (also used as synchronous tick rate)
IMG_W, IMG_H = 1280, 920
CAM_FPS = 20

# Keyboard steering smoothing and pedal steps
STEER_RATE = 2.5
THROTTLE_STEP = 0.02
BRAKE_STEP = 0.05


# -------------------- Steering Wheel / Pedal Mapping --------------------
# These axis indices depend on your device. Typical Logitech wheels use:
# steer = 0, throttle = 1, brake = 2 (but confirm with pygame joystick info)
AXIS_STEER = 0
AXIS_THROTTLE = 1
AXIS_BRAKE = 2

# Small deadzones to avoid sensor jitter
DEADZONE_STEER = 0.02
DEADZONE_PEDAL = 0.01


# -------------------- DAgger "Recovery-only" Logging + Noise Injection --------------------
# Every NOISE_INTERVAL_SEC seconds (sim time), inject a steering disturbance.
NOISE_INTERVAL_SEC = 20.0       # how often to trigger noise (in simulation seconds)
NOISE_TICKS = 20                # how many simulation ticks noise lasts (NOT recorded)
STEER_NOISE_MAG = 0.25          # magnitude of steer perturbation (typical 0.15-0.35)

# After noise ends, record recovery actions for this duration
RECOVERY_SEC = 3.0              # how long to record after noise (recovery only)

# JPEG output quality
SAVE_JPEG_QUALITY = 90
# ------------------------------------------------------------------------


def to_bgr(image: carla.Image) -> np.ndarray:
    """
    Convert a CARLA raw BGRA image buffer into a BGR numpy array suitable for OpenCV.

    CARLA Image raw_data is BGRA (4 channels). We keep only the first 3 channels.
    """
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(image.height, image.width, 4)
    return arr[:, :, :3]  # BGR (OpenCV default)


def pedal_inverted_to_01(v: float, deadzone: float = 0.0) -> float:
    """
    Convert a typical pedal axis value from [1..-1] to [0..1].

    Many pedal axes return:
      1.0  -> not pressed
     -1.0  -> fully pressed

    We map:
      1.0  -> 0.0
     -1.0  -> 1.0

    deadzone:
      If v is near 1.0 (unpressed), snap it to 1.0 to reduce jitter.
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


def save_sample(out_dir: str, step: int, frame_bgr: np.ndarray, expert: dict, applied: dict, meta: dict) -> dict:
    """
    Save one dataset sample:
    - Write the current RGB image as JPG
    - Return a JSON-serializable label row (to be written to JSONL)

    out_dir:
      Run output directory (dagger_runs/<run_id>)
    step:
      Sample index used to name the file
    frame_bgr:
      OpenCV BGR image
    expert:
      Control command from the human (before noise)
    applied:
      Control command applied to CARLA (after noise injection, if any)
    meta:
      Extra metadata for reproducibility and debugging
    """
    img_rel = f"rgb/{step:06d}.jpg"
    cv2.imwrite(
        os.path.join(out_dir, img_rel),
        frame_bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(SAVE_JPEG_QUALITY)],
    )

    row = {
        "step": step,
        "image": img_rel,
        "expert": expert,
        "applied": applied,
        "meta": meta,
    }
    return row


def main():
    # -------------------- Connect to CARLA --------------------
    client = carla.Client(HOST, PORT)
    client.set_timeout(5.0)

    # Load a desired map if not already loaded
    DESIRED_MAP = "Town01"
    current_map = client.get_world().get_map().name
    if DESIRED_MAP not in current_map:
        print(f"[INFO] Loading map {DESIRED_MAP}...")
        world = client.load_world(DESIRED_MAP)
    else:
        print(f"[INFO] Already in {DESIRED_MAP}")
        world = client.get_world()

    # Save current world settings to restore later
    original_settings = world.get_settings()

    vehicle = None
    camera = None
    image_queue = Queue()

    # -------------------- Initialize Pygame Input --------------------
    pygame.init()
    pygame.joystick.init()
    joystick = None

    # If a joystick exists, use it; otherwise fallback to keyboard driving
    if pygame.joystick.get_count() > 0:
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        print(
            f"[INFO] Joystick: {joystick.get_name()} | "
            f"Axes: {joystick.get_numaxes()} | Buttons: {joystick.get_numbuttons()}"
        )
    else:
        print("[INFO] No joystick/wheel found. Keyboard controls: W/S/A/D, SPACE (brake), R (reverse), Q (quit)")

    # -------------------- Output Folder --------------------
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("dagger_runs", run_id)
    os.makedirs(os.path.join(out_dir, "rgb"), exist_ok=True)
    log_path = os.path.join(out_dir, "labels.jsonl")
    print(f"[INFO] Saving recovery-only dataset to: {out_dir}")

    # -------------------- DAgger State --------------------
    step = 0
    record_enabled = True

    # Noise and recovery tick counters
    noise_ticks_left = 0
    recovery_ticks_left = 0

    # Simulation time accumulator since the last noise trigger
    sim_since_last_noise = 0.0

    def trigger_noise_and_recovery():
        """
        Start a noise episode followed by a recovery recording window.

        - noise_ticks_left: how many ticks we will inject steering noise
        - recovery_ticks_left: how many ticks we record AFTER noise ends
        """
        nonlocal noise_ticks_left, recovery_ticks_left, sim_since_last_noise
        noise_ticks_left = int(NOISE_TICKS)
        recovery_ticks_left = int(round(RECOVERY_SEC * CAM_FPS))
        sim_since_last_noise = 0.0
        print(f"[NOISE] Trigger -> NOISE_TICKS={noise_ticks_left}, RECOVERY_TICKS={recovery_ticks_left}")

    try:
        # -------------------- Enable Synchronous Mode --------------------
        # We run the simulation in fixed time steps. Each loop calls world.tick().
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / CAM_FPS
        settings.no_rendering_mode = False
        world.apply_settings(settings)

        bp_lib = world.get_blueprint_library()

        # -------------------- Spawn Vehicle --------------------
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

        # -------------------- Attach RGB Camera Sensor --------------------
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        cam_bp.set_attribute("sensor_tick", str(1.0 / CAM_FPS))

        # Camera transform relative to vehicle: slightly forward and above hood
        cam_tf = carla.Transform(carla.Location(x=0.8, z=1.3))
        camera = world.spawn_actor(cam_bp, cam_tf, attach_to=vehicle)
        camera.listen(image_queue.put)  # push frames to a thread-safe queue

        # OpenCV window for visualization
        cv2.namedWindow("CAM", cv2.WINDOW_AUTOSIZE)

        # -------------------- Control Variables --------------------
        steer = 0.0
        throttle = 0.0
        brake = 0.0
        reverse = False
        hand_brake = False

        # CARLA control object reused each frame
        control = carla.VehicleControl()

        # Prime the simulation once before entering loop
        world.tick()

        print("[INFO] Keys: Q quit | R reverse | G toggle recording | (Auto-noise every 20s)")
        running = True

        while running:
            # --------------- Advance simulation by one fixed tick ---------------
            world.tick()

            # Update simulated time since last noise
            sim_since_last_noise += float(settings.fixed_delta_seconds)

            # --------------- Get the newest camera image ---------------
            img = None
            while True:
                try:
                    img = image_queue.get_nowait()
                except Empty:
                    break

            frame = None
            if img is not None:
                frame = to_bgr(img)

                # Overlay the current state for debugging
                overlay = frame.copy()
                status = [
                    f"record={record_enabled}",
                    f"noise_ticks_left={noise_ticks_left}",
                    f"recovery_ticks_left={recovery_ticks_left}",
                    f"t_since_noise={sim_since_last_noise:0.1f}s",
                ]
                cv2.putText(
                    overlay,
                    " | ".join(status),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.imshow("CAM", overlay)
                cv2.waitKey(1)

            # --------------- Handle pygame events ---------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            keys = pygame.key.get_pressed()

            # Toggle recording with G (simple debounce via sleep)
            if keys[pygame.K_g]:
                record_enabled = not record_enabled
                print(f"[INFO] record_enabled -> {record_enabled}")
                time.sleep(0.2)

            # Quit with Q
            if keys[pygame.K_q]:
                running = False

            # -------------------- Read Human "Expert" Input --------------------
            if joystick is None:
                # ---- Keyboard steering ----
                if keys[pygame.K_a]:
                    steer -= STEER_RATE * settings.fixed_delta_seconds
                elif keys[pygame.K_d]:
                    steer += STEER_RATE * settings.fixed_delta_seconds
                else:
                    # steering decay toward zero when no key pressed
                    steer *= 0.9

                # ---- Keyboard throttle/brake ----
                if keys[pygame.K_w]:
                    throttle = min(1.0, throttle + THROTTLE_STEP)
                    brake = 0.0
                elif keys[pygame.K_s]:
                    brake = min(1.0, brake + BRAKE_STEP)
                    throttle = max(0.0, throttle - THROTTLE_STEP)
                else:
                    throttle *= 0.98
                    brake *= 0.9

                # Full brake
                if keys[pygame.K_SPACE]:
                    brake = 1.0

                # Reverse logic: hold R to enable reverse; release when stopped
                if keys[pygame.K_r]:
                    reverse = True
                if not keys[pygame.K_r] and reverse and throttle == 0.0:
                    reverse = False

            else:
                # ---- Steering wheel ----
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

                # Reverse logic (keyboard R)
                if keys[pygame.K_r]:
                    reverse = True
                if not keys[pygame.K_r] and reverse and throttle == 0.0:
                    reverse = False

            # Clamp expert input
            steer = float(np.clip(steer, -1.0, 1.0))
            throttle = float(np.clip(throttle, 0.0, 1.0)) / 4.0  # your original scaling
            brake = float(np.clip(brake, 0.0, 1.0))

            # "Expert" = what the human is doing (clean labels)
            expert = {
                "steer": steer,
                "throttle": throttle,
                "brake": brake,
                "reverse": bool(reverse),
            }

            # ------------------------------------------------------------------
            # (1) Trigger noise periodically if not already in noise/recovery
            # ------------------------------------------------------------------
            if noise_ticks_left == 0 and recovery_ticks_left == 0 and sim_since_last_noise >= NOISE_INTERVAL_SEC:
                trigger_noise_and_recovery()

            # ------------------------------------------------------------------
            # (2) Build applied control:
            #     start from expert, optionally add steering noise during NOISE.
            # ------------------------------------------------------------------
            applied = dict(expert)
            used_noise = False

            if noise_ticks_left > 0:
                # Inject a random steering push (not recorded)
                sign = 1.0 if np.random.randn() >= 0 else -1.0
                applied["steer"] = float(np.clip(applied["steer"] + sign * STEER_NOISE_MAG, -1.0, 1.0))
                used_noise = True
                noise_ticks_left -= 1

                # When noise ends, recovery_ticks_left remains active for next ticks

            elif recovery_ticks_left > 0:
                # Recovery phase (recorded)
                recovery_ticks_left -= 1

            # ------------------------------------------------------------------
            # (3) Apply control to CARLA vehicle
            # ------------------------------------------------------------------
            control.steer = float(np.clip(applied["steer"], -1.0, 1.0))
            control.throttle = float(np.clip(applied["throttle"], 0.0, 1.0))
            control.brake = float(np.clip(applied["brake"], 0.0, 1.0))
            control.hand_brake = hand_brake
            control.reverse = bool(applied["reverse"])
            vehicle.apply_control(control)

            # ------------------------------------------------------------------
            # (4) Save ONLY RECOVERY ticks (never save NOISE ticks)
            # ------------------------------------------------------------------
            in_recovery = (recovery_ticks_left > 0) and (noise_ticks_left == 0)
            in_noise = (noise_ticks_left > 0) or used_noise

            if record_enabled and (frame is not None) and in_recovery and (not in_noise):
                meta = {
                    "mode": "recovery_only",
                    "noise_interval_sec": NOISE_INTERVAL_SEC,
                    "noise_mag": STEER_NOISE_MAG,
                    "noise_ticks": NOISE_TICKS,
                    "recovery_sec": RECOVERY_SEC,
                    "sim_time_since_last_noise": sim_since_last_noise,
                }

                row = save_sample(
                    out_dir=out_dir,
                    step=step,
                    frame_bgr=frame,
                    expert=expert,
                    applied=applied,
                    meta=meta,
                )

                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

                step += 1

        cv2.destroyAllWindows()

    finally:
        # -------------------- Cleanup / Restore State --------------------
        try:
            if camera is not None:
                camera.stop()
        except Exception:
            pass

        if camera is not None:
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()

        # Restore original world settings (important if you changed sync mode)
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

        pygame.quit()


if __name__ == "__main__":
    main()
