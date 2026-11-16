#!/usr/bin/env python3
"""Generador del dataset masivo con WAYPOINTS + imágenes frontales canónicas + BEV real (cámara cenital).

Por cada frame capturado guarda:
  - Imagen frontal canónica (1920x1080, FOV 110, x=2.0 z=1.6 pitch=-15), idéntica a dataset_masivo_continuo.
  - Imagen BEV REAL de alta calidad: cámara cenital (pitch=-90) pegada al ego, "arriba = adelante del auto".
  - Pose del ego (x, y, z, yaw) derivada de la pose exacta de la cámara en ese frame.
  - Controles (steering, throttle, brake) y velocidad.
  - Waypoints futuros en el marco del ego (x = adelante, y = DERECHA, convención CARLA, steering>0 = derecha):
      * wpa{k}_x/y: k=1..N_ARC a distancias de arco fijas (ARC_STEP m)  -> independiente de la velocidad.
      * wpt{k}_x/y: k=1..N_TIME a tiempos fijos (TIME_STEP s)           -> codifica la velocidad.

Modos:
  --mode replay     Reproduce los .log del recorder (logs_dagger_expert) y captura con las cámaras.
  --mode autopilot  Conduce con el Traffic Manager (+ perturbaciones DAgger) y captura directamente.
  --mode rebuild    Recalcula los waypoints (otros horizontes) desde dataset.csv, sin CARLA.

Las discontinuidades (teletransportes DAgger / rescates) se detectan y NINGÚN waypoint cruza una:
los frames cuya ventana futura cruzaría un salto quedan con wp_*_valid = 0.
"""

import argparse
import atexit
import glob
import json
import math
import os
import queue
import random
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Configuración de sensores (calibración guardada también en meta.json)
# ------------------------------------------------------------------
FRONT = dict(w=1920, h=1080, fov=110.0, x=2.0, y=0.0, z=1.6, pitch=-15.0, yaw=0.0)
BEV = dict(size=800, fov=90.0, x=10.0, y=0.0, z=30.0, pitch=-90.0, yaw=0.0)

ARC_STEP, N_ARC = 2.0, 10      # 2 m * 10 = 20 m
TIME_STEP, N_TIME = 0.5, 8     # 0.5 s * 8 = 4 s

# Umbrales de discontinuidad (entre dos frames consecutivos)
JUMP_LATERAL_M = 0.45
JUMP_Z_M = 0.6
JUMP_DIST_M = 6.0
JUMP_DT_S = 0.6


# ------------------------------------------------------------------
# Waypoints (offline, sin CARLA)
# ------------------------------------------------------------------
def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def find_breaks(t, x, y, z, yaw):
    """brk[i]=True  <=>  hay discontinuidad entre el frame i y el i+1."""
    n = len(t)
    brk = np.zeros(n, dtype=bool)
    if n < 2:
        return brk
    dt = np.diff(t)
    dx, dy, dz = np.diff(x), np.diff(y), np.diff(z)
    ymid = yaw[:-1] + 0.5 * wrap_pi(yaw[1:] - yaw[:-1])
    lateral = np.abs(-dx * np.sin(ymid) + dy * np.cos(ymid))
    dist = np.hypot(dx, dy)
    jump = (
        (lateral > JUMP_LATERAL_M)
        | (np.abs(dz) > JUMP_Z_M)
        | (dist > JUMP_DIST_M)
        | (dt > JUMP_DT_S)
        | (dt <= 0)
    )
    brk[:-1] = jump
    return brk


def compute_waypoints(t, x, y, z, yaw, arc_step=ARC_STEP, n_arc=N_ARC,
                      time_step=TIME_STEP, n_time=N_TIME):
    """Devuelve (wa[n,n_arc,2], wt[n,n_time,2], speed_ms[n], segment_id[n], brk[n]).

    Coordenadas en el marco del ego de CADA frame: [adelante, derecha].
    NaN donde la ventana futura no existe dentro del mismo segmento continuo.
    """
    n = len(t)
    wa = np.full((n, n_arc, 2), np.nan)
    wt = np.full((n, n_time, 2), np.nan)
    speed = np.zeros(n)
    seg_id = np.zeros(n, dtype=int)
    brk = find_breaks(t, x, y, z, yaw)
    cut = np.where(brk)[0]
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [n - 1]])

    for sid, (a, b) in enumerate(zip(starts, ends)):
        seg_id[a:b + 1] = sid
        m = b - a + 1
        if m < 2:
            continue
        ts, xs, ys, yw = t[a:b + 1], x[a:b + 1], y[a:b + 1], yaw[a:b + 1]
        step = np.hypot(np.diff(xs), np.diff(ys))
        dts = np.diff(ts)
        v = step / dts
        speed[a:b + 1] = np.concatenate([[v[0]], v])
        s = np.concatenate([[0.0], np.cumsum(step)])
        s_inc = s + np.arange(m) * 1e-6
        c, sn = np.cos(yw)[:, None], np.sin(yw)[:, None]

        # --- waypoints por distancia de arco ---
        tgt = s[:, None] + arc_step * np.arange(1, n_arc + 1)[None, :]
        ok = tgt <= s[-1] + 1e-9
        fx, fy = np.interp(tgt, s_inc, xs), np.interp(tgt, s_inc, ys)
        dxw, dyw = fx - xs[:, None], fy - ys[:, None]
        fwd, right = dxw * c + dyw * sn, -dxw * sn + dyw * c
        fwd[~ok], right[~ok] = np.nan, np.nan
        wa[a:b + 1, :, 0], wa[a:b + 1, :, 1] = fwd, right

        # --- waypoints por tiempo ---
        tt = ts[:, None] + time_step * np.arange(1, n_time + 1)[None, :]
        okt = tt <= ts[-1] + 1e-9
        fx, fy = np.interp(tt, ts, xs), np.interp(tt, ts, ys)
        dxw, dyw = fx - xs[:, None], fy - ys[:, None]
        fwd, right = dxw * c + dyw * sn, -dxw * sn + dyw * c
        fwd[~okt], right[~okt] = np.nan, np.nan
        wt[a:b + 1, :, 0], wt[a:b + 1, :, 1] = fwd, right

    return wa, wt, speed, seg_id, brk


def waypoint_columns(n_arc=N_ARC, n_time=N_TIME):
    cols = []
    for k in range(1, n_arc + 1):
        cols += [f"wpa{k}_x", f"wpa{k}_y"]
    for k in range(1, n_time + 1):
        cols += [f"wpt{k}_x", f"wpt{k}_y"]
    return cols


def attach_waypoints(df, **kw):
    """df de UNA sesión ordenado por tiempo -> agrega columnas de waypoints/validez/segmento."""
    t = df["sim_t"].to_numpy(float)
    x, y, z = (df[c].to_numpy(float) for c in ("ego_x", "ego_y", "ego_z"))
    yaw = np.radians(df["ego_yaw_deg"].to_numpy(float))
    wa, wt, speed, seg_id, brk = compute_waypoints(t, x, y, z, yaw, **kw)
    n_arc, n_time = wa.shape[1], wt.shape[1]
    block = np.concatenate([wa.reshape(len(df), -1), wt.reshape(len(df), -1)], axis=1)
    wp = pd.DataFrame(block, columns=waypoint_columns(n_arc, n_time), index=df.index)
    out = df.copy()
    out["speed_kmh"] = speed * 3.6
    out["segment"] = seg_id
    out["discontinuity"] = brk.astype(int)
    out["wp_arc_valid"] = (~np.isnan(wa).any(axis=(1, 2))).astype(int)
    out["wp_time_valid"] = (~np.isnan(wt).any(axis=(1, 2))).astype(int)
    return pd.concat([out, wp], axis=1)


# ------------------------------------------------------------------
# Captura (requiere CARLA)
# ------------------------------------------------------------------
class PairBuffer:
    """Empareja imagen frontal y BEV por número de frame (ambas con el mismo sensor_tick)."""

    PAIR_TOL_S = 0.06  # las dos cámaras pueden disparar en frames vecinos (modo async)

    def __init__(self):
        self.lock = threading.Lock()
        self.unmatched = {"front": [], "bev": []}
        self.counts = {"front": 0, "bev": 0}
        self.ready = queue.Queue()

    def add(self, kind, image):
        try:
            arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape(
                image.height, image.width, 4)[:, :, :3].copy()
            other = "bev" if kind == "front" else "front"
            item = (arr, np.array(image.transform.get_matrix()), image.timestamp, image.frame)
            with self.lock:
                self.counts[kind] += 1
                best_i = None
                for i, o in enumerate(self.unmatched[other]):
                    if abs(o[2] - item[2]) <= self.PAIR_TOL_S and (
                            best_i is None
                            or abs(o[2] - item[2]) < abs(self.unmatched[other][best_i][2] - item[2])):
                        best_i = i
                if best_i is not None:
                    best = self.unmatched[other].pop(best_i)
                    d = {kind: item, other: best}
                    self.ready.put((d["front"][3], d))
                else:
                    self.unmatched[kind].append(item)
                for k in ("front", "bev"):
                    if len(self.unmatched[k]) > 30:
                        del self.unmatched[k][:-15]
        except Exception as e:
            print(f"[ERROR in buffer.add] {e}", flush=True)

    def clear(self):
        with self.lock:
            self.unmatched = {"front": [], "bev": []}
        while not self.ready.empty():
            try:
                self.ready.get_nowait()
            except queue.Empty:
                break


def _rel_matrix(carla, cfg):
    tf = carla.Transform(
        carla.Location(x=cfg["x"], y=cfg["y"], z=cfg["z"]),
        carla.Rotation(pitch=cfg["pitch"], yaw=cfg["yaw"], roll=0.0),
    )
    return np.array(tf.get_matrix())


class Rig:
    """Dos cámaras (frontal canónica + BEV cenital) pegadas al ego."""

    def __init__(self, carla, world, ego, sensor_tick):
        self.carla, self.world, self.ego = carla, world, ego
        self.buf = PairBuffer()
        lib = world.get_blueprint_library()

        def make(cfg_bp, w, h, fov, tf_cfg, kind, blur):
            bp = lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(w))
            bp.set_attribute("image_size_y", str(h))
            bp.set_attribute("fov", str(fov))
            bp.set_attribute("sensor_tick", str(sensor_tick))
            if blur is not None:
                bp.set_attribute("motion_blur_intensity", str(blur))
            tf = carla.Transform(
                carla.Location(x=tf_cfg["x"], y=tf_cfg["y"], z=tf_cfg["z"]),
                carla.Rotation(pitch=tf_cfg["pitch"], yaw=tf_cfg["yaw"], roll=0.0))
            cam = world.spawn_actor(bp, tf, attach_to=ego)
            cam.listen(lambda im, k=kind: self.buf.add(k, im))
            return cam

        self.front = make(None, FRONT["w"], FRONT["h"], FRONT["fov"], FRONT, "front", None)
        self.bev = make(None, BEV["size"], BEV["size"], BEV["fov"], BEV, "bev", 0.0)
        self.rel_front_inv = np.linalg.inv(_rel_matrix(carla, FRONT))

    def ego_pose(self, front_matrix):
        T = front_matrix @ self.rel_front_inv
        return float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), math.degrees(math.atan2(T[1, 0], T[0, 0]))

    def destroy(self):
        for cam in (self.front, self.bev):
            try:
                cam.stop()
            except Exception:
                pass
        time.sleep(0.5)
        for cam in (self.front, self.bev):
            try:
                cam.destroy()
            except Exception:
                pass


def _remove_session_rows(csv_path, session_name):
    """Elimina filas previas de `session_name` si existiera un volcado parcial."""
    if not os.path.exists(csv_path):
        return
    prefix = session_name + ","
    has_match = False
    with open(csv_path, "r") as f:
        for line in f:
            if line.startswith(prefix):
                has_match = True
                break
    if not has_match:
        return
    tmp = csv_path + ".tmp"
    with open(csv_path, "r") as src, open(tmp, "w") as dst:
        for line in src:
            if not line.startswith(prefix):
                dst.write(line)
    os.replace(tmp, csv_path)


def finalize_session_df(name, df_raw, csv_path, extra=None, wp_kw=None):
    """Calcula waypoints sobre la sesión completa ordenada por tiempo y la agrega al CSV."""
    if len(df_raw) < 3:
        return dict(session=name, rows=0)
    df = df_raw.sort_values("sim_t").drop_duplicates(subset=["sim_t"], keep="last").reset_index(drop=True)
    df = attach_waypoints(df, **(wp_kw or {}))
    _remove_session_rows(csv_path, name)
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=header, index=False, float_format="%.5f")
    info = dict(
        session=name,
        rows=len(df),
        images=int(df["has_image"].sum()),
        arc_valid=int(((df["wp_arc_valid"] == 1) & (df["has_image"] == 1)).sum()),
        time_valid=int(((df["wp_time_valid"] == 1) & (df["has_image"] == 1)).sum()),
        jumps=int(df["discontinuity"].sum()),
        sim_seconds=float(df["sim_t"].iloc[-1]),
    )
    if extra:
        info.update(extra)
    return info


class Session:
    """Acumula las filas de una sesión (o chunk de replay) y las vuelca al CSV."""

    def __init__(self, name, out_dir, args, writer_pool, start_counter=0):
        self.name, self.out_dir, self.args, self.pool = name, out_dir, args, writer_pool
        self.rows, self.counter, self.futures = [], start_counter, []
        self.front_dir = os.path.join(out_dir, "front", name)
        self.bev_dir = os.path.join(out_dir, "bev", name)
        os.makedirs(self.front_dir, exist_ok=True)
        os.makedirs(self.bev_dir, exist_ok=True)

    def _write(self, path, img, params):
        cv2.imwrite(path, img, params)

    def flush(self):
        for f in self.futures:
            f.result()
        self.futures = []

    def add(self, rig, frame, item, ego_ctrl, t0):
        front, bev = item["front"], item["bev"]
        x, y, z, yaw = rig.ego_pose(front[1])
        steer, throttle, brake = float(ego_ctrl.steer), float(ego_ctrl.throttle), float(ego_ctrl.brake)
        idle = abs(throttle) < 1e-3 and abs(steer) < 1e-3
        name = f"{self.counter:07d}"
        front_rel = os.path.join("front", self.name, name + ".jpg")
        bev_rel = os.path.join("bev", self.name, name + (".png" if self.args.bev_format == "png" else ".jpg"))
        if not idle:
            if len(self.futures) > 96:
                self.futures = [f for f in self.futures if not f.done()]
                while len(self.futures) > 96:
                    self.futures.pop(0).result()
            self.futures.append(self.pool.submit(
                self._write, os.path.join(self.out_dir, front_rel), front[0],
                [cv2.IMWRITE_JPEG_QUALITY, self.args.jpeg_quality]))
            bev_params = ([cv2.IMWRITE_PNG_COMPRESSION, 1] if self.args.bev_format == "png"
                          else [cv2.IMWRITE_JPEG_QUALITY, 97])
            self.futures.append(self.pool.submit(
                self._write, os.path.join(self.out_dir, bev_rel), bev[0], bev_params))
        self.rows.append(dict(
            session=self.name, frame_id=int(frame), sim_t=float(front[2] - t0), idx=self.counter,
            pair_dt=float(bev[2] - front[2]),
            front_path=front_rel if not idle else "", bev_path=bev_rel if not idle else "",
            has_image=int(not idle), idle=int(idle),
            ego_x=x, ego_y=y, ego_z=z, ego_yaw_deg=yaw,
            steering=steer, throttle=throttle, brake=brake,
        ))
        self.counter += 1

    def finalize(self, csv_path, extra=None):
        self.flush()
        if len(self.rows) < 3:
            return dict(session=self.name, rows=0)
        return finalize_session_df(
            self.name,
            pd.DataFrame(self.rows),
            csv_path,
            extra=extra,
            wp_kw=dict(
                arc_step=self.args.arc_step,
                n_arc=self.args.n_arc,
                time_step=self.args.time_step,
                n_time=self.args.n_time,
            ),
        )


def parse_log_info(text):
    dur = re.search(r"Duration:\s+([0-9.]+)", text)
    frm = re.search(r"Frames:\s+([0-9]+)", text)
    mp = re.search(r"Map:\s+([^\n\r]+)", text)
    if not (dur and frm and mp):
        return None
    return float(dur.group(1)), int(frm.group(1)), mp.group(1).strip().split("/")[-1]


def _pick_ego(world):
    vehicles = world.get_actors().filter("vehicle.*")
    for a in vehicles:
        if "tesla.model3" in a.type_id:
            return a
    return vehicles[0] if len(vehicles) > 0 else None


def find_ego(client, timeout=30):
    for _ in range(timeout):
        world = client.get_world()
        ego = _pick_ego(world)
        if ego is not None:
            return world, ego
        time.sleep(1)
    return client.get_world(), None


def _find_ego_sync(world, max_ticks=25):
    for _ in range(max_ticks):
        world.tick()
        ego = _pick_ego(world)
        if ego is not None:
            return ego, world.get_snapshot().timestamp.elapsed_seconds
    return None, None


def _wait_sync_pair(rig, target_frame, timeout=15.0):
    deadline = time.time() + timeout
    while True:
        rem = deadline - time.time()
        if rem <= 0:
            raise RuntimeError(f"Timeout ({timeout:.1f}s) esperando frame síncrono {target_frame} (callbacks={rig.buf.counts})")
        try:
            frame, item = rig.buf.ready.get(timeout=min(rem, 1.0))
        except queue.Empty:
            continue
        if frame < target_frame:
            continue
        return frame, item


class CarlaManager:
    """Supervisa conexión y ciclo de vida opcional del servidor CarlaUE4."""

    def __init__(self, carla_mod=None, host="127.0.0.1", port=2000, carla_sh="", server_log_path=None):
        self.carla = carla_mod
        self.host = host
        self.port = int(port)
        self.carla_sh = carla_sh
        self.server_log_path = server_log_path
        self.proc = None
        self.log_fp = None
        atexit.register(self.stop)

    @staticmethod
    def _port_open(host, port):
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            return False

    def _probe_world_subprocess(self, rpc_timeout=5.0):
        # Probar en subproceso evita que un Client fallido o anterior deje hilos C++/Boost.Asio vivos.
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--_internal", "probe", "--host", self.host, "--port", str(self.port),
        ]
        try:
            res = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=rpc_timeout + 4.0
            )
            return res.returncode == 0
        except subprocess.TimeoutExpired:
            return False

    def _connect_fresh(self, rpc_timeout=45.0):
        # Siempre instanciar Client nuevo para evitar estado RPC roto tras reinicio/arranque.
        if self.carla is None:
            import carla as _carla
            self.carla = _carla
        c = self.carla.Client(self.host, self.port)
        c.set_timeout(5.0)
        c.get_world()
        c.set_timeout(rpc_timeout)
        return c

    def is_alive(self):
        if self.carla_sh:
            return self.proc is not None and self.proc.poll() is None and self._port_open(self.host, self.port)
        return self._port_open(self.host, self.port)

    def start(self, wait_timeout=90.0, return_client=True):
        if not self.carla_sh:
            raise RuntimeError("Se intentó iniciar CARLA sin --carla-sh")
        self.stop()
        if self._port_open(self.host, self.port):
            raise RuntimeError(f"Puerto {self.host}:{self.port} ya ocupado antes de iniciar {self.carla_sh}")
        cmd = ["/bin/sh", self.carla_sh, "-RenderOffScreen", "-nosound"]
        if self.port != 2000:
            cmd.append(f"-carla-rpc-port={self.port}")
        cwd = os.path.dirname(os.path.abspath(self.carla_sh))
        if self.server_log_path:
            os.makedirs(os.path.dirname(self.server_log_path), exist_ok=True)
            self.log_fp = open(self.server_log_path, "a")
            stdout_dst = self.log_fp
        else:
            stdout_dst = subprocess.DEVNULL
        print(f"[CARLA] Iniciando servidor: {' '.join(cmd)}", flush=True)
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=stdout_dst, stderr=subprocess.STDOUT, start_new_session=True
        )
        t0 = time.time()
        deadline = t0 + wait_timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"CarlaUE4 terminó durante arranque (code={self.proc.returncode})")
            if self._port_open(self.host, self.port) and self._probe_world_subprocess(rpc_timeout=5.0):
                print(f"[CARLA] Listo en {time.time() - t0:.1f}s", flush=True)
                return self._connect_fresh() if return_client else None
            time.sleep(0.5)
        self.stop()
        raise RuntimeError(f"Timeout ({wait_timeout}s) esperando servidor CARLA en {self.host}:{self.port}")

    def stop(self):
        was_running = self.proc is not None
        if self.proc is not None:
            try:
                if self.proc.poll() is None:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=5)
            except Exception:
                pass
            self.proc = None
        if self.log_fp is not None:
            try:
                self.log_fp.close()
            except Exception:
                pass
            self.log_fp = None
        if was_running:
            for _ in range(15):
                if not self._port_open(self.host, self.port):
                    break
                time.sleep(0.2)

    def ensure_running(self):
        if self.carla_sh:
            if not self.is_alive():
                self.start(return_client=False)
        elif not self._port_open(self.host, self.port):
            raise RuntimeError(f"No hay servidor CARLA escuchando en {self.host}:{self.port}")

    def ensure_client(self):
        if self.carla_sh:
            if self.is_alive():
                try:
                    return self._connect_fresh()
                except Exception:
                    pass
            return self.start(return_client=True)
        return self._connect_fresh()

    def restart(self, return_client=True):
        if not self.carla_sh:
            raise RuntimeError("CARLA falló o requiere reinicio, pero no se configuró --carla-sh")
        self.stop()
        time.sleep(1.5)
        return self.start(return_client=return_client)


def _plan_chunks(duration, dt, chunk_seconds):
    total_ticks = max(1, int(math.floor(duration / dt)))
    if not chunk_seconds or chunk_seconds <= 0:
        return [(0, 0.0, 0, total_ticks)], total_ticks
    step_ticks = max(1, int(round(chunk_seconds / dt)))
    chunks, start_tick, cid = [], 0, 0
    while start_tick < total_ticks:
        n_ticks = min(step_ticks, total_ticks - start_tick)
        chunks.append((cid, start_tick * dt, start_tick, n_ticks))
        start_tick += n_ticks
        cid += 1
    return chunks, total_ticks


def _run_replay_chunk_sync(carla, client, args, out_dir, pool, name, log_path, map_name,
                           chunk_idx, n_chunks, start_s, start_tick, n_ticks, dt, total_dur,
                           prev_pose=None):
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != map_name:
        world = client.load_world(map_name)
        time.sleep(2.0)

    try:
        client.stop_replayer(False)
    except Exception:
        pass

    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = dt
    world.apply_settings(s)
    # Estabilizar ticks síncronos antes de replay_file() evita condición de carrera (~10%) tras load_world().
    world.tick()
    world.tick()

    # Con start_s > 0, CarlaReplayer queda en start_s tras tick de spawn + 1 tick síncrono al usar seek = start_s - 0.05.
    seek_s = max(0.0, float(start_s) - 0.05) if start_s > 0 else 0.0
    rep_msg = client.replay_file(log_path, seek_s, 0.0, 0)
    if not isinstance(rep_msg, str) or "Replaying" not in rep_msg:
        raise RuntimeError(f"replay_file devolvió respuesta inesperada en chunk {chunk_idx}: {rep_msg!r}")
    client.set_replayer_time_factor(1.0)

    if not world.get_settings().synchronous_mode:
        s = world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = dt
        world.apply_settings(s)

    ego, _ = _find_ego_sync(world, max_ticks=25)
    if ego is None:
        raise RuntimeError(f"sin vehículo ego en chunk {chunk_idx} ({name} @ {start_s:.1f}s)")
    # Segundo tick fija el replayer exactamente en t = start_s (ej. 120.0 / 600.0 = cierre del chunk anterior).
    world.tick()
    pre_ts = world.get_snapshot().timestamp.elapsed_seconds

    if prev_pose is not None:
        loc = ego.get_location()
        dist_seam = math.hypot(float(loc.x) - float(prev_pose[0]), float(loc.y) - float(prev_pose[1]))
        if dist_seam > 0.25:
            raise RuntimeError(
                f"Desajuste de costura en chunk {chunk_idx} (@ {start_s:.1f}s): "
                f"ego=({loc.x:.3f},{loc.y:.3f}) vs prev=({prev_pose[0]:.3f},{prev_pose[1]:.3f}), dist={dist_seam:.3f}m"
            )

    weather_str = str(world.get_weather())
    rig = Rig(carla, world, ego, sensor_tick=0.0)
    sess = Session(name, out_dir, args, pool, start_counter=start_tick)
    t0 = pre_ts - float(start_s)
    wall0, last_print = time.time(), 0
    try:
        for i in range(n_ticks):
            fid = world.tick()
            frame, item = _wait_sync_pair(rig, fid, timeout=15.0)
            sess.add(rig, frame, item, ego.get_control(), t0)
            if (i + 1) - last_print >= 200 or (i + 1) == n_ticks:
                last_print = i + 1
                el = time.time() - wall0
                sim_now = item["front"][2] - t0
                print(
                    f"  {name} [chunk {chunk_idx + 1}/{n_chunks}] | sim {sim_now:7.1f}/{total_dur:.1f}s | "
                    f"chunk {i + 1}/{n_ticks} | {((i + 1) / max(el, 1e-6)):.1f} fps real",
                    flush=True,
                )
    finally:
        try:
            client.set_timeout(3.0)
            rig.destroy()
            client.stop_replayer(False)
            s = world.get_settings()
            s.synchronous_mode = False
            world.apply_settings(s)
            client.set_timeout(45.0)
        except Exception:
            pass

    sess.flush()
    if len(sess.rows) != n_ticks:
        raise RuntimeError(f"Chunk {chunk_idx} incompleto: {len(sess.rows)}/{n_ticks} filas")
    return pd.DataFrame(sess.rows), weather_str


def _run_replay_log_async(carla, client, args, out_dir, pool, name, log_path, map_name, duration):
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != map_name:
        world = client.load_world(map_name)
        time.sleep(3)
    s = world.get_settings()
    if s.synchronous_mode:
        s.synchronous_mode = False
        world.apply_settings(s)
    client.stop_replayer(False)
    client.replay_file(log_path, 0, 0, 0)
    client.set_replayer_time_factor(1.0)

    world, ego = find_ego(client)
    if ego is None:
        raise RuntimeError("sin vehículo ego en replay async")
    rig = Rig(carla, world, ego, args.sensor_tick)
    weather = world.get_weather()
    sess = Session(name, out_dir, args, pool)
    t0, last_print, timeouts, wall0 = None, 0, 0, time.time()
    try:
        while True:
            try:
                frame, item = rig.buf.ready.get(timeout=0.2)
            except queue.Empty:
                timeouts += 1
                if timeouts > 40:  # 8 s sin frames
                    raise RuntimeError(f"Sin frames durante 8s (callbacks: {rig.buf.counts})")
                continue
            timeouts = 0
            ts = item["front"][2]
            if t0 is None:
                t0 = ts
            if ts - t0 >= duration:
                break
            sess.add(rig, frame, item, ego.get_control(), t0)
            if sess.counter - last_print >= 100:
                last_print = sess.counter
                el = time.time() - wall0
                print(
                    f"  {name} | sim {ts - t0:7.1f}/{duration:.0f}s | frames {sess.counter} | "
                    f"{sess.counter / max(el, 1e-6):.1f} fps real | pendientes {rig.buf.ready.qsize()}",
                    flush=True,
                )
    finally:
        try:
            client.set_timeout(3.0)
            rig.destroy()
            client.stop_replayer(False)
            client.set_timeout(45.0)
        except Exception:
            pass
    sess.flush()
    return pd.DataFrame(sess.rows), str(weather)


def _clean_chunk_images(out_dir, name, start_tick, n_ticks):
    """Elimina imágenes huérfanas de un rango de ticks antes de reintentar un chunk."""
    end_tick = start_tick + n_ticks
    for sub in ("front", "bev"):
        d = os.path.join(out_dir, sub, name)
        if not os.path.isdir(d):
            continue
        with os.scandir(d) as it:
            for ent in it:
                if not ent.is_file():
                    continue
                stem, _ = os.path.splitext(ent.name)
                if len(stem) == 7 and stem.isdigit():
                    idx = int(stem)
                    if start_tick <= idx < end_tick:
                        try:
                            os.remove(ent.path)
                        except OSError:
                            pass


def _is_valid_chunk_df(dfc, start_tick, n_ticks, out_dir=None):
    if len(dfc) != n_ticks or n_ticks <= 0:
        return False
    for col in ("idx", "sim_t", "ego_x", "ego_y", "has_image", "front_path", "bev_path"):
        if col not in dfc.columns:
            return False
    idx_s = pd.to_numeric(dfc["idx"], errors="coerce")
    if idx_s.isna().any() or int(idx_s.iloc[0]) != start_tick or int(idx_s.iloc[-1]) != start_tick + n_ticks - 1:
        return False
    if pd.to_numeric(dfc["sim_t"], errors="coerce").isna().any() or pd.to_numeric(dfc["ego_x"], errors="coerce").isna().any():
        return False
    if out_dir is not None:
        im_df = dfc[pd.to_numeric(dfc["has_image"], errors="coerce").fillna(0).astype(int) == 1]
        for fp, bp in zip(im_df["front_path"], im_df["bev_path"]):
            if not (isinstance(fp, str) and fp and os.path.exists(os.path.join(out_dir, fp))):
                return False
            if not (isinstance(bp, str) and bp and os.path.exists(os.path.join(out_dir, bp))):
                return False
    return True


def _query_log_info_subprocess(carla_mgr, args, log_path):
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--_internal", "info",
        "--host", args.host,
        "--port", str(args.port),
        "--_log-path", log_path,
    ]
    last_err = ""
    for attempt in range(2):
        carla_mgr.ensure_running()
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=45.0)
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if line.startswith("[INFO_JSON]"):
                        data = json.loads(line[len("[INFO_JSON]"):])
                        info = data.get("info")
                        return tuple(info) if info is not None else None
            last_err = f"code={res.returncode} stderr={res.stderr.strip()[-300:]}"
        except subprocess.TimeoutExpired:
            last_err = "timeout consultando recorder info"
        if attempt == 0 and carla_mgr.carla_sh:
            carla_mgr.restart(return_client=False)
    raise RuntimeError(f"No se pudo leer metadata de {log_path}: {last_err}")


def _run_chunk_subprocess(spec, spec_path):
    tmp_spec = spec_path + ".tmp"
    with open(tmp_spec, "w") as f:
        json.dump(spec, f, indent=2)
    os.replace(tmp_spec, spec_path)
    cmd = [
        sys.executable, os.path.abspath(__file__),
        "--_internal", "chunk",
        "--_worker-spec", spec_path,
    ]
    timeout_s = max(120.0, float(spec["n_ticks"]) * 0.35 + 120.0)
    proc = subprocess.Popen(cmd)
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"worker timeout tras {timeout_s:.0f}s")
    if rc != 0:
        raise RuntimeError(f"worker terminó con código {rc}")


def _run_internal_action(args):
    import carla

    if args._internal == "probe":
        c = carla.Client(args.host, int(args.port))
        c.set_timeout(5.0)
        c.get_world()
        return

    if args._internal == "info":
        c = carla.Client(args.host, int(args.port))
        c.set_timeout(20.0)
        info = parse_log_info(c.show_recorder_file_info(args._log_path, False))
        print("[INFO_JSON]" + json.dumps({"info": list(info) if info is not None else None}), flush=True)
        return

    if args._internal == "chunk":
        with open(args._worker_spec, "r") as f:
            spec = json.load(f)
        _clean_chunk_images(spec["out_dir"], spec["name"], int(spec["start_tick"]), int(spec["n_ticks"]))
        cv2.setNumThreads(2)
        c = carla.Client(spec["host"], int(spec["port"]))
        c.set_timeout(45.0)
        w_args = argparse.Namespace(jpeg_quality=int(spec["jpeg_quality"]), bev_format=spec["bev_format"])
        with ThreadPoolExecutor(max_workers=int(spec["workers"])) as pool:
            dfc, w_str = _run_replay_chunk_sync(
                carla, c, w_args, spec["out_dir"], pool,
                spec["name"], spec["log_path"], spec["map_name"],
                int(spec["cid"]), int(spec["n_chunks"]),
                float(spec["start_s"]), int(spec["start_tick"]), int(spec["n_ticks"]),
                float(spec["dt"]), float(spec["total_dur"]),
                prev_pose=spec.get("prev_pose"),
            )
        pid_csv = f"{spec['result_csv']}.{os.getpid()}"
        dfc.to_csv(pid_csv, index=False, float_format="%.5f")
        os.replace(pid_csv, spec["result_csv"])
        if spec.get("weather_out") and w_str:
            pid_w = f"{spec['weather_out']}.{os.getpid()}"
            with open(pid_w, "w") as wf:
                json.dump({"weather": w_str}, wf)
            os.replace(pid_w, spec["weather_out"])
        return


def run_replay(carla, carla_mgr, args, out_dir, pool, done, csv_path, sessions_path):
    logs = sorted(glob.glob(os.path.join(args.logs_dir, "*.log")))
    if args.limit_logs:
        logs = logs[:args.limit_logs]

    client = None
    prev_map = None
    chunks_since_restart = 0
    dt = float(args.sensor_tick)

    for li, log_path in enumerate(logs, 1):
        name = os.path.splitext(os.path.basename(log_path))[0]
        if name in done:
            print(f"[SKIP] {name} ya procesado", flush=True)
            continue

        if args.async_replay:
            if client is None:
                client = carla_mgr.ensure_client()
            info = parse_log_info(client.show_recorder_file_info(log_path, False))
        else:
            info = _query_log_info_subprocess(carla_mgr, args, log_path)

        if info is None:
            print(f"[WARN] {name}: log inválido, se omite", flush=True)
            continue
        duration, frames_total, map_name = info
        if args.max_seconds:
            duration = min(duration, args.max_seconds)

        print(f"\n{'=' * 60}\n[LOG {li}/{len(logs)}] {name} | {duration:.1f}s | {map_name}\n{'=' * 60}", flush=True)

        if args.async_replay:
            df_all, weather_str = _run_replay_log_async(carla, client, args, out_dir, pool, name, log_path, map_name, duration)
        else:
            ckpt_dir = os.path.join(out_dir, "checkpoints", name)
            os.makedirs(ckpt_dir, exist_ok=True)
            chunks, _ = _plan_chunks(duration, dt, args.chunk_seconds)
            chunk_dfs = []
            weather_str = None
            meta_ckpt_path = os.path.join(ckpt_dir, "log_meta.json")
            if os.path.exists(meta_ckpt_path):
                try:
                    weather_str = json.load(open(meta_ckpt_path)).get("weather")
                except Exception:
                    weather_str = None

            spec_path = os.path.join(ckpt_dir, "worker_spec.json")
            for cid, start_s, start_tick, n_ticks in chunks:
                cfile = os.path.join(ckpt_dir, f"chunk_{cid:04d}.csv")
                if os.path.exists(cfile):
                    dfc = pd.read_csv(cfile, keep_default_na=False)
                    if _is_valid_chunk_df(dfc, start_tick, n_ticks, out_dir=out_dir):
                        print(f"  [SKIP CHUNK] {name} {cid + 1}/{len(chunks)} ({len(dfc)} filas)", flush=True)
                        chunk_dfs.append(dfc)
                        continue
                    print(f"  [REDO CHUNK] {name} {cid + 1}/{len(chunks)} inválido/incompleto ({len(dfc)}/{n_ticks})", flush=True)

                if carla_mgr.carla_sh and (
                    (prev_map is not None and prev_map != map_name)
                    or (args.restart_chunks > 0 and chunks_since_restart >= args.restart_chunks)
                    or (not carla_mgr.is_alive())
                ):
                    print(f"  [CARLA] Reinicio preventivo tras {chunks_since_restart} chunks...", flush=True)
                    carla_mgr.restart(return_client=False)
                    chunks_since_restart = 0
                prev_map = map_name

                prev_pose = None
                if chunk_dfs:
                    prev_pose = [
                        float(chunk_dfs[-1]["ego_x"].iloc[-1]),
                        float(chunk_dfs[-1]["ego_y"].iloc[-1]),
                    ]
                tmp_cfile = cfile + ".tmp"
                tmp_wfile = os.path.join(ckpt_dir, f"weather_{cid:04d}.json")
                spec = dict(
                    out_dir=out_dir,
                    name=name,
                    log_path=log_path,
                    map_name=map_name,
                    cid=cid,
                    n_chunks=len(chunks),
                    start_s=start_s,
                    start_tick=start_tick,
                    n_ticks=n_ticks,
                    dt=dt,
                    total_dur=duration,
                    jpeg_quality=args.jpeg_quality,
                    bev_format=args.bev_format,
                    workers=args.workers,
                    host=args.host,
                    port=args.port,
                    result_csv=tmp_cfile,
                    weather_out=tmp_wfile if weather_str is None else None,
                    prev_pose=prev_pose,
                )

                dfc, w_str = None, None
                for attempt in range(1, args.max_retries + 1):
                    try:
                        carla_mgr.ensure_running()
                        _run_chunk_subprocess(spec, spec_path)
                        if not os.path.exists(tmp_cfile):
                            raise RuntimeError("worker terminó sin generar CSV temporal")
                        dfc = pd.read_csv(tmp_cfile, keep_default_na=False)
                        if not _is_valid_chunk_df(dfc, start_tick, n_ticks, out_dir=out_dir):
                            raise RuntimeError(f"chunk CSV inválido ({len(dfc)}/{n_ticks} filas)")
                        if os.path.exists(tmp_wfile):
                            try:
                                w_str = json.load(open(tmp_wfile)).get("weather")
                            except Exception:
                                w_str = None
                        break
                    except Exception as e:
                        for fp in (tmp_cfile, tmp_wfile):
                            if os.path.exists(fp):
                                try:
                                    os.remove(fp)
                                except OSError:
                                    pass
                        print(f"  [WARN] {name} chunk {cid + 1}/{len(chunks)} intento {attempt}/{args.max_retries} falló: {e}", flush=True)
                        if attempt >= args.max_retries:
                            raise
                        if carla_mgr.carla_sh:
                            carla_mgr.restart(return_client=False)
                            chunks_since_restart = 0

                if os.path.exists(tmp_wfile):
                    try:
                        os.remove(tmp_wfile)
                    except OSError:
                        pass
                if weather_str is None and w_str:
                    weather_str = w_str
                    with open(meta_ckpt_path, "w") as mf:
                        json.dump(dict(map=map_name, duration=duration, frames=frames_total, weather=weather_str), mf, indent=2)

                os.replace(tmp_cfile, cfile)
                chunk_dfs.append(dfc)
                chunks_since_restart += 1

            if os.path.exists(spec_path):
                try:
                    os.remove(spec_path)
                except OSError:
                    pass
            df_all = pd.concat(chunk_dfs, ignore_index=True)

        if weather_str is None:
            weather_str = ""

        min_required_t = max(0.0, duration - max(2.0, 3.0 * dt))
        last_sim_t = float(df_all["sim_t"].iloc[-1]) if len(df_all) > 0 else -1.0
        if len(df_all) < 3 or last_sim_t < min_required_t:
            raise RuntimeError(
                f"Log {name} INCOMPLETO: {len(df_all)} filas, last_sim_t={last_sim_t:.2f}s < requerido {min_required_t:.2f}s. NO se marca como completado."
            )

        res = finalize_session_df(
            name, df_all, csv_path,
            extra=dict(map=map_name, source="replay", weather=weather_str),
            wp_kw=dict(arc_step=args.arc_step, n_arc=args.n_arc, time_step=args.time_step, n_time=args.n_time),
        )
        print(f"[OK] {name}: {res}", flush=True)
        with open(sessions_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        done.add(name)
        with open(os.path.join(out_dir, "done_logs.txt"), "a") as f:
            f.write(name + "\n")

    if carla_mgr.carla_sh:
        carla_mgr.stop()


def run_autopilot(carla, client, args, out_dir, pool, csv_path, sessions_path):
    world = client.load_world(args.town)
    time.sleep(3)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    random.seed(args.seed)
    lib = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    stride = max(1, int(round(args.sensor_tick / 0.05)))
    run_name = f"autopilot_{args.town}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    total_sim = args.minutes * 60.0
    seg_len = args.segment_minutes * 60.0
    seg_idx, sim_done = 0, 0.0

    while sim_done < total_sim:
        ego = None
        for _ in range(20):
            bp = lib.find("vehicle.tesla.model3")
            bp.set_attribute("role_name", "ego")
            ego = world.try_spawn_actor(bp, random.choice(spawns))
            if ego:
                break
        if ego is None:
            raise RuntimeError("No se pudo spawnear el ego")
        col_bp = lib.find("sensor.other.collision")
        col = world.spawn_actor(col_bp, carla.Transform(), attach_to=ego)
        collided = {"v": False}
        col.listen(lambda e: collided.__setitem__("v", True))
        rig = Rig(carla, world, ego, args.sensor_tick)
        ego.set_autopilot(True, tm.get_port())
        tm.ignore_lights_percentage(ego, 100)
        tm.ignore_signs_percentage(ego, 100)
        tm.auto_lane_change(ego, False)
        tm.distance_to_leading_vehicle(ego, 4.0)

        name = f"{run_name}_{seg_idx:02d}"
        sess = Session(name, out_dir, args, pool)
        t0 = None
        next_speed_change, next_dagger = 0.0, args.dagger_interval
        seg_t = 0.0
        expected, wall0, last_print = None, time.time(), 0
        print(f"\n[SEGMENTO {seg_idx}] {name}", flush=True)
        try:
            while seg_t < seg_len and sim_done + seg_t < total_sim:
                world.tick()
                seg_t += 0.05
                if collided["v"] or ego.get_location().z < -2.0:
                    print("[ALERTA] colisión/caída: reubicando", flush=True)
                    wp = world.get_map().get_waypoint(random.choice(spawns).location)
                    tf = wp.transform
                    tf.location.z += 0.5
                    ego.set_transform(tf)
                    collided["v"] = False
                    next_dagger = seg_t + args.dagger_interval
                    rig.buf.clear()
                    expected = None
                    continue
                if seg_t >= next_speed_change:
                    target = random.uniform(*args.speed_range)
                    limit = max(ego.get_speed_limit(), 30.0)
                    tm.vehicle_percentage_speed_difference(ego, float(np.clip((1 - target / limit) * 100, -50, 90)))
                    next_speed_change = seg_t + random.uniform(20, 60)
                if args.dagger_interval > 0 and seg_t >= next_dagger:
                    tf = ego.get_transform()
                    rv = tf.get_right_vector()
                    d, sgn = random.uniform(0.8, 1.5), random.choice([-1, 1])
                    tf.location.x += rv.x * d * sgn
                    tf.location.y += rv.y * d * sgn
                    tf.location.z += 0.1
                    tf.rotation.yaw += random.uniform(3.0, 12.0) * sgn
                    ego.set_transform(tf)
                    next_dagger = seg_t + args.dagger_interval * random.uniform(0.7, 1.3)
                while not rig.buf.ready.empty():
                    try:
                        frame, item = rig.buf.ready.get_nowait()
                    except queue.Empty:
                        break
                    if t0 is None:
                        t0 = item["front"][2]
                    sess.add(rig, frame, item, ego.get_control(), t0)
                    if sess.counter - last_print >= 100:
                        last_print = sess.counter
                        el = time.time() - wall0
                        print(f"  {name} | sim {seg_t:7.1f}/{seg_len:.0f}s | frames {sess.counter} | "
                              f"{sess.counter / max(el, 1e-6):.1f} fps real", flush=True)
        finally:
            rig.destroy()
            try:
                col.stop()
                col.destroy()
                ego.set_autopilot(False, tm.get_port())
                ego.destroy()
            except Exception:
                pass
        res = sess.finalize(csv_path, extra=dict(map=args.town, source="autopilot", weather=str(world.get_weather())))
        print(f"[OK] {name}: {res}", flush=True)
        with open(sessions_path, "a") as f:
            f.write(json.dumps(res) + "\n")
        sim_done += seg_t
        seg_idx += 1

    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)
    tm.set_synchronous_mode(False)


def run_rebuild(args, out_dir):
    csv_path = os.path.join(out_dir, "dataset.csv")
    df = pd.read_csv(csv_path)
    base_cols = [c for c in df.columns if not (re.match(r"wp[at]\d+_[xy]", c) or c in (
        "wp_arc_valid", "wp_time_valid", "segment", "discontinuity"))]
    outs = []
    for name, g in df[base_cols].groupby("session", sort=False):
        outs.append(attach_waypoints(g.sort_values("sim_t").reset_index(drop=True),
                                     arc_step=args.arc_step, n_arc=args.n_arc,
                                     time_step=args.time_step, n_time=args.n_time))
    res = pd.concat(outs, ignore_index=True)
    dst = os.path.join(out_dir, f"dataset_arc{args.arc_step:g}x{args.n_arc}_t{args.time_step:g}x{args.n_time}.csv")
    res.to_csv(dst, index=False, float_format="%.5f")
    print(f"[OK] {dst}: {len(res)} filas")


def write_meta(out_dir, args):
    mpp = 2 * BEV["z"] * math.tan(math.radians(BEV["fov"] / 2)) / BEV["size"]
    meta = dict(
        created=datetime.now().isoformat(timespec="seconds"),
        conventions=dict(
            ego_frame="x = adelante, y = DERECHA (CARLA); steering > 0 gira a la derecha; yaw en grados (CARLA)",
            waypoints=dict(
                arc=dict(step_m=ARC_STEP, n=N_ARC, columns="wpa{k}_x, wpa{k}_y (k=1..n)"),
                time=dict(step_s=TIME_STEP, n=N_TIME, columns="wpt{k}_x, wpt{k}_y (k=1..n)"),
                validity="usar wp_arc_valid / wp_time_valid == 1 (ventana futura completa y sin saltos)"),
            filtering_hint="entrenar con has_image==1 & wp_arc_valid==1; speed_kmh se calcula por diferencias de pose",
        ),
        front_camera=FRONT,
        bev_camera=dict(**BEV, meters_per_pixel=mpp,
                        pixel_to_ego="fwd = x_cam + (H/2 - v)*mpp ; right = (u - W/2)*mpp ; arriba de la imagen = adelante del auto"),
        front_format=f"jpg q{args.jpeg_quality}", bev_format=args.bev_format,
        sensor_tick_s=args.sensor_tick,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["replay", "autopilot", "rebuild"], default="replay")
    p.add_argument("--logs-dir", default="/home/daniel/code/2025-phd-daniel-guerrero/data/logs_dagger_expert")
    p.add_argument("--out-dir", default="/home/daniel/code/2025-phd-daniel-guerrero/data/dataset_waypoints")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--sensor-tick", type=float, default=0.1)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--bev-format", choices=["png", "jpg"], default="png")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit-logs", type=int, default=0)
    p.add_argument("--max-seconds", type=float, default=0, help="(replay) recorta cada log, para pruebas")
    p.add_argument("--chunk-seconds", type=float, default=600.0,
                   help="(replay) segundos por chunk con checkpoint (0=sin chunks)")
    p.add_argument("--carla-sh", default="",
                   help="Ruta a CarlaUE4.sh para auto-iniciar y reiniciar CARLA ante chunks/fallos")
    p.add_argument("--restart-chunks", type=int, default=1,
                   help="Con --carla-sh, reiniciar CARLA cada N chunks (0=solo por cambio de mapa o fallo)")
    p.add_argument("--max-retries", type=int, default=3, help="Reintentos por chunk fallido en replay")
    p.add_argument("--async-replay", action="store_true", help="Usar replay asíncrono heredado")
    p.add_argument("--town", default="Town04_Opt")
    p.add_argument("--minutes", type=float, default=30)
    p.add_argument("--segment-minutes", type=float, default=10)
    p.add_argument("--dagger-interval", type=float, default=10.0)
    p.add_argument("--speed-range", type=float, nargs=2, default=[25.0, 55.0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arc-step", type=float, default=ARC_STEP)
    p.add_argument("--n-arc", type=int, default=N_ARC)
    p.add_argument("--time-step", type=float, default=TIME_STEP)
    p.add_argument("--n-time", type=int, default=N_TIME)
    p.add_argument("--_internal", choices=["probe", "info", "chunk"], default="", help=argparse.SUPPRESS)
    p.add_argument("--_log-path", default="", help=argparse.SUPPRESS)
    p.add_argument("--_worker-spec", default="", help=argparse.SUPPRESS)
    args = p.parse_args()

    if args._internal:
        _run_internal_action(args)
        return

    os.makedirs(args.out_dir, exist_ok=True)
    if args.mode == "rebuild":
        run_rebuild(args, args.out_dir)
        return

    write_meta(args.out_dir, args)
    csv_path = os.path.join(args.out_dir, "dataset.csv")
    sessions_path = os.path.join(args.out_dir, "sessions.jsonl")
    done_path = os.path.join(args.out_dir, "done_logs.txt")
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()

    use_parent_carla = (args.mode != "replay") or args.async_replay
    carla = None
    if use_parent_carla:
        import carla as _carla
        carla = _carla

    cv2.setNumThreads(2)
    carla_mgr = CarlaManager(
        carla, args.host, args.port, carla_sh=args.carla_sh,
        server_log_path=os.path.join(args.out_dir, "carla_server.log"),
    )
    try:
        if args.mode == "replay" and not args.async_replay:
            run_replay(None, carla_mgr, args, args.out_dir, None, done, csv_path, sessions_path)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                if args.mode == "replay":
                    run_replay(carla, carla_mgr, args, args.out_dir, pool, done, csv_path, sessions_path)
                else:
                    client = carla_mgr.ensure_client()
                    run_autopilot(carla, client, args, args.out_dir, pool, csv_path, sessions_path)
    except KeyboardInterrupt:
        print("\n[INFO] Interrumpido por el usuario.", flush=True)
    finally:
        if args.carla_sh:
            carla_mgr.stop()
    print("\n[FIN] Dataset en:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
