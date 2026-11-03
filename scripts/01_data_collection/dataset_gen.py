import cv2
import serial
import can
import csv
import time
from datetime import datetime
import threading
import os
import sys
import signal
import subprocess

# ---------------- CONFIGURACIÓN ----------------
CAMERA_ID = 0
ESP32_PORT = '/dev/ttyUSB0'
ESP32_BAUD = 115200

CAN_BUSTYPE = 'socketcan' 
CAN_CHANNEL = 'can0'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PID_FILE = os.path.join(BASE_DIR, '.dataset_gen.pid')
STATE_FILE = os.path.join(BASE_DIR, '.dataset_gen.state')
STOP_FLAG_FILE = os.path.join(BASE_DIR, '.dataset_gen.stop')
LOG_FILE = os.path.join(BASE_DIR, 'dataset_gen.log')

# Variables globales
current_steering = ""  # Ahora es un string para guardar el dato crudo
current_throttle = 0.0
running = True
data_lock = threading.Lock()


def safe_print(msg):
    """Imprime en consola sin fallar si la sesión SSH/Termux se desconecta al bloquear el celular."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    line = f"[{timestamp}] {msg}"
    try:
        print(msg, flush=True)
    except (BrokenPipeError, OSError):
        pass
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as lf:
            lf.write(line + '\n')
    except OSError:
        pass


def handle_stop_signal(signum, frame):
    """Manejador de señales (SIGINT/SIGTERM) para detener la recolección de forma segura."""
    global running
    sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    safe_print(f"\n[INFO] Señal {sig_name} recibida. Cerrando archivo CSV y liberando cámara...")
    running = False


# ---------------- HILOS DE LECTURA ----------------

def read_esp32():
    """Hilo para leer datos crudos desde el ESP32."""
    global current_steering, running
    try:
        ser = serial.Serial(ESP32_PORT, ESP32_BAUD, timeout=1)
        safe_print(f"[OK] Conectado al ESP32 en {ESP32_PORT}")
        while running:
            if ser.in_waiting > 0:
                # Lee la línea, limpia saltos de carro y la guarda tal cual
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line: # Solo si no está vacía
                    with data_lock:
                        current_steering = line
    except Exception as e:
        safe_print(f"[ERROR] ESP32: {e}")

def read_can():
    """Hilo para leer el acelerador (ID 0x0C9) usando socketcan."""
    global current_throttle, running
    try:
        # Usamos interface en lugar de bustype para evitar el warning
        bus = can.interface.Bus(interface=CAN_BUSTYPE, channel=CAN_CHANNEL)
        safe_print(f"[OK] Conectado al bus CAN en {CAN_CHANNEL}")
        
        while running:
            msg = bus.recv(timeout=1.0)
            if msg is not None:
                if msg.arbitration_id == 0x0C9: 
                    raw_val = msg.data[0] 
                    with data_lock:
                        current_throttle = raw_val
    except Exception as e:
        safe_print(f"[ERROR] CAN: {e}")


# ---------------- CONTROL DE PROCESO (START / STOP / STATUS) ----------------

def get_running_pid():
    """Devuelve el PID si el recolector está corriendo actualmente, o None."""
    if not os.path.exists(PID_FILE):
        return None
    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        try:
            os.remove(PID_FILE)
        except OSError:
            pass
        return None


def stop_collector():
    """Envía señal de parada segura al proceso en ejecución y espera a que guarde todo."""
    pid = get_running_pid()
    if pid is None:
        print("[INFO] No hay ninguna recolección corriendo en este momento.")
        return 0

    print(f"[INFO] Deteniendo recolección de forma segura (PID {pid})...")
    # Crear archivo bandera y enviar SIGINT
    try:
        with open(STOP_FLAG_FILE, 'w') as f:
            f.write('stop\n')
    except OSError:
        pass

    try:
        os.kill(pid, signal.SIGINT)
    except OSError as e:
        print(f"[ERROR] No se pudo enviar señal al proceso {pid}: {e}")
        return 1

    # Esperar hasta 10 segundos a que cierre limpiamente
    for _ in range(50):
        if get_running_pid() is None:
            print("[OK] Recolección detenida y guardada correctamente.")
            if os.path.exists(STATE_FILE):
                with open(STATE_FILE, 'r', encoding='utf-8') as sf:
                    print(sf.read().strip())
            return 0
        time.sleep(0.2)

    print("[WARNING] El proceso tardó en responder; enviando SIGTERM...")
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    return 0


def status_collector():
    """Muestra el estado actual de la recolección."""
    pid = get_running_pid()
    if pid is not None:
        print(f"[ACTIVO] Recolector corriendo en segundo plano (PID: {pid})")
    else:
        print("[INACTIVO] El recolector NO está corriendo.")

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r', encoding='utf-8') as sf:
            print(sf.read().strip())
    return 0


def start_background():
    """Lanza el script en segundo plano desacoplado de la terminal SSH/Termux."""
    pid = get_running_pid()
    if pid is not None:
        print(f"[INFO] Ya hay una recolección corriendo (PID: {pid}). Usa '--status' o '--stop'.")
        return 1

    log_f = open(LOG_FILE, 'a', encoding='utf-8')
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__)],
        cwd=BASE_DIR,
        stdin=subprocess.DEVNULL,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    time.sleep(1.0)
    if proc.poll() is None:
        print(f"[OK] Recolección iniciada en segundo plano (PID: {proc.pid}).")
        print("     Puedes bloquear tu celular o cerrar Termux sin que se detenga.")
        print("     Para detener de forma segura ejecuta: detener  (o python dataset_gen.py --stop)")
        print("     Para ver el progreso ejecuta:         estado   (o python dataset_gen.py --status)")
        return 0
    else:
        print("[ERROR] El proceso terminó prematuramente. Últimas líneas del log:")
        try:
            with open(LOG_FILE, 'r', encoding='utf-8') as lf:
                print(''.join(lf.readlines()[-15:]))
        except OSError:
            pass
        return 1


# ---------------- HILO PRINCIPAL ----------------

def main():
    global running, current_steering, current_throttle

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ('--stop', 'stop', 'detener'):
            sys.exit(stop_collector())
        elif arg in ('--status', 'status', 'estado'):
            sys.exit(status_collector())
        elif arg in ('--bg', '--start', 'start', 'iniciar'):
            sys.exit(start_background())

    existing_pid = get_running_pid()
    if existing_pid is not None and existing_pid != os.getpid():
        print(f"[ERROR] Ya hay otra instancia de dataset_gen.py corriendo (PID {existing_pid}).")
        print("        Deténla primero con: python dataset_gen.py --stop")
        sys.exit(1)

    # Ignorar SIGHUP para que al bloquear el celular / caer el SSH no muera el proceso
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, handle_stop_signal)
    signal.signal(signal.SIGTERM, handle_stop_signal)

    # Limpiar bandera de parada previa y registrar PID actual
    if os.path.exists(STOP_FLAG_FILE):
        try:
            os.remove(STOP_FLAG_FILE)
        except OSError:
            pass

    with open(PID_FILE, 'w') as pf:
        pf.write(f"{os.getpid()}\n")
    
    # 1. Crear nombre de carpeta dinámico con fecha y hora
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    DATASET_DIR = os.path.join(BASE_DIR, f"dataset_{timestamp_str}")
    IMAGES_DIR = os.path.join(DATASET_DIR, "imagenes")
    CSV_FILE = os.path.join(DATASET_DIR, "registro.csv")
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # 2. Iniciar hilos
    thread_esp = threading.Thread(target=read_esp32, daemon=True)
    thread_can = threading.Thread(target=read_can, daemon=True)
    thread_esp.start()
    thread_can.start()
    
    # 3. Iniciar cámara
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        safe_print(f"[ERROR] No se pudo abrir la cámara {CAMERA_ID}")
        running = False
        try:
            os.remove(PID_FILE)
        except OSError:
            pass
        sys.exit(1)

    safe_print(f"\n[OK] Todo listo. Guardando en la carpeta: {DATASET_DIR}/")
    safe_print("-------------------------------------------------")
    safe_print("Aunque bloquees el celular o se desconecte SSH, seguirá grabando.")
    safe_print("Para detener: presiona Ctrl+C o ejecuta 'detener' / 'python dataset_gen.py --stop'")
    safe_print("-------------------------------------------------\n")
    
    # Configuración de tiempo
    save_interval = 0.100  # 100 milisegundos
    last_save_time = 0
    frame_count = 0

    try:
        # Preparar el archivo CSV
        with open(CSV_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'imagen', 'steering_raw', 'throttle_raw'])
            file.flush()
            
            while running:
                if os.path.exists(STOP_FLAG_FILE):
                    safe_print("\n[INFO] Archivo de parada detectado. Deteniendo...")
                    running = False
                    break

                # Leer constantemente limpia el buffer de la cámara para que no haya lag
                ret, frame = cap.read()
                if not ret:
                    safe_print("[WARNING] Fallo al capturar frame de la cámara. Reintentando...")
                    time.sleep(0.01)
                    continue
                    
                now = time.time()
                
                # Solo guardamos si han pasado 100ms (0.1s) desde la última captura
                if (now - last_save_time) >= save_interval:
                    last_save_time = now
                    
                    img_name = f"frame_{now:.3f}.jpg"
                    img_path = os.path.join(IMAGES_DIR, img_name)
                    
                    # Guardar la imagen en disco
                    cv2.imwrite(img_path, frame)
                    
                    # Extraer las variables de los otros hilos de forma segura
                    with data_lock:
                        steer = current_steering
                        thrott = current_throttle
                        
                    # Escribir en el archivo y sincronizar a disco para evitar corrupción
                    writer.writerow([now, img_name, steer, thrott])
                    frame_count += 1
                    if frame_count % 10 == 0:
                        file.flush()
                        os.fsync(file.fileno())
                        try:
                            with open(STATE_FILE, 'w', encoding='utf-8') as sf:
                                sf.write(
                                    f"Carpeta: {DATASET_DIR}\n"
                                    f"Frames guardados: {frame_count}\n"
                                    f"Último steering: {steer} | throttle: {thrott}\n"
                                    f"Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                                )
                        except OSError:
                            pass

            # Asegurar vaciado final del CSV antes de cerrar
            file.flush()
            os.fsync(file.fileno())
                    
    except KeyboardInterrupt:
        # Esto se ejecuta cuando presionas Ctrl+C
        safe_print("\n[INFO] Señal de interrupción recibida (Ctrl+C). Deteniendo...")
        running = False
    finally:
        # Limpieza al salir
        cap.release()
        for fpath in (PID_FILE, STOP_FLAG_FILE):
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError:
                    pass
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as sf:
                sf.write(
                    f"Última sesión finalizada: {DATASET_DIR}\n"
                    f"Total de frames guardados: {frame_count}\n"
                    f"Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
        except OSError:
            pass
        safe_print(f"[INFO] Recolección finalizada correctamente ({frame_count} frames). Dataset guardado en: {DATASET_DIR}/")

if __name__ == '__main__':
    main()
