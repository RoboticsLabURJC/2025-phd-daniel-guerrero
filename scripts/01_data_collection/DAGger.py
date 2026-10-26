# joystick_con_ruido_temporal.py
#
# Script para inspeccionar un joystick utilizando pygame.
# Adicionalmente, se aplica un ruido temporal a los valores de los ejes:
# el ruido solo se activa cada cierto intervalo de tiempo (por defecto, 5 segundos).
#
# En un sistema de aprendizaje por imitación (por ejemplo, DAgger),
# el valor "experto" representaría la acción limpia del humano,
# mientras que el valor con ruido podría utilizarse como acción ejecutada
# por el sistema para generar estados más variados.

import pygame
import sys
import time
import numpy as np

# Parámetros de lectura del joystick
DEADZONE = 0.03   # umbral para ignorar pequeños movimientos en los ejes
ROUND_TO = 3      # número de decimales para redondear

# Parámetros de ruido temporal
NOISE_STD = 0.15        # intensidad del ruido gaussiano
NOISE_INTERVAL = 5.0    # segundos entre perturbaciones

# Registro del último instante en que se aplicó ruido
last_noise_time = time.time()


def apply_temporal_noise(action):
    """
    Recibe una acción (por ejemplo, un vector con valores de control)
    y aplica ruido gaussiano solo si ha transcurrido un intervalo de tiempo
    mayor o igual a NOISE_INTERVAL desde la última perturbación.

    Parámetros:
        action (array-like): acción original (sin ruido).

    Retorna:
        noisy_action (array-like): acción modificada o no, según corresponda.
        applied (bool): indica si en esta llamada se aplicó ruido temporal.
    """
    global last_noise_time

    current_time = time.time()

    # Verifica si han pasado los segundos definidos por NOISE_INTERVAL
    if current_time - last_noise_time >= NOISE_INTERVAL:
        # Crea una copia en forma de arreglo numpy
        action = np.array(action, dtype=float)

        # Aplica ruido gaussiano según la desviación estándar definida
        noisy_action = action + np.random.normal(0.0, NOISE_STD, size=action.shape)

        # Limita el rango de la acción a [-1, 1] (típico en ejes de joystick)
        noisy_action = np.clip(noisy_action, -1.0, 1.0)

        # Actualiza el instante de la última perturbación
        last_noise_time = current_time

        return noisy_action, True
    else:
        # No se aplica perturbación en este instante
        return np.array(action, dtype=float), False


def init_joysticks():
    """
    Inicializa el subsistema de joysticks de pygame y registra
    todos los dispositivos detectados.
    """
    pygame.joystick.init()
    joysticks = []

    for i in range(pygame.joystick.get_count()):
        js = pygame.joystick.Joystick(i)
        js.init()
        joysticks.append(js)
        print(
            f"[INFO] Joystick #{i}: {js.get_name()} | "
            f"ejes={js.get_numaxes()} | botones={js.get_numbuttons()} | hats={js.get_numhats()}"
        )

    if not joysticks:
        print("[ADVERTENCIA] No se detectaron joysticks o volantes conectados.")

    return joysticks


def main():
    """
    Bucle principal de la aplicación.

    Lee eventos de joystick y teclado, muestra cambios en ejes, botones
    y crucetas, e imprime tanto el valor experto (sin ruido) como el valor
    que incorpora ruido temporal.
    """
    pygame.init()
    screen = pygame.display.set_mode((640, 200))
    pygame.display.set_caption("Inspector de Joystick con ruido temporal (Q/ESC para salir)")
    clock = pygame.time.Clock()

    joysticks = init_joysticks()

    # Diccionarios para almacenar el último estado conocido
    last_axes = {}
    last_buttons = {}
    last_hats = {}

    for js in joysticks:
        jid = js.get_id()
        last_axes[jid] = [None] * js.get_numaxes()
        last_buttons[jid] = [None] * js.get_numbuttons()
        last_hats[jid] = [None] * js.get_numhats()

    running = True
    while running:
        for event in pygame.event.get():
            # Salida por cierre de ventana
            if event.type == pygame.QUIT:
                running = False

            # Salida por tecla ESC o Q
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                running = False

            # Detección de joystick
            if event.type == pygame.JOYDEVICEADDED:
                js = pygame.joystick.Joystick(event.device_index)
                js.init()
                jid = js.get_id()
                print(f"[CONECTADO] Joystick #{jid}: {js.get_name()}")
                last_axes[jid] = [None] * js.get_numaxes()
                last_buttons[jid] = [None] * js.get_numbuttons()
                last_hats[jid] = [None] * js.get_numhats()

            # Detección de joystick desconectado en caliente
            if event.type == pygame.JOYDEVICEREMOVED:
                jid = event.instance_id
                print(f"[DESCONECTADO] Joystick #{jid}")
                last_axes.pop(jid, None)
                last_buttons.pop(jid, None)
                last_hats.pop(jid, None)

            # Manejo del movimiento en un eje
            if event.type == pygame.JOYAXISMOTION:
                jid = event.instance_id
                axis = event.axis
                val = event.value

                # Aplica zona muerta para eliminar pequeños ruidos
                if abs(val) < DEADZONE:
                    val = 0.0

                # Valor experto: lectura limpia del eje
                expert_val = round(val, ROUND_TO)

                prev = None
                if jid in last_axes and axis < len(last_axes[jid]):
                    prev = last_axes[jid][axis]

                # Solo imprimir si el valor experto cambió
                if prev != expert_val:
                    # Convertir el valor experto en una "acción" de 1 dimensión
                    action = np.array([expert_val], dtype=float)

                    # Ruido temporal
                    noisy_action, applied = apply_temporal_noise(action)

                    print(
                        f"[EJE] js={jid} axis={axis} "
                        f"expert={expert_val} noisy={float(noisy_action[0])} ruido_aplicado={applied}"
                    )

                    if jid in last_axes and axis < len(last_axes[jid]):
                        last_axes[jid][axis] = expert_val

            # Manejo de pulsación de botón
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"[BOTON] js={event.instance_id} button={event.button} DOWN")
                if event.instance_id in last_buttons and event.button < len(last_buttons[event.instance_id]):
                    last_buttons[event.instance_id][event.button] = 1

            # Manejo de liberación de botón
            if event.type == pygame.JOYBUTTONUP:
                print(f"[BOTON] js={event.instance_id} button={event.button} UP")
                if event.instance_id in last_buttons and event.button < len(last_buttons[event.instance_id]):
                    last_buttons[event.instance_id][event.button] = 0

            # Manejo de cruceta (HAT)
            if event.type == pygame.JOYHATMOTION:
                print(f"[HAT] js={event.instance_id} hat={event.hat} value={event.value}")
                if event.instance_id in last_hats and event.hat < len(last_hats[event.instance_id]):
                    last_hats[event.instance_id][event.hat] = event.value

        # Actualización básica de la ventana
        screen.fill((20, 20, 20))
        pygame.display.flip()

        # Control de la frecuencia del bucle principal
        clock.tick(200)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
