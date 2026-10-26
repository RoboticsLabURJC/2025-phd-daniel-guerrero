# leer_joystick.py
import pygame
import sys
import time

DEADZONE = 0.03   # ignora pequeños ruidos en los ejes
ROUND_TO = 3      # decimales al imprimir valores de ejes

def init_joysticks():
    pygame.joystick.init()
    joysticks = []
    for i in range(pygame.joystick.get_count()):
        js = pygame.joystick.Joystick(i)
        js.init()
        joysticks.append(js)
        print(f"[INFO] Joystick #{i}: {js.get_name()} | ejes={js.get_numaxes()} | botones={js.get_numbuttons()} | hats={js.get_numhats()}")
    if not joysticks:
        print("[ADVERTENCIA] No se detectaron joysticks/volantes.")
    return joysticks

def main():
    pygame.init()
    screen = pygame.display.set_mode((640, 200))
    pygame.display.set_caption("Inspector de Joystick (Q/ESC para salir)")
    clock = pygame.time.Clock()

    joysticks = init_joysticks()

    print("\n[AYUDA] Mueve el volante/palancas, presiona botones o la cruceta.")
    print("[AYUDA] Presiona Q o ESC para salir.\n")

    # Estados previos para imprimir solo cambios
    last_axes = {}
    last_buttons = {}
    last_hats = {}

    for js in joysticks:
        last_axes[js.get_id()] = [None] * js.get_numaxes()
        last_buttons[js.get_id()] = [None] * js.get_numbuttons()
        last_hats[js.get_id()] = [None] * js.get_numhats()

    running = True
    while running:
        for event in pygame.event.get():
            # Salida
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and (event.key == pygame.K_ESCAPE or event.key == pygame.K_q):
                running = False

            # Conectar/desconectar en caliente
            if event.type == pygame.JOYDEVICEADDED:
                js = pygame.joystick.Joystick(event.device_index)
                js.init()
                print(f"[CONECTADO] Joystick #{js.get_id()}: {js.get_name()}")
                last_axes[js.get_id()] = [None] * js.get_numaxes()
                last_buttons[js.get_id()] = [None] * js.get_numbuttons()
                last_hats[js.get_id()] = [None] * js.get_numhats()

            if event.type == pygame.JOYDEVICEREMOVED:
                jid = event.instance_id
                print(f"[DESCONECTADO] Joystick #{jid}")
                last_axes.pop(jid, None)
                last_buttons.pop(jid, None)
                last_hats.pop(jid, None)

            # Movimiento de ejes
            if event.type == pygame.JOYAXISMOTION:
                jid = event.instance_id
                axis = event.axis
                val = event.value
                # Aplica deadzone
                if abs(val) < DEADZONE:
                    val = 0.0
                val_r = round(val, ROUND_TO)
                prev = None
                if jid in last_axes and axis < len(last_axes[jid]):
                    prev = last_axes[jid][axis]
                if prev != val_r:
                    print(f"[EJE] js={jid} axis={axis} value={val_r}")
                    if jid in last_axes and axis < len(last_axes[jid]):
                        last_axes[jid][axis] = val_r

            # Botones
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"[BOTON] js={event.instance_id} button={event.button} DOWN")
                if event.instance_id in last_buttons and event.button < len(last_buttons[event.instance_id]):
                    last_buttons[event.instance_id][event.button] = 1

            if event.type == pygame.JOYBUTTONUP:
                print(f"[BOTON] js={event.instance_id} button={event.button} UP")
                if event.instance_id in last_buttons and event.button < len(last_buttons[event.instance_id]):
                    last_buttons[event.instance_id][event.button] = 0

            # Cruceta / HAT
            if event.type == pygame.JOYHATMOTION:
                print(f"[HAT] js={event.instance_id} hat={event.hat} value={event.value}")
                if event.instance_id in last_hats and event.hat < len(last_hats[event.instance_id]):
                    last_hats[event.instance_id][event.hat] = event.value

        # Opcional: imprimir estado completo cada cierto tiempo (descomenta)
        # if int(time.time()) % 5 == 0:
        #     for js in joysticks:
        #         jid = js.get_id()
        #         axes_vals = [round(js.get_axis(i), ROUND_TO) for i in range(js.get_numaxes())]
        #         print(f"[ESTADO] js={jid} axes={axes_vals}")

        # Mantén el bucle ligero
        screen.fill((20, 20, 20))
        pygame.display.flip()
        clock.tick(200)  # procesa hasta ~200 eventos por segundo

    pygame.quit()
    sys.exit(0)

if __name__ == "__main__":
    main()
