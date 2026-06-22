### Política de *Noise Injection* aplicada

Se implementó una política de **inyección de ruido sobre la señal de dirección (`steer`)** durante la conducción manual en CARLA, con el objetivo de generar estados perturbados y enriquecer el dataset con maniobras de recuperación.

La simulación se ejecutó a **20 Hz**, aplicando perturbaciones sobre la acción humana en la dirección.

La acción ejecutada por el vehículo se definió como:

$$
a_t^{exec} = a_t^{human} + \epsilon_t
$$

donde el término de ruido sigue una distribución gaussiana:

$$
\epsilon_t \sim \mathcal{N}(0, \sigma^2)
$$

La perturbación se mantuvo activa en ventanas temporales de aproximadamente **8 frames (~0.4 s a 20 Hz)** antes de recalcular una nueva magnitud de ruido.

---

### Registro de datos

Durante la sesión se grabaron dos fuentes de información:

- **Archivo `*.log` del recorder de CARLA**  
  Este archivo almacena la simulación completa y los **comandos efectivamente aplicados al vehículo**, es decir, la acción perturbada:

$$
a_t^{exec}
$$

- **Archivo `CSV` sincronizado por tick**  
  Se registró de forma paralela la **acción humana original sobre `steer`**, correspondiente a la corrección experta:

$$
a_t^{human}
$$

Esto fue necesario porque el archivo `*.log` conserva únicamente la acción aplicada en la simulación, mientras que para entrenamiento por imitación y DAgger interesa preservar la **acción humana de corrección** como etiqueta del dataset.

En otras palabras:

- `log` → acción aplicada con ruido
- `csv` → acción experta humana

---

### Objetivo

La finalidad de esta política fue inducir **desalineaciones controladas del vehículo** para capturar estados fuera de la distribución normal de conducción y registrar las maniobras de recuperación realizadas por el conductor.

Este enfoque permite construir un dataset orientado a **robustez y recuperación en conducción autónoma**, siguiendo una estrategia práctica cercana a **DAgger con perturbación controlada**.