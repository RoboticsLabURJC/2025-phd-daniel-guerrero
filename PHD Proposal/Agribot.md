# Sistema Autónomo para la Exploración Forestal: Detección de Incendios, Monitoreo del Suelo, Recuperación Ecológica y Vigilancia de la Biodiversidad con IA

---

## Resumen
Este proyecto propone el desarrollo de un sistema autónomo para la exploración forestal, combinando cuatro funcionalidades clave:  
1. **Detección y prevención temprana de incendios forestales** mediante sensores térmicos, cámaras RGB-D, procesamiento de imágenes y algoritmos de IA.  
2. **Mapeo y monitoreo del suelo** para analizar la salud ecológica del bosque, evaluando parámetros como pH, humedad y temperatura.  
3. **Recuperación ecológica simulada**, donde el robot evaluará zonas degradadas, determinará el tipo de sustrato necesario para la recuperación del suelo y generará mapas de intervención.  
4. **Vigilancia de la biodiversidad**, detectando amenazas como la tala clandestina, la presencia de enfermedades en la flora y la evaluación continua de zonas reforestadas para garantizar la salud de los árboles.  

El sistema se desarrollará en simuladores avanzados (**Gazebo, CARLA y Open3DE**), permitiendo probar y optimizar los algoritmos de navegación, detección y análisis del terreno en entornos forestales complejos.  

El proyecto tendrá un enfoque en la sostenibilidad ambiental, la conservación ecológica y la protección contra desastres naturales.

---

## Objetivos

### Objetivo General
Desarrollar un sistema autónomo para la exploración forestal que combine la detección temprana de incendios, el monitoreo del suelo, la evaluación de zonas degradadas para la recuperación ecológica y la vigilancia de la biodiversidad, validando su desempeño en simuladores avanzados.

### Objetivos Específicos
1. Diseñar un sistema de navegación autónoma basado en **SLAM 3D** para la exploración de entornos forestales irregulares.  
2. Integrar un módulo de detección de incendios con **sensores térmicos, cámaras RGB-D e IA** para identificar puntos calientes o riesgos de combustión.  
3. Implementar un sistema de monitoreo del suelo con sensores de **pH, humedad y temperatura**, generando mapas georreferenciados.  
4. Incorporar un módulo de **recuperación ecológica** que detecte zonas forestales degradadas y analice el tipo de sustrato necesario para su restauración.  
5. Desarrollar un sistema de **vigilancia de la biodiversidad** para detectar problemas como:  
    - Tala clandestina.  
    - Enfermedades visibles en la flora.  
    - Detección de cambios en la cobertura vegetal.  
    - Monitoreo continuo de zonas reforestadas para garantizar la salud de los árboles.  
6. Validar el desempeño del robot en simuladores avanzados (Gazebo, CARLA y Open3DE), evaluando la precisión de navegación, detección de incendios, monitoreo del suelo y vigilancia de la biodiversidad.  
7. Comparar el rendimiento del sistema con otras tecnologías de monitoreo ambiental en simulación.  

---

## Justificación
Los incendios forestales, la degradación del suelo, la tala ilegal y la propagación de enfermedades vegetales representan amenazas graves para los ecosistemas, reduciendo la biodiversidad y afectando la capacidad de recuperación del bosque. Además, las zonas reforestadas requieren un monitoreo continuo para garantizar la supervivencia de los árboles jóvenes y el éxito de la restauración.

Este proyecto busca abordar estos problemas mediante un robot forestal autónomo capaz de:  
- **Prevenir incendios** al identificar puntos calientes tempranamente.  
- **Monitorear la salud del suelo**, facilitando la toma de decisiones para la conservación ecológica.  
- **Identificar zonas degradadas**, generando mapas de intervención para la recuperación del terreno.  
- **Detectar actividades ilegales** como la tala clandestina o la pérdida repentina de vegetación.  
- **Identificar enfermedades visibles** en la flora mediante IA.  
- **Garantizar la salud de las zonas reforestadas**, evaluando la supervivencia de los árboles jóvenes.  
- **Optimizar la gestión forestal** mediante la automatización, reduciendo costos operativos.  
- **Facilitar la exploración en zonas de difícil acceso**, aumentando la cobertura de monitoreo.  

---

## Metodología

### Fase 1: Revisión del Estado del Arte (3 meses)
- Análisis de técnicas de navegación autónoma en entornos forestales.  
- Estudio de sistemas de detección de incendios con visión térmica.  
- Revisión de tecnologías para la medición del suelo (pH, humedad, temperatura) y análisis de sustratos.  
- Estudio de algoritmos de detección de tala clandestina y enfermedades vegetales.  
- Revisión de modelos de simulación para evaluar la regeneración forestal.  
- Selección de herramientas de simulación (**Gazebo, CARLA, Open3DE**) y frameworks de IA (**PyTorch, TensorFlow**).  

---

### Fase 2: Simulación (6 meses)
- Creación de un entorno forestal virtual con obstáculos naturales (árboles, rocas, terrenos irregulares).  
- Modelado del robot con sensores LiDAR, cámaras RGB-D, térmicas y módulos de detección de incendios.  
- Simulación de incendios controlados para evaluar la capacidad de detección.  
- Generación de mapas del suelo con datos simulados de pH, humedad y temperatura.  
- Modelado de zonas degradadas con diferentes niveles de daño ecológico.  
- Simulación de tala clandestina y detección de enfermedades en la flora.  
- Creación de zonas reforestadas con árboles jóvenes y generación de un modelo de evaluación de salud basado en color y densidad.  

---

### Fase 3: Diseño del Sistema de Navegación (9 meses)

#### Navegación autónoma
- Implementación de **SLAM 3D** para mapeo y localización en entornos forestales.  
- Algoritmos de evasión dinámica (DWA o MPC) para evitar árboles, rocas y desniveles.  
- Uso de **redes neuronales convolucionales (CNN)** para identificar caminos transitables.  

#### Detección de incendios
- Integración de **sensores térmicos simulados** para la detección de puntos calientes.  
- IA para la identificación automática de llamas o humo en imágenes térmicas.  
- Generación de alertas con geolocalización en el entorno virtual.  

#### Monitoreo del suelo
- Integración de sensores simulados de **pH, humedad y temperatura**.  
- Generación de mapas georreferenciados del estado del suelo.  
- Exportación de los datos a GIS para análisis geoespacial.  

#### Módulo de recuperación ecológica
- Análisis de zonas degradadas en el entorno simulado.  
- Clasificación del tipo de degradación:  
    - Baja: pérdida de nutrientes.  
    - Media: erosión leve.  
    - Alta: pérdida de vegetación y compactación del suelo.  
- Determinación del tipo de sustrato necesario para la recuperación:  
    - Compost o materia orgánica.  
    - Riego controlado.  
    - Reforestación simulada.  
- Generación de un mapa de intervención con recomendaciones de recuperación.  

#### Vigilancia de la biodiversidad
- Detección de **tala clandestina** mediante comparación de imágenes satelitales simuladas con el entorno actual.  
- Identificación de **enfermedades vegetales** a través de análisis de color, textura y patrones en las hojas.  
- Monitoreo continuo de **zonas reforestadas** evaluando:  
    - Supervivencia de árboles jóvenes.  
    - Crecimiento y densidad de la vegetación.  
    - Presencia de deficiencias visibles.  

---

### Fase 4: Optimización (6 meses)
- Ajuste fino de los algoritmos de navegación para mejorar la evasión de obstáculos.  
- Optimización del modelo de detección de incendios con **aprendizaje por refuerzo**.  
- Validación del módulo de recuperación ecológica con múltiples escenarios de degradación.  
- Pruebas de detección de tala y enfermedades simuladas.  

---

### Fase 5: Análisis y Documentación (6 meses)
- Comparación del sistema con otras tecnologías de monitoreo forestal en simulación.  
- Análisis de datos para evaluar la eficiencia, precisión y robustez del sistema.  
- Publicación de resultados en revistas científicas y redacción de la tesis doctoral.  

---

## Recursos Necesarios

### Software
- **Simulación:** Gazebo, CARLA, Open3DE.  
- **Frameworks de IA:** PyTorch, TensorFlow.  
- **Navegación:** ROS2 (Robot Operating System).  
- **Visualización:** RViz o Foxglove.  
- **GIS:** QGIS para el análisis geoespacial.  

---

## Impacto Esperado

### Científico
- Avance en la navegación autónoma en entornos forestales no estructurados.  

### Ambiental
- Detección temprana de incendios.  
- Monitoreo de zonas reforestadas.  
- Protección de la biodiversidad.
