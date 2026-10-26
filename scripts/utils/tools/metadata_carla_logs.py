import os
import glob
import re
import carla
from datetime import datetime

# ================= CONFIGURACIÓN =================
CARPETA_LOGS = "/home/daniel/code/2025-phd-daniel-guerrero/scripts/dataset_generation/logs_crudos" # Cambia esto a la ruta de tus .log
ARCHIVO_MD = "CATALOGO_LOGS.md"
# =================================================

def bytes_a_mb(bytes_size):
    """Convierte bytes a Megabytes con 2 decimales"""
    return round(bytes_size / (1024 * 1024), 2)

def main():
    print(f"🔍 Buscando archivos .log en: {CARPETA_LOGS}")
    archivos_log = glob.glob(os.path.join(CARPETA_LOGS, "*.log"))
    
    if not archivos_log:
        print("⚠️ No se encontraron archivos .log en la carpeta especificada.")
        return

    # Intentar conectar con el simulador CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        print("✅ Conectado al simulador CARLA. Procesando metadatos...")
    except RuntimeError:
        print("❌ ERROR: No se pudo conectar a CARLA. ¿Está el simulador abierto?")
        return

    datos_catalogo = []

    for log_path in archivos_log:
        nombre_archivo = os.path.basename(log_path)
        ruta_absoluta = os.path.abspath(log_path)
        peso_mb = bytes_a_mb(os.path.getsize(ruta_absoluta))

        try:
            # Extraer info cruda del archivo (False = solo resumen, sin detalles de frames)
            info_cruda = client.show_recorder_file_info(ruta_absoluta, False)
            
            # Usar Expresiones Regulares para cazar los datos específicos
            mapa = re.search(r'Map:\s*(.+)', info_cruda)
            fecha = re.search(r'Date:\s*(.+)', info_cruda)
            frames = re.search(r'Frames:\s*(\d+)', info_cruda)
            duracion = re.search(r'Duration:\s*([\d\.]+)', info_cruda)

            # Limpiar los resultados
            mapa_str = mapa.group(1).strip() if mapa else "Desconocido"
            fecha_str = fecha.group(1).strip() if fecha else "Desconocida"
            frames_str = frames.group(1) if frames else "0"
            duracion_str = f"{float(duracion.group(1)):.2f}" if duracion else "0.00"

            datos_catalogo.append({
                "Archivo": nombre_archivo,
                "Mapa": mapa_str,
                "Fecha": fecha_str,
                "Frames": frames_str,
                "Duración (s)": duracion_str,
                "Peso (MB)": f"{peso_mb} MB"
            })
            print(f"  -> Procesado: {nombre_archivo}")

        except Exception as e:
            print(f"  -> ⚠️ Error al leer {nombre_archivo}: El archivo podría estar corrupto.")
            datos_catalogo.append({
                "Archivo": nombre_archivo,
                "Mapa": "ERROR", "Fecha": "ERROR", "Frames": "ERROR", 
                "Duración (s)": "ERROR", "Peso (MB)": f"{peso_mb} MB"
            })

    # ================= GENERACIÓN DEL ARCHIVO MARKDOWN =================
    fecha_actualizacion = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md_content = f"# 📊 Catálogo de Telemetría CARLA\n\n"
    md_content += f"**Última actualización:** {fecha_actualizacion}\n\n"
    md_content += f"Este documento lista todos los archivos `.log` de entrenamiento disponibles en el directorio.\n\n"
    
    # Cabecera de la tabla
    md_content += "| Nombre del Archivo | Mapa (Town) | Fecha de Grabación | Frames | Duración (seg) | Peso |\n"
    md_content += "|--------------------|-------------|--------------------|--------|----------------|------|\n"

    # Filas de la tabla
    for d in sorted(datos_catalogo, key=lambda x: x["Archivo"]):
        md_content += f"| `{d['Archivo']}` | {d['Mapa']} | {d['Fecha']} | {d['Frames']} | {d['Duración (s)']}s | {d['Peso (MB)']} |\n"

    # Guardar el archivo
    ruta_md = os.path.join(CARPETA_LOGS, ARCHIVO_MD)
    with open(ruta_md, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n🎉 ¡Catálogo actualizado con éxito en: {ruta_md}!")

if __name__ == "__main__":
    main()