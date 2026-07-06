#include <Wire.h>
#include <AS5600.h>

AS5600 as5600;

void setup() {
  // Inicializa comunicación serial a la misma velocidad que probamos
  Serial.begin(115200);
  
  // Inicializa el bus I2C en los pines estándar del ESP32 (SDA=21, SCL=22)
  Wire.begin(21, 22);
  
  // Inicializa el objeto del sensor
  as5600.begin();

  // Verificación rápida del sensor
  if (!as5600.isConnected()) {
    Serial.println("Error: No se detecta el AS5600. Verifica el cableado.");
    while (1); // Detiene la ejecución si hay falla física
  }
}

void loop() {
  // El AS5600 entrega una resolución de 12 bits (0 a 4095)
  // Convertimos esa lectura directa a un valor en grados (0 a 360)
  float raw_angle = as5600.readAngle();
  float grados = raw_angle * 360.0 / 4096.0;

  // Imprime con formato de etiqueta explícito para el Serial Plotter
  Serial.print("Angulo:");
  Serial.println(grados);

  // Muestreo constante cada 30 ms
  delay(30);
}