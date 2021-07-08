#include <SoftwareSerial.h>
#include <Adafruit_BMP085.h>

Adafruit_BMP085 bmp;
SoftwareSerial blueSerial(10, 11); // RX, TX

#define ATMOS_OFFSET 36000
int atmos_pressure = -1;
int measure_atmos_state = 20;
long measure_atmos_tmp = 0;
int measure_atmos_times = 0;

void setup() {
  blueSerial.begin(9600);
  delay(1000);
  blueSerial.println("BMP085 setting... ");
  if (bmp.begin()) {
    blueSerial.println("BMP085 ok. ");
  } else {
    errorRightHere("Could not find a valid BMP085 sensor, check wiring!");
  }
}

void loop() {
  // blueSerial.println("aa");
  int p = bmp.readPressure() - ATMOS_OFFSET;
  // blueSerial.println("bb");
  if (measure_atmos_state >= 0) {
    measure_atmos_tmp += p;
    measure_atmos_times ++;
    if (measure_atmos_state == 0) {
      atmos_pressure = measure_atmos_tmp / measure_atmos_times;
    }
    measure_atmos_state --;
  } else {
    p -= atmos_pressure;
    p = max(0, p);
    sendP(p);
  }
}

char buf[16];
void sendP(int p) {
  unsigned long t = millis();
  blueSerial.print(t % 16384);
  blueSerial.print(',');
  blueSerial.println(p);
}

// void sendP(int p) {
//   blueSerial.print(millis());
//   blueSerial.print(',');
//   blueSerial.println(p);
// }

void errorRightHere(String msg) {
  while (true) {
    blueSerial.println(msg);
    delay(1000);
  }
}
