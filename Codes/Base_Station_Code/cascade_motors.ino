#include <Servo.h>

Servo servo1;   // NOW pin 10
Servo servo2;   // NOW pin 9

int angle1 = 0;
int angle2 = 0;

int totalAngle = 0;
int currentAngle = 0;   // <-- keeps the actual last servo angle (0..180)
bool stopMotion = false;
bool angleSent = false;

const int stepDelay = 400;
const int stepSize  = 1;

void setup() {
  Serial.begin(9600);

  servo1.attach(10);   // SWAPPED
  servo2.attach(9);    // SWAPPED

  servo1.write(0);
  servo2.write(0);
  currentAngle = 0;
  angle1 = 0;
  angle2 = 0;
  delay(300);

  Serial.println("READY");
}

void loop() {

  checkSerialCommand();

  if (stopMotion) {
    if (!angleSent) {
      // Send the last reported servo angle (not the accumulated total)
      Serial.print("ANGLE:");
      Serial.println(currentAngle);
      Serial.flush();
      angleSent = true;
    }
    return;
  }

  runScanPattern();
}

// ======================================================
// READ STOP / RUN COMMAND
// ======================================================
void checkSerialCommand() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  if (cmd.equalsIgnoreCase("STOP")) {
    stopMotion = true;
    angleSent = false;   // ensure we will send angle once when stopped
  } else if (cmd.equalsIgnoreCase("RUN")) {
    // resume scanning
    stopMotion = false;
    angleSent = false;
  }
}

// ======================================================
// SCAN ROUTINE
// ======================================================
void runScanPattern() {

  sweep(servo1, angle1, 0, 180);
  if (stopMotion) return;

  sweep(servo2, angle2, 0, 180);
  if (stopMotion) return;

  sweep(servo2, angle2, 180, 0);
  if (stopMotion) return;

  sweep(servo1, angle1, 180, 0);
  if (stopMotion) return;
}

// ======================================================
// SWEEP FUNCTION
// ======================================================
void sweep(Servo &servo, int &angleVar, int start, int end) {

  int direction = (end > start) ? stepSize : -stepSize;

  for (int a = start;
       (direction > 0) ? a <= end : a >= end;
       a += direction)
  {
    checkSerialCommand();
    if (stopMotion) return;

    servo.write(a);
    angleVar = a;
    currentAngle = a;    // update the shared current angle (the one we will report)

    totalAngle += abs(direction);

    delay(stepDelay);
  }
}