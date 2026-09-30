#include <Servo.h>

const unsigned long BAUD_RATE = 115200;
const byte FINGER_COUNT = 5;

// Thumb, Index, Middle, Ring, Pinky. Must match SERVO_PINS in openCVRobotics.py.
const byte SERVO_PINS[FINGER_COUNT] = {10, 9, 6, 5, 3};
Servo servos[FINGER_COUNT];

char inputBuffer[40];
byte inputLength = 0;

void applyFrame(char *frame) {
  int angles[FINGER_COUNT];
  int parsed = sscanf(
    frame,
    "A,%d,%d,%d,%d,%d",
    &angles[0],
    &angles[1],
    &angles[2],
    &angles[3],
    &angles[4]
  );

  if (parsed != FINGER_COUNT) {
    return;
  }

  for (byte finger = 0; finger < FINGER_COUNT; finger++) {
    servos[finger].write(constrain(angles[finger], 0, 180));
  }
}

void setup() {
  Serial.begin(BAUD_RATE);

  for (byte finger = 0; finger < FINGER_COUNT; finger++) {
    servos[finger].attach(SERVO_PINS[finger]);
    servos[finger].write(90);
  }
}

void loop() {
  while (Serial.available() > 0) {
    char incoming = Serial.read();

    if (incoming == '\n') {
      inputBuffer[inputLength] = '\0';
      applyFrame(inputBuffer);
      inputLength = 0;
    } else if (incoming != '\r') {
      if (inputLength < sizeof(inputBuffer) - 1) {
        inputBuffer[inputLength++] = incoming;
      } else {
        inputLength = 0;
      }
    }
  }
}