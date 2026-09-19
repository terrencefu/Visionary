#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pca(0x40);

const uint8_t PAN_CHANNEL  = 0;
const uint8_t TILT_CHANNEL = 1;

// Independent movement limits
const int PAN_MIN_US  = 400;
const int PAN_MAX_US  = 2700;
const int TILT_MIN_US = 700;
const int TILT_MAX_US = 1500;

// Home positions
const int PAN_HOME_US  = 1500;
const int TILT_HOME_US = 1000;

struct Axis {
  uint8_t channel;
  int minUs;
  int maxUs;
  int pulse;
  bool enabled;
};

Axis pan = {PAN_CHANNEL, PAN_MIN_US, PAN_MAX_US, PAN_HOME_US, false};
Axis tilt = {TILT_CHANNEL, TILT_MIN_US, TILT_MAX_US, TILT_HOME_US, false};

String inputLine;
bool discardLine = false;

void moveTo(Axis &axis, int target) {
  if (target < axis.minUs || target > axis.maxUs) {
    Serial.print("Rejected. Allowed range: ");
    Serial.print(axis.minUs);
    Serial.print("-");
    Serial.print(axis.maxUs);
    Serial.println(" us.");
    return;
  }

  // Apply the requested pulse immediately, without a software ramp.
  pca.writeMicroseconds(axis.channel, target);
  axis.pulse = target;
  axis.enabled = true;

  Serial.print(axis.channel == PAN_CHANNEL ? "Pan -> " : "Tilt -> ");
  Serial.print(target);
  Serial.println(" us.");
}

void stopAll() {
  pca.setPWM(PAN_CHANNEL, 0, 4096);
  pca.setPWM(TILT_CHANNEL, 0, 4096);

  pan.enabled = false;
  tilt.enabled = false;

  Serial.println("Signals OFF. Support the assembly.");
}

void printAxis(const char *name, const Axis &axis) {
  Serial.print(name);
  Serial.print(": commanded ");
  Serial.print(axis.pulse);
  Serial.print(" us; ");

  if (!axis.enabled) {
    Serial.println("OFF");

  } else {
    Serial.println("HOLDING");
  }
}

void printHelp() {
  Serial.println();
  Serial.println("P <us>  - pan, 400-2700 us");
  Serial.println("T <us>  - tilt, 700-1500 us");
  Serial.println("HOME    - pan 1500, tilt 1000");
  Serial.println("OFF     - disable both signals");
  Serial.println("STATUS  - show commanded positions");
  Serial.println("HELP    - show commands");
  Serial.println();
}

void handleCommand(String command) {
  command.trim();
  command.toUpperCase();

  if (command.length() == 0) {
    return;
  }

  if (command == "OFF") {
    stopAll();
    return;
  }

  if (command == "HOME") {
    moveTo(pan, PAN_HOME_US);
    moveTo(tilt, TILT_HOME_US);
    return;
  }

  if (command == "STATUS") {
    printAxis("Pan", pan);
    printAxis("Tilt", tilt);
    return;
  }

  if (command.startsWith("P ") || command.startsWith("T ")) {
    String value = command.substring(2);
    value.trim();

    bool valid = value.length() > 0 && value.length() <= 4;

    for (unsigned int i = 0; i < value.length(); i++) {
      if (value[i] < '0' || value[i] > '9') {
        valid = false;
      }
    }

    if (!valid) {
      Serial.println("Use a whole number, e.g. P 1500 or T 1000.");
      return;
    }

    int target = value.toInt();

    if (command[0] == 'P') {
      moveTo(pan, target);
    } else {
      moveTo(tilt, target);
    }

    return;
  }

  printHelp();
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!pca.begin()) {
    while (true) {
      Serial.println("ERROR: PCA9685 not detected.");
      delay(1000);
    }
  }

  // Disable all outputs before configuring.
  for (uint8_t channel = 0; channel < 16; channel++) {
    pca.setPWM(channel, 0, 4096);
  }

  pca.setPWMFreq(50);
  delay(10);

  inputLine.reserve(40);

  Serial.println("Ready. Outputs OFF.");
  Serial.println("Home: P 1500 / T 1000.");
  printHelp();
}

void loop() {
  // Non-blocking serial input.
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n') {
      if (!discardLine) {
        handleCommand(inputLine);
      }

      inputLine = "";
      discardLine = false;
    } else if (c != '\r' && !discardLine) {
      if (inputLine.length() >= 40) {
        inputLine = "";
        discardLine = true;
        Serial.println("Input too long; discarded.");
      } else {
        inputLine += c;
      }
    }
  }

}
