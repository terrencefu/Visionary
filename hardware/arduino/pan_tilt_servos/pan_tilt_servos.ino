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

// Motion speed: 100 us of travel takes 2 seconds.
// Increase MS_PER_US for slower movement.
const unsigned long MS_PER_US = 20;
const unsigned long MIN_MOVE_MS = 800;
const unsigned long UPDATE_MS = 20;

struct Axis {
  uint8_t channel;
  int minUs;
  int maxUs;
  int pulse;
  int startPulse;
  int target;
  bool enabled;
  bool moving;
  unsigned long started;
  unsigned long duration;
};

Axis pan = {
  PAN_CHANNEL,
  PAN_MIN_US,
  PAN_MAX_US,
  PAN_HOME_US,
  PAN_HOME_US,
  PAN_HOME_US,
  false,
  false,
  0,
  0
};

Axis tilt = {
  TILT_CHANNEL,
  TILT_MIN_US,
  TILT_MAX_US,
  TILT_HOME_US,
  TILT_HOME_US,
  TILT_HOME_US,
  false,
  false,
  0,
  0
};

String inputLine;
bool discardLine = false;
unsigned long lastUpdate = 0;

// S-curve: zero speed and acceleration at both ends.
float ease(float t) {
  return t * t * t * (10.0f + t * (-15.0f + 6.0f * t));
}

void updateAxis(Axis &axis, unsigned long now) {
  if (!axis.enabled || !axis.moving) {
    return;
  }

  unsigned long elapsed = now - axis.started;
  int nextPulse;

  if (elapsed >= axis.duration) {
    nextPulse = axis.target;
    axis.moving = false;
  } else {
    float t = (float)elapsed / axis.duration;
    float position =
      axis.startPulse +
      (axis.target - axis.startPulse) * ease(t);

    nextPulse = (int)(position + 0.5f);
  }

  if (nextPulse != axis.pulse) {
    axis.pulse = nextPulse;
    pca.writeMicroseconds(axis.channel, axis.pulse);
  }
}

void moveTo(Axis &axis, int target) {
  if (target < axis.minUs || target > axis.maxUs) {
    Serial.print("Rejected. Allowed range: ");
    Serial.print(axis.minUs);
    Serial.print("-");
    Serial.print(axis.maxUs);
    Serial.println(" us.");
    return;
  }

  // Finish the current curve before starting another.
  if (axis.moving) {
    Serial.println("Still moving. Wait, or send OFF.");
    return;
  }

  if (!axis.enabled) {
    // No position feedback: first enable cannot be ramped
    // reliably from the actual physical starting angle.
    axis.pulse = target;
    axis.startPulse = target;
    axis.target = target;
    axis.enabled = true;
    axis.moving = false;

    pca.writeMicroseconds(axis.channel, target);

    Serial.print(axis.channel == PAN_CHANNEL ? "Pan" : "Tilt");
    Serial.println(" initially positioned; now holding.");
    return;
  }

  int distance = abs(target - axis.pulse);

  if (distance == 0) {
    Serial.println("Already commanding that position.");
    return;
  }

  axis.startPulse = axis.pulse;
  axis.target = target;
  axis.duration = (unsigned long)distance * MS_PER_US;

  if (axis.duration < MIN_MOVE_MS) {
    axis.duration = MIN_MOVE_MS;
  }

  axis.started = millis();
  axis.moving = true;

  Serial.print(axis.channel == PAN_CHANNEL ? "Pan -> " : "Tilt -> ");
  Serial.print(target);
  Serial.print(" us over ");
  Serial.print(axis.duration / 1000.0f, 2);
  Serial.println(" seconds.");
}

void stopAll() {
  pca.setPWM(PAN_CHANNEL, 0, 4096);
  pca.setPWM(TILT_CHANNEL, 0, 4096);

  pan.enabled = false;
  tilt.enabled = false;
  pan.moving = false;
  tilt.moving = false;

  pan.target = pan.pulse;
  tilt.target = tilt.pulse;

  Serial.println("Signals OFF. Support the assembly.");
}

void printAxis(const char *name, const Axis &axis) {
  Serial.print(name);
  Serial.print(": commanded ");
  Serial.print(axis.pulse);
  Serial.print(" us; target ");
  Serial.print(axis.target);
  Serial.print(" us; ");

  if (!axis.enabled) {
    Serial.println("OFF");
  } else if (axis.moving) {
    Serial.println("MOVING");
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
    if (pan.moving || tilt.moving) {
      Serial.println("Wait for movement to finish before HOME.");
      return;
    }

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

  unsigned long now = millis();

  if (now - lastUpdate >= UPDATE_MS) {
    lastUpdate = now;
    updateAxis(pan, now);
    updateAxis(tilt, now);
  }
}